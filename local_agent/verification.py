"""
Worker Verification Flow — Capture corrections from human review.

Every invoice gets verified by a worker.
Worker approves correct fields, corrects wrong ones.
Corrections are captured and fed to the feedback processor.
"""
import time
import logging
from typing import Dict, List, Optional
from enum import Enum

import storage

logger = logging.getLogger(__name__)


class VerificationStatus(str, Enum):
    PENDING = "pending"
    IN_REVIEW = "in_review"
    VERIFIED = "verified"
    SKIPPED = "skipped"

# Header fields to compare (flat scalars)
HEADER_SCALAR_FIELDS = [
    'InvoiceNo', 'Date', 'InvoiceCurrency', 'FreightTerms',
    'IncoTerms', 'TermsOfPayment',
]
# Header fields that are dicts with Name/Address
HEADER_DICT_FIELDS = ['Exporter', 'Importer']
HEADER_DICT_SUBFIELDS = ['Name', 'Address']

# Line item fields to compare
LINE_ITEM_FIELDS = [
    'PartNo', 'ItemCode', 'ItemDescription', 'Quantity',
    'UnitOfQty', 'UnitPrice', 'RITC', 'CountryOfOrigin',
]

# Fields where comparison should be case-insensitive
CASE_INSENSITIVE_FIELDS = {
    'InvoiceCurrency', 'UnitOfQty', 'CountryOfOrigin', 'IncoTerms', 'FreightTerms',
}


def _normalize_value(value, field_name: str = '') -> str:
    """Normalize a field value for comparison.

    - None and empty string are treated as equivalent (both become '')
    - Strips whitespace
    - Case-insensitive for currency, UOM, country, incoterms, freight terms
    """
    if value is None:
        return ''
    s = str(value).strip()
    if not s:
        return ''
    if field_name in CASE_INSENSITIVE_FIELDS:
        s = s.lower()
    return s


class VerificationManager:
    """Manages the worker verification workflow."""

    def __init__(self):
        pass

    def create_verification(self, job_id: str, agent_result: Dict,
                            format_key: str = None) -> Dict:
        """
        Create a verification task from an agent result.

        Determines review level based on quality_score:
        - score >= 0.95: AUTO_APPROVE (worker spot-checks)
        - score 0.7-0.95: QUICK_REVIEW (flagged fields highlighted)
        - score < 0.7: FULL_REVIEW (worker checks every field)

        Returns the verification dict and saves to VERIFICATION_DIR/{job_id}.json.
        """
        score = 0.0
        try:
            score = float(agent_result.get('quality_score', 0))
        except (TypeError, ValueError):
            pass

        if score >= 0.95:
            review_level = 'auto_approve'
        elif score >= 0.7:
            review_level = 'quick_review'
        else:
            review_level = 'full_review'

        verification = {
            'job_id': job_id,
            'status': VerificationStatus.PENDING.value,
            'review_level': review_level,
            'agent_result': agent_result,
            'worker_result': None,
            'format_key': format_key,
            'created_at': time.time(),
        }

        self._save(job_id, verification)
        logger.info(
            f"Verification created: job={job_id} review_level={review_level} "
            f"score={score:.3f}"
        )
        return verification

    def submit_verification(self, job_id: str, worker_result: Dict,
                            worker_id: str = None,
                            worker_notes: str = None) -> Dict:
        """
        Submit worker-verified result.

        Calculates corrections by diffing agent_result vs worker_result.
        Updates and saves the verification file.
        """
        verification = self.get_verification(job_id)
        if verification is None:
            raise FileNotFoundError(f"No verification found for job {job_id}")

        agent_result = verification.get('agent_result', {})
        corrections = self._diff_results(agent_result, worker_result)
        accuracy = self._calculate_accuracy(corrections)

        verification['status'] = VerificationStatus.VERIFIED.value
        verification['worker_result'] = worker_result
        verification['corrections'] = corrections
        verification['accuracy'] = accuracy
        verification['worker_id'] = worker_id
        verification['worker_notes'] = worker_notes
        verification['verified_at'] = time.time()

        self._save(job_id, verification)
        logger.info(
            f"Verification submitted: job={job_id} worker={worker_id} "
            f"accuracy={accuracy.get('field_accuracy', 0):.3f} "
            f"corrected={accuracy.get('corrected_fields', 0)}"
        )

        return {
            'job_id': job_id,
            'status': verification['status'],
            'corrections': corrections,
            'accuracy': accuracy,
            'worker_id': worker_id,
            'worker_notes': worker_notes,
            'verified_at': verification['verified_at'],
        }

    def get_verification(self, job_id: str) -> Optional[Dict]:
        """Load verification state for a job."""
        return storage.get_doc('verifications', job_id)

    def list_pending(self) -> List[Dict]:
        """List all pending verification tasks."""
        all_docs = storage.query_docs('verifications', 'status', 'pending')
        pending = []
        for data in all_docs:
            if data.get('status') == VerificationStatus.PENDING.value:
                pending.append({
                    'job_id': data.get('job_id', ''),
                    'filename': data.get('agent_result', {}).get('filename', ''),
                    'review_level': data.get('review_level', ''),
                    'confidence': data.get('agent_result', {}).get('quality_score', 0),
                    'created_at': data.get('created_at', 0),
                })
        return pending

    # ── Private helpers ───────────────────────────────────────

    def _save(self, job_id: str, data: Dict):
        """Save verification."""
        storage.put_doc('verifications', job_id, data)

    def _diff_results(self, agent_result: Dict, worker_result: Dict) -> List[Dict]:
        """
        Compare agent vs worker results field by field.

        Compares header scalar fields, header dict fields (Exporter/Importer),
        and line items (position-based matching by index).

        Returns a list of correction records.
        """
        corrections = []

        # 1. Header scalar fields
        for field in HEADER_SCALAR_FIELDS:
            agent_val = agent_result.get(field)
            worker_val = worker_result.get(field)
            action = self._compare_values(agent_val, worker_val, field)
            corrections.append({
                'item_index': None,
                'field': field,
                'agent_value': agent_val,
                'worker_value': worker_val,
                'action': action,
            })

        # 2. Header dict fields (Exporter, Importer)
        for field in HEADER_DICT_FIELDS:
            agent_dict = agent_result.get(field) or {}
            worker_dict = worker_result.get(field) or {}
            if not isinstance(agent_dict, dict):
                agent_dict = {}
            if not isinstance(worker_dict, dict):
                worker_dict = {}

            for sub in HEADER_DICT_SUBFIELDS:
                full_field = f'{field}.{sub}'
                agent_val = agent_dict.get(sub)
                worker_val = worker_dict.get(sub)
                action = self._compare_values(agent_val, worker_val, full_field)
                corrections.append({
                    'item_index': None,
                    'field': full_field,
                    'agent_value': agent_val,
                    'worker_value': worker_val,
                    'action': action,
                })

        # 3. Line items — position-based matching
        agent_items = agent_result.get('LineItems') or []
        worker_items = worker_result.get('LineItems') or []
        if not isinstance(agent_items, list):
            agent_items = []
        if not isinstance(worker_items, list):
            worker_items = []

        max_len = max(len(agent_items), len(worker_items))

        for idx in range(max_len):
            agent_item = agent_items[idx] if idx < len(agent_items) else None
            worker_item = worker_items[idx] if idx < len(worker_items) else None

            if agent_item is None and worker_item is not None:
                # Worker added an item the agent missed
                for field in LINE_ITEM_FIELDS:
                    corrections.append({
                        'item_index': idx,
                        'field': field,
                        'agent_value': None,
                        'worker_value': worker_item.get(field),
                        'action': 'added',
                    })
                continue

            if worker_item is None and agent_item is not None:
                # Worker removed an item the agent hallucinated
                for field in LINE_ITEM_FIELDS:
                    corrections.append({
                        'item_index': idx,
                        'field': field,
                        'agent_value': agent_item.get(field),
                        'worker_value': None,
                        'action': 'removed',
                    })
                continue

            # Both exist — compare field by field
            if not isinstance(agent_item, dict):
                agent_item = {}
            if not isinstance(worker_item, dict):
                worker_item = {}

            for field in LINE_ITEM_FIELDS:
                agent_val = agent_item.get(field)
                worker_val = worker_item.get(field)
                action = self._compare_values(agent_val, worker_val, field)
                corrections.append({
                    'item_index': idx,
                    'field': field,
                    'agent_value': agent_val,
                    'worker_value': worker_val,
                    'action': action,
                })

        return corrections

    def _compare_values(self, agent_val, worker_val, field_name: str) -> str:
        """Compare two values after normalization. Returns action string."""
        norm_agent = _normalize_value(agent_val, field_name)
        norm_worker = _normalize_value(worker_val, field_name)

        if norm_agent == norm_worker:
            return 'unchanged'
        return 'corrected'

    def _calculate_accuracy(self, corrections: List[Dict]) -> Dict:
        """
        Calculate accuracy metrics from corrections list.

        Counts by action type, computes overall field_accuracy,
        and per_field_accuracy (what % of each field name was correct).
        """
        total = len(corrections)
        if total == 0:
            return {
                'total_fields': 0,
                'correct_fields': 0,
                'corrected_fields': 0,
                'added_fields': 0,
                'removed_fields': 0,
                'field_accuracy': 1.0,
                'per_field_accuracy': {},
            }

        correct = sum(1 for c in corrections if c['action'] == 'unchanged')
        corrected = sum(1 for c in corrections if c['action'] == 'corrected')
        added = sum(1 for c in corrections if c['action'] == 'added')
        removed = sum(1 for c in corrections if c['action'] == 'removed')

        field_accuracy = correct / total if total > 0 else 0.0

        # Per-field accuracy
        field_counts: Dict[str, Dict[str, int]] = {}
        for c in corrections:
            fname = c['field']
            if fname not in field_counts:
                field_counts[fname] = {'correct': 0, 'total': 0}
            field_counts[fname]['total'] += 1
            if c['action'] == 'unchanged':
                field_counts[fname]['correct'] += 1

        per_field = {}
        for fname, counts in field_counts.items():
            if counts['total'] > 0:
                per_field[fname] = round(counts['correct'] / counts['total'], 3)

        return {
            'total_fields': total,
            'correct_fields': correct,
            'corrected_fields': corrected,
            'added_fields': added,
            'removed_fields': removed,
            'field_accuracy': round(field_accuracy, 3),
            'per_field_accuracy': per_field,
        }
