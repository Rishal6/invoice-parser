"""
Feedback Processor — Captures worker corrections, creates patterns.

Diffs agent result vs worker-verified result.
Uses LLM to understand WHY corrections were made.
Creates new patterns for the pattern library.
"""
import json
import os
import re
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional

import boto3
from botocore.config import Config

import storage

logger = logging.getLogger(__name__)

REGION = os.environ.get('AWS_REGION', 'us-east-1')
FEEDBACK_MODEL = os.environ.get('FEEDBACK_MODEL', 'us.anthropic.claude-sonnet-4-5-20250929-v1:0')

bedrock = boto3.client('bedrock-runtime', region_name=REGION, config=Config(
    read_timeout=300, retries={'max_attempts': 3}
))

# Line item fields to compare
ITEM_FIELDS = [
    'PartNo', 'ItemCode', 'ItemDescription', 'Quantity',
    'UnitOfQty', 'UnitPrice', 'RITC', 'CountryOfOrigin',
]

# Header fields to compare
HEADER_FIELDS = [
    'InvoiceNo', 'Date', 'InvoiceCurrency', 'FreightTerms',
    'IncoTerms', 'TermsOfPayment',
]


class FeedbackProcessor:
    def __init__(self, pattern_library=None):
        """
        Args:
            pattern_library: PatternLibrary instance from patterns.py
                             (imported and passed in to avoid circular imports)
        """
        self.pattern_library = pattern_library

    def diff_results(self, agent_result: Dict, worker_result: Dict) -> List[Dict]:
        """
        Compare agent extraction vs worker-verified extraction.

        Args:
            agent_result: Original agent extraction (the ExtractionResult)
            worker_result: Worker-verified version (same structure, with corrections)

        Returns: List of corrections found between the two results.
        """
        corrections = []

        # Diff header fields
        for field in HEADER_FIELDS:
            agent_val = agent_result.get(field)
            worker_val = worker_result.get(field)
            if _normalize(agent_val) != _normalize(worker_val):
                action = _classify_action(agent_val, worker_val)
                corrections.append({
                    'item_index': None,
                    'field': field,
                    'agent_value': agent_val,
                    'worker_value': worker_val,
                    'action': action,
                })

        # Diff nested header fields (Exporter, Importer)
        for entity in ('Exporter', 'Importer'):
            agent_entity = agent_result.get(entity) or {}
            worker_entity = worker_result.get(entity) or {}
            for sub in ('Name', 'Address'):
                agent_val = agent_entity.get(sub)
                worker_val = worker_entity.get(sub)
                if _normalize(agent_val) != _normalize(worker_val):
                    action = _classify_action(agent_val, worker_val)
                    corrections.append({
                        'item_index': None,
                        'field': f'{entity}.{sub}',
                        'agent_value': agent_val,
                        'worker_value': worker_val,
                        'action': action,
                    })

        # Diff line items
        agent_items = agent_result.get('LineItems', [])
        worker_items = worker_result.get('LineItems', [])

        # Match items by index (worker edits in-place)
        max_items = max(len(agent_items), len(worker_items))
        for i in range(max_items):
            agent_item = agent_items[i] if i < len(agent_items) else {}
            worker_item = worker_items[i] if i < len(worker_items) else {}

            # Item added by worker (agent didn't have it)
            if i >= len(agent_items):
                for field in ITEM_FIELDS:
                    val = worker_item.get(field)
                    if val is not None:
                        corrections.append({
                            'item_index': i,
                            'field': field,
                            'agent_value': None,
                            'worker_value': val,
                            'action': 'added',
                        })
                continue

            # Item removed by worker
            if i >= len(worker_items):
                corrections.append({
                    'item_index': i,
                    'field': '_entire_item',
                    'agent_value': _item_summary(agent_item),
                    'worker_value': None,
                    'action': 'removed',
                })
                continue

            # Compare each field
            for field in ITEM_FIELDS:
                agent_val = agent_item.get(field)
                worker_val = worker_item.get(field)
                if _normalize(agent_val) != _normalize(worker_val):
                    action = _classify_action(agent_val, worker_val)
                    corrections.append({
                        'item_index': i,
                        'field': field,
                        'agent_value': agent_val,
                        'worker_value': worker_val,
                        'action': action,
                    })

        return corrections

    def process_feedback(self, job_id: str, corrections: List[Dict],
                         source_text: str, invoice_header: Dict,
                         format_key: str = None,
                         worker_notes: str = None) -> Dict:
        """
        Process worker corrections into patterns.

        Steps:
        1. Log the feedback (always)
        2. If corrections exist, use LLM to understand WHY
        3. LLM creates pattern from corrections
        4. Save pattern to library
        """
        feedback_id = f"fb_{job_id[:8]}_{int(time.time()) % 1000:03d}"
        patterns_created = []

        # If there are non-trivial corrections, generate a pattern
        if corrections and not _all_trivial(corrections):
            try:
                pattern = self._generate_pattern_from_corrections(
                    corrections, source_text, invoice_header,
                    format_key, worker_notes,
                )
                if pattern:
                    pattern_id = pattern.get('pattern_id', feedback_id)
                    # Save to library if available
                    if self.pattern_library and hasattr(self.pattern_library, 'add_pattern'):
                        self.pattern_library.add_pattern(pattern)
                    patterns_created.append(pattern_id)
            except Exception as e:
                logger.error(f"Pattern generation failed for {job_id}: {e}")

        # Always log the feedback
        self._log_feedback(job_id, corrections, patterns_created, worker_notes)

        return {
            'corrections_count': len(corrections),
            'patterns_created': len(patterns_created),
            'pattern_ids': patterns_created,
            'feedback_id': feedback_id,
        }

    def _generate_pattern_from_corrections(self, corrections: List[Dict],
                                            source_text: str,
                                            invoice_header: Dict,
                                            format_key: str,
                                            worker_notes: str = None) -> Optional[Dict]:
        """
        Use LLM to create a reusable pattern from corrections.
        Returns None if corrections are trivial or LLM call fails.
        """
        # Build corrections description
        correction_lines = []
        for c in corrections:
            prefix = f"Item {c['item_index']}, " if c.get('item_index') is not None else ""
            correction_lines.append(
                f'- {prefix}{c["field"]}: was "{c["agent_value"]}", now "{c["worker_value"]}"'
            )
        corrections_text = "\n".join(correction_lines)

        # Build invoice context
        exporter = ""
        if isinstance(invoice_header.get('Exporter'), dict):
            exporter = invoice_header['Exporter'].get('Name', '')
        elif isinstance(invoice_header.get('Exporter'), str):
            exporter = invoice_header['Exporter']

        currency = invoice_header.get('InvoiceCurrency', '')

        from knowledge import FORMAT_NAMES
        format_name = FORMAT_NAMES.get(format_key, format_key or 'Unknown')

        notes_section = f'\nWorker notes: "{worker_notes}"' if worker_notes else ""

        user_prompt = (
            f"A worker corrected these fields on an invoice extraction:\n\n"
            f"Corrections:\n{corrections_text}\n\n"
            f"Invoice context:\n"
            f"- Format: {format_name}\n"
            f"- Exporter: {exporter}\n"
            f"- Currency: {currency}\n"
            f"{notes_section}\n\n"
            f"Source text (first 2000 chars):\n{source_text[:2000]}\n\n"
            f"Analyze these corrections and create a reusable pattern that can fix "
            f"similar errors on future invoices of the same format.\n\n"
            f"Return JSON:\n"
            f'{{\n'
            f'  "pattern_id": "short_snake_case_id",\n'
            f'  "description": "One sentence: what error this pattern catches",\n'
            f'  "error_context": "Why this error happens on this format",\n'
            f'  "fix_prompt": "Specific instructions for the extraction LLM to avoid this error",\n'
            f'  "affected_fields": ["field1", "field2"],\n'
            f'  "format": "format_key or null",\n'
            f'  "company": "company name or null if applies to all companies in this format"\n'
            f'}}'
        )

        system_prompt = (
            "You analyze invoice extraction errors and create reusable correction patterns. "
            "Return ONLY valid JSON, no commentary."
        )

        try:
            response = bedrock.converse(
                modelId=FEEDBACK_MODEL,
                system=[{"text": system_prompt}],
                messages=[{"role": "user", "content": [{"text": user_prompt}]}],
                inferenceConfig={"maxTokens": 2000, "temperature": 0.0}
            )
            text = response['output']['message']['content'][0]['text']
        except Exception as e:
            logger.error(f"Pattern generation LLM call failed: {e}")
            return None

        # Parse JSON from response
        try:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                pattern = json.loads(json_match.group(0))
            else:
                pattern = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse pattern response: {text[:200]}")
            return None

        # Validate required keys
        required = ['pattern_id', 'description', 'fix_prompt', 'affected_fields']
        if not all(k in pattern for k in required):
            logger.warning(f"Pattern missing required keys: {pattern.keys()}")
            return None

        # Attach metadata
        pattern.setdefault('format', format_key)
        pattern['created_at'] = int(time.time())
        pattern['source'] = 'feedback'

        return pattern

    def _log_feedback(self, job_id: str, corrections: List[Dict],
                      patterns_created: List[str], worker_notes: str = None):
        """Save feedback entry via storage."""
        feedback_id = f"fb_{job_id[:8]}_{int(time.time()) % 1000:03d}"

        entry = {
            'feedback_id': feedback_id,
            'job_id': job_id,
            'timestamp': int(time.time()),
            'corrections_count': len(corrections),
            'corrections': corrections,
            'patterns_created': patterns_created,
            'worker_notes': worker_notes,
        }

        storage.put_doc('feedback', feedback_id, entry)

    def process_plain_text_feedback(self, job_id: str, text: str,
                                     agent_result: Dict,
                                     source_text: str) -> Dict:
        """
        Process plain text feedback from worker.
        Worker just types: "part number is 4397437 not H87, RITC is 8708809900"

        Uses LLM to parse into structured corrections, then calls process_feedback.
        """
        # Show first 3 items for context
        agent_items = agent_result.get('LineItems', [])[:3]
        items_json = json.dumps(agent_items, indent=2, default=str)

        user_prompt = (
            f"A worker provided this feedback on an invoice extraction:\n"
            f"'{text}'\n\n"
            f"The agent extracted these items:\n{items_json}\n\n"
            f"Parse the worker's feedback into structured corrections.\n"
            f"Return JSON:\n"
            f'{{\n'
            f'  "corrections": [\n'
            f'    {{"item_index": 0, "field": "PartNo", "agent_value": "H87", "worker_value": "4397437"}},\n'
            f'    ...\n'
            f'  ]\n'
            f'}}'
        )

        system_prompt = (
            "You parse worker feedback on invoice extractions into structured corrections. "
            "Return ONLY valid JSON, no commentary."
        )

        try:
            response = bedrock.converse(
                modelId=FEEDBACK_MODEL,
                system=[{"text": system_prompt}],
                messages=[{"role": "user", "content": [{"text": user_prompt}]}],
                inferenceConfig={"maxTokens": 2000, "temperature": 0.0}
            )
            resp_text = response['output']['message']['content'][0]['text']
        except Exception as e:
            logger.error(f"Plain text feedback LLM call failed: {e}")
            # Still log the raw feedback even if parsing fails
            self._log_feedback(job_id, [], [], worker_notes=text)
            return {
                'corrections_count': 0,
                'patterns_created': 0,
                'pattern_ids': [],
                'feedback_id': f"fb_{job_id[:8]}_{int(time.time()) % 1000:03d}",
                'error': str(e),
            }

        # Parse corrections from LLM response
        try:
            json_match = re.search(r'\{.*\}', resp_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
            else:
                parsed = json.loads(resp_text)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse plain text feedback response: {resp_text[:200]}")
            self._log_feedback(job_id, [], [], worker_notes=text)
            return {
                'corrections_count': 0,
                'patterns_created': 0,
                'pattern_ids': [],
                'feedback_id': f"fb_{job_id[:8]}_{int(time.time()) % 1000:03d}",
            }

        corrections = parsed.get('corrections', [])

        # Fill in action field
        for c in corrections:
            c.setdefault('action', 'corrected')

        # Build header from agent_result
        invoice_header = {
            k: agent_result.get(k)
            for k in HEADER_FIELDS + ['Exporter', 'Importer']
            if agent_result.get(k) is not None
        }

        return self.process_feedback(
            job_id=job_id,
            corrections=corrections,
            source_text=source_text,
            invoice_header=invoice_header,
            worker_notes=text,
        )

    def get_accuracy_for_job(self, agent_result: Dict, worker_result: Dict) -> Dict:
        """Calculate accuracy metrics for a single job."""
        corrections = self.diff_results(agent_result, worker_result)

        agent_items = agent_result.get('LineItems', [])
        worker_items = worker_result.get('LineItems', [])
        num_items = max(len(agent_items), len(worker_items))

        if num_items == 0:
            return {
                'total_fields': 0,
                'correct_fields': 0,
                'corrected_fields': 0,
                'field_accuracy': 1.0,
                'per_field_accuracy': {},
            }

        # Count total fields and per-field corrections
        total_fields = num_items * len(ITEM_FIELDS)
        # Add header fields
        total_fields += len(HEADER_FIELDS) + 4  # +4 for Exporter/Importer Name/Address

        # Build set of corrected (item_index, field) pairs
        corrected_set = set()
        for c in corrections:
            corrected_set.add((c.get('item_index'), c['field']))

        corrected_fields = len(corrected_set)
        correct_fields = total_fields - corrected_fields

        # Per-field accuracy for item fields
        per_field_accuracy = {}
        for field in ITEM_FIELDS:
            field_corrections = sum(
                1 for c in corrections
                if c['field'] == field and c.get('item_index') is not None
            )
            if num_items > 0:
                per_field_accuracy[field] = round(1.0 - (field_corrections / num_items), 3)
            else:
                per_field_accuracy[field] = 1.0

        return {
            'total_fields': total_fields,
            'correct_fields': correct_fields,
            'corrected_fields': corrected_fields,
            'field_accuracy': round(correct_fields / total_fields, 3) if total_fields > 0 else 1.0,
            'per_field_accuracy': per_field_accuracy,
        }


# ─── HELPERS ──────────────────────────────────────────────

def _normalize(val) -> str:
    """Normalize a value for comparison. Strips whitespace, lowercases."""
    if val is None:
        return ''
    return str(val).strip().lower()


def _classify_action(agent_val, worker_val) -> str:
    """Classify the type of correction."""
    if agent_val is None or _normalize(agent_val) == '':
        return 'added'
    if worker_val is None or _normalize(worker_val) == '':
        return 'removed'
    return 'corrected'


def _all_trivial(corrections: List[Dict]) -> bool:
    """Check if all corrections are trivial (whitespace-only changes)."""
    for c in corrections:
        agent = str(c.get('agent_value') or '').strip()
        worker = str(c.get('worker_value') or '').strip()
        # If values differ beyond whitespace/case, not trivial
        if agent.lower() != worker.lower():
            return False
    return True


def _item_summary(item: dict) -> str:
    """Short summary of a line item for logging."""
    return (
        f"{item.get('PartNo', '?')} | "
        f"{str(item.get('ItemDescription', ''))[:30]} | "
        f"qty={item.get('Quantity')}"
    )
