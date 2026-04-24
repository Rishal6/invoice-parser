"""
Pattern Library — RLHF-style error pattern learning.

Stores learned error→fix patterns.
Uses LLM to match current errors against known patterns.
Tracks confidence and promotes proven patterns to KB.

NO regex for matching. LLM reads pattern descriptions and decides.
"""
import json
import os
import re
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import boto3
from botocore.config import Config

import storage

logger = logging.getLogger(__name__)

REGION = os.environ.get('AWS_REGION', 'us-east-1')
PATTERN_MODEL = os.environ.get('PATTERN_MODEL', 'us.anthropic.claude-sonnet-4-5-20250929-v1:0')

bedrock = boto3.client('bedrock-runtime', region_name=REGION, config=Config(
    read_timeout=300, retries={'max_attempts': 3}
))


# ─── LLM HELPER ──────────────────────────────────────────

def _call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 2000) -> str:
    """Call Bedrock Converse API. Returns raw text response."""
    try:
        response = bedrock.converse(
            modelId=PATTERN_MODEL,
            system=[{"text": system_prompt}],
            messages=[{"role": "user", "content": [{"text": user_prompt}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0}
        )
        return response['output']['message']['content'][0]['text']
    except Exception as e:
        logger.error(f"Pattern LLM call failed: {e}")
        return ""


def _parse_json_from_text(text: str) -> dict:
    """Extract JSON object from LLM response text."""
    if not text:
        return {}
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    return {}


def _parse_json_array_from_text(text: str) -> list:
    """Extract JSON array from LLM response text."""
    if not text:
        return []
    json_match = re.search(r'\[.*\]', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    return []


# ─── PATTERN LIBRARY ─────────────────────────────────────

class PatternLibrary:
    def __init__(self):
        self.patterns: List[Dict] = []
        self._load()

    def _load(self):
        """Load patterns from storage."""
        doc = storage.get_doc('patterns', 'library')
        if doc and isinstance(doc.get('patterns'), list):
            self.patterns = doc['patterns']
            logger.info(f"Loaded {len(self.patterns)} patterns from storage")
        else:
            self.patterns = []
            logger.info("No patterns found in storage, starting empty")

    def _save(self):
        """Save patterns to storage."""
        storage.put_doc('patterns', 'library', {'patterns': self.patterns})

    def match_errors(self, errors: List[Dict], source_text: str,
                     invoice_header: Dict, format_key: str = None) -> List[Dict]:
        """
        Use LLM to match current errors against pattern library.

        Args:
            errors: List of review issues (from review_chunk).
                    Each has: type, severity, description, affectedItem,
                    expected, actual, fixInstruction
            source_text: The source text or "[scanned PDF — extracted via vision]"
            invoice_header: {Classification, InvoiceNo, Exporter, Currency, etc.}
            format_key: Detected format from knowledge.py (e.g. "cfdi")

        Returns:
            List of matched patterns with their fix_prompts.
            [{"pattern_id": "...", "fix_prompt": "...", "confidence": 0.8}, ...]
            Only patterns with confidence >= 0.3 are returned.
        """
        if not self.patterns or not errors:
            return []

        # Build pattern descriptions for LLM
        pattern_descriptions = []
        for p in self.patterns:
            pattern_descriptions.append(
                f"- pattern_id: {p['pattern_id']}\n"
                f"  format: {p.get('format', 'any')}\n"
                f"  description: {p['description']}\n"
                f"  error_context: {p.get('error_context', '')}\n"
                f"  affected_fields: {p.get('affected_fields', [])}\n"
                f"  confidence: {p.get('confidence', 0)}"
            )

        # Build error descriptions
        error_descriptions = []
        for e in errors:
            error_descriptions.append(
                f"- [{e.get('severity', '?')}] {e.get('type', '?')}: {e.get('description', '')}\n"
                f"  affected: {e.get('affectedItem', '?')}, "
                f"expected: {e.get('expected', '?')}, actual: {e.get('actual', '?')}"
            )

        exporter = ""
        if isinstance(invoice_header.get('Exporter'), dict):
            exporter = invoice_header['Exporter'].get('Name', '')
        elif isinstance(invoice_header.get('Exporter'), str):
            exporter = invoice_header['Exporter']

        # Truncate source text to avoid token bloat
        source_snippet = source_text[:500] if len(source_text) > 500 else source_text

        system_prompt = (
            "You are an error pattern matcher for invoice extraction. "
            "Match current errors against known patterns. Return valid JSON only."
        )

        user_prompt = (
            f"Here are known error patterns from our library:\n"
            f"{''.join(pattern_descriptions)}\n\n"
            f"Current invoice context:\n"
            f"- Format: {format_key or 'unknown'}\n"
            f"- Exporter: {exporter or 'unknown'}\n"
            f"- Errors found:\n{''.join(error_descriptions)}\n\n"
            f"Source text snippet:\n{source_snippet}\n\n"
            f"Which patterns match these errors? Return JSON:\n"
            f'{{\n'
            f'  "matches": [\n'
            f'    {{"pattern_id": "...", "reason": "..."}}\n'
            f'  ]\n'
            f'}}'
        )

        text = _call_llm(system_prompt, user_prompt)
        parsed = _parse_json_from_text(text)
        matches_raw = parsed.get('matches', [])

        if not matches_raw:
            logger.info("No pattern matches found by LLM")
            return []

        # Enrich with fix_prompt and confidence, filter by confidence >= 0.3
        results = []
        pattern_lookup = {p['pattern_id']: p for p in self.patterns}
        for m in matches_raw:
            pid = m.get('pattern_id', '')
            pattern = pattern_lookup.get(pid)
            if not pattern:
                logger.warning(f"LLM returned unknown pattern_id: {pid}")
                continue
            if pattern.get('confidence', 0) < 0.3:
                logger.info(f"Pattern {pid} matched but confidence {pattern.get('confidence', 0)} < 0.3, skipping")
                continue
            results.append({
                'pattern_id': pid,
                'fix_prompt': pattern.get('fix_prompt', ''),
                'confidence': pattern.get('confidence', 0),
                'reason': m.get('reason', ''),
            })

        logger.info(f"Matched {len(results)} patterns (from {len(matches_raw)} LLM matches)")
        return results

    def classify_errors(self, errors: List[Dict], source_text: str,
                        invoice_header: Dict) -> List[Dict]:
        """
        Use LLM to classify errors — understand WHY they happened.

        Returns enriched errors with:
        - root_cause: why this error happened
        - format_related: is this a format-specific issue?
        - fix_type: "prompt" (needs LLM) or "knowledge" (needs KB update)
        """
        if not errors:
            return []

        error_descriptions = json.dumps(errors[:10], indent=2, default=str)

        # Build header context string
        header_str = json.dumps(invoice_header, indent=2, default=str)

        # Truncate source text
        source_snippet = source_text[:500] if len(source_text) > 500 else source_text

        system_prompt = (
            "You are an invoice extraction error analyst. "
            "Classify errors to understand root causes. Return valid JSON only."
        )

        user_prompt = (
            f"Analyze these extraction errors. For each error, explain:\n"
            f"1. Root cause — WHY did this error happen?\n"
            f"2. Is this format-specific or general?\n"
            f"3. What kind of fix is needed?\n\n"
            f"Errors:\n{error_descriptions}\n\n"
            f"Invoice context:\n{header_str}\n\n"
            f"Source text snippet:\n{source_snippet}\n\n"
            f"Return a JSON array of enriched errors. Each object should have all original fields plus:\n"
            f'- "root_cause": string explaining why\n'
            f'- "format_related": true/false\n'
            f'- "fix_type": "prompt" or "knowledge"'
        )

        text = _call_llm(system_prompt, user_prompt)
        enriched = _parse_json_array_from_text(text)

        if not enriched:
            # Fall back: return original errors with empty classifications
            logger.warning("LLM classification returned no results, returning originals")
            return [
                {**e, 'root_cause': 'unknown', 'format_related': False, 'fix_type': 'prompt'}
                for e in errors
            ]

        logger.info(f"Classified {len(enriched)} errors")
        return enriched

    def build_fix_prompt(self, matched_patterns: List[Dict]) -> str:
        """
        Build a combined fix prompt from all matched patterns.

        Returns a string to inject into re_extract prompt.
        Prefixed with "KNOWN ERROR PATTERNS FOR THIS INVOICE:"
        """
        if not matched_patterns:
            return ""

        lines = ["KNOWN ERROR PATTERNS FOR THIS INVOICE:"]
        for i, mp in enumerate(matched_patterns, 1):
            fix = mp.get('fix_prompt', '')
            if fix:
                lines.append(f"{i}. {fix}")

        return "\n".join(lines)

    def add_pattern(self, pattern: Dict) -> str:
        """
        Add a new pattern to the library.
        Returns the pattern_id.
        """
        # Ensure required fields
        pattern_id = pattern.get('pattern_id')
        if not pattern_id:
            # Generate an id from description
            desc = pattern.get('description', 'unknown')
            pattern_id = desc[:40].lower().replace(' ', '_').replace('-', '_')
            pattern_id = re.sub(r'[^a-z0-9_]', '', pattern_id)
            pattern['pattern_id'] = pattern_id

        # Set defaults for tracking fields
        pattern.setdefault('format', None)
        pattern.setdefault('company', None)
        pattern.setdefault('description', '')
        pattern.setdefault('error_context', '')
        pattern.setdefault('fix_prompt', '')
        pattern.setdefault('affected_fields', [])
        pattern.setdefault('confidence', 0.2)
        pattern.setdefault('times_used', 1)
        pattern.setdefault('times_worked', 0)
        pattern.setdefault('success_rate', 0.0)
        pattern.setdefault('created_at', int(time.time()))
        pattern.setdefault('source', 'human_feedback')

        # Check for duplicate pattern_id
        existing_ids = {p['pattern_id'] for p in self.patterns}
        if pattern_id in existing_ids:
            # Append timestamp to make unique
            pattern_id = f"{pattern_id}_{int(time.time())}"
            pattern['pattern_id'] = pattern_id

        self.patterns.append(pattern)
        self._save()
        logger.info(f"Added pattern: {pattern_id}")
        return pattern_id

    def update_confidence(self, pattern_id: str, worked: bool):
        """
        Update pattern confidence after use.

        If worked: times_used += 1, times_worked += 1
        If not: times_used += 1
        Recalculate success_rate = times_worked / times_used

        Confidence calculation:
        - Base: success_rate
        - Boost if times_used >= 5: confidence = success_rate
        - New patterns (times_used < 5): confidence = success_rate * 0.5
        """
        for p in self.patterns:
            if p['pattern_id'] == pattern_id:
                p['times_used'] = p.get('times_used', 0) + 1
                if worked:
                    p['times_worked'] = p.get('times_worked', 0) + 1

                times_used = p['times_used']
                times_worked = p['times_worked']
                p['success_rate'] = round(times_worked / times_used, 3) if times_used > 0 else 0.0

                if times_used >= 5:
                    p['confidence'] = p['success_rate']
                else:
                    p['confidence'] = round(p['success_rate'] * 0.5, 3)

                self._save()
                logger.info(
                    f"Updated pattern {pattern_id}: "
                    f"worked={worked}, used={times_used}, "
                    f"success_rate={p['success_rate']}, confidence={p['confidence']}"
                )
                return

        logger.warning(f"Pattern not found for update: {pattern_id}")

    def get_promotable_patterns(self) -> List[Dict]:
        """
        Get patterns ready for KB promotion.

        Criteria:
        - times_used >= 5
        - success_rate >= 0.9
        - company is None (format-level, not company-specific)
        - not already promoted
        """
        promotable = []
        for p in self.patterns:
            if p.get('promoted_to_kb'):
                continue
            if p.get('times_used', 0) < 5:
                continue
            if p.get('success_rate', 0) < 0.9:
                continue
            if p.get('company') is not None:
                continue
            promotable.append(p)

        logger.info(f"Found {len(promotable)} promotable patterns")
        return promotable

    def promote_to_kb(self, pattern_id: str, kb_file: str):
        """
        Mark pattern as promoted to KB.
        Sets pattern confidence to 1.0 and adds 'promoted_to_kb' flag.
        Appends the pattern's knowledge to the specified KB file.
        """
        for p in self.patterns:
            if p['pattern_id'] == pattern_id:
                p['confidence'] = 1.0
                p['promoted_to_kb'] = True
                p['promoted_at'] = int(time.time())
                p['promoted_to_file'] = kb_file

                # Append knowledge to KB file
                kb_path = Path(__file__).parent / 'data' / 'kb' / kb_file
                try:
                    kb_path.parent.mkdir(parents=True, exist_ok=True)
                    knowledge_entry = (
                        f"\n\n## Pattern: {p['pattern_id']}\n"
                        f"Format: {p.get('format', 'any')}\n"
                        f"Description: {p['description']}\n"
                        f"Fix: {p.get('fix_prompt', '')}\n"
                        f"Affected fields: {', '.join(p.get('affected_fields', []))}\n"
                        f"Success rate: {p.get('success_rate', 0)} over {p.get('times_used', 0)} uses\n"
                    )
                    with open(kb_path, 'a') as f:
                        f.write(knowledge_entry)
                    logger.info(f"Promoted pattern {pattern_id} to KB file {kb_file}")
                except IOError as e:
                    logger.error(f"Failed to write to KB file {kb_file}: {e}")

                self._save()
                return

        logger.warning(f"Pattern not found for promotion: {pattern_id}")
