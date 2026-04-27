"""
Code Generator — LLM writes Python/regex extraction code for invoice templates.

Uses an agentic loop: generate → test → review → rewrite → test again.
Same pattern as the extraction pipeline (extract → review → re-extract).

Given OCR text, column headers, and a verified golden extraction result,
generates a pure Python function (regex + string parsing) that replicates
the extraction without any LLM call.
"""
import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import boto3
from botocore.config import Config

logger = logging.getLogger(__name__)

REGION = os.environ.get('AWS_REGION', 'us-east-1')
CODEGEN_MODEL = os.environ.get('CODEGEN_MODEL', 'us.anthropic.claude-sonnet-4-5-20250929-v1:0')

bedrock = boto3.client('bedrock-runtime', region_name=REGION, config=Config(
    read_timeout=300, retries={'max_attempts': 3}
))

MAX_REWRITE_ROUNDS = 2

# ─── SYSTEM PROMPTS ──────────────────────────────────────

CODEGEN_SYSTEM = (
    "You are a senior Python engineer who writes production-grade text parsers. "
    "You analyze raw text structure, identify repeating patterns, and write "
    "robust regex + string parsing code that handles OCR noise. "
    "Return ONLY valid JSON — no markdown, no commentary."
)

REVIEWER_SYSTEM = (
    "You are a code reviewer who compares parser output against a golden answer. "
    "You find every discrepancy: missing items, wrong values, extra items. "
    "Return ONLY valid JSON."
)

# ─── GENERATION PROMPT ───────────────────────────────────

CODEGEN_PROMPT = """You have a verified invoice extraction (the GOLDEN ANSWER) and the raw text it was extracted from.
Write a Python function that reproduces this extraction from the raw text using ONLY regex and string parsing.

## FUNCTION SIGNATURE (must be exact):
```
def extract(raw_text: str) -> dict
```

## OUTPUT SCHEMA:
```json
{{
  "InvoiceNo": str | null,
  "Date": str | null,
  "InvoiceCurrency": str | null,
  "FreightTerms": str | null,
  "IncoTerms": str | null,
  "TermsOfPayment": str | null,
  "Exporter": {{"Name": str | null, "Address": str | null}},
  "Importer": {{"Name": str | null, "Address": str | null}},
  "LineItems": [
    {{
      "PartNo": str | null, "ItemCode": str | null, "ItemDescription": str | null,
      "Quantity": str | null, "UnitOfQty": str | null, "UnitPrice": str | null,
      "RITC": str | null, "CountryOfOrigin": str | null
    }}
  ]
}}
```

## COLUMN ORDER:
{column_headers}

## GOLDEN ANSWER ({golden_item_count} items):
### Header:
{golden_header}

### Line Items (first 5):
{golden_items}

## RAW TEXT:
{raw_text_sample}

## ENGINEERING REQUIREMENTS:
1. Use ONLY `re` and `json` modules. No other imports.
2. Find the table header row by matching column keywords. Use the header row's character positions to define column boundaries.
3. Parse each data row using those column positions — NOT hardcoded widths.
4. Numeric fields: strip commas, currency symbols. Return digits + optional decimal only.
5. Header fields: targeted regex matching label then capturing value.
6. Handle OCR noise: extra spaces, O/0 confusion, slight misalignment.
7. Wrap each section in try/except. NEVER crash. Return partial results on error.
8. Must work for ANY invoice with the same column layout — use structure not hardcoded values.

Return:
```json
{{"template_id": "<snake_case_id>", "structured": true, "code": "<full function string>", "column_headers": {column_headers_json}}}
```

If too unstructured: {{"template_id": null, "structured": false, "code": null, "column_headers": []}}"""

# ─── REVIEW PROMPT ───────────────────────────────────────

REVIEW_PROMPT = """Compare the GENERATED output against the GOLDEN ANSWER field by field.

## GOLDEN ANSWER ({golden_item_count} items):
### Header:
{golden_header}

### Line Items (first 10):
{golden_items}

## GENERATED OUTPUT ({generated_item_count} items):
### Header:
{generated_header}

### Line Items (first 10):
{generated_items}

## REVIEW CHECKLIST:
1. Header fields: compare each field. Note mismatches.
2. Item count: golden has {golden_item_count}, generated has {generated_item_count}. How many missing?
3. For each golden item (by position), check: PartNo, ItemDescription, Quantity, UnitOfQty, UnitPrice match?
4. Find items in generated that don't exist in golden (hallucinated).
5. Check value formatting: Quantity should be digits only, UnitPrice digits only, Currency ISO code.

Return:
```json
{{
  "passed": true/false,
  "score": 0.0-1.0,
  "item_match_rate": 0.0-1.0,
  "header_issues": ["InvoiceNo: expected 'INV-001' got null", ...],
  "missing_items": ["Item 3: PartNo ABC-003 not found in generated", ...],
  "wrong_values": ["Item 1 Quantity: expected '500' got '50'", ...],
  "hallucinated_items": ["Generated item 5 not in golden", ...],
  "fix_instructions": "Specific instructions to fix the code: ..."
}}
```"""

# ─── REWRITE PROMPT ──────────────────────────────────────

REWRITE_PROMPT = """Your previous extraction code had issues. Fix them.

## PREVIOUS CODE:
```python
{previous_code}
```

## REVIEW RESULT:
{review_result}

## RAW TEXT (for reference):
{raw_text_sample}

## GOLDEN ANSWER (target — first 5 items):
{golden_items}

## FIX INSTRUCTIONS:
{fix_instructions}

Write the FIXED `extract` function. Address every issue in the review.
Focus on:
- Missing items: your regex pattern is too strict or not matching all rows
- Wrong values: your column position detection is off
- Hallucinated items: your regex is matching non-data rows (headers, totals, footers)

Return:
```json
{{"template_id": "{template_id}", "structured": true, "code": "<fixed function string>", "column_headers": {column_headers_json}}}
```"""


# ─── HELPERS ─────────────────────────────────────────────

def _sample_pages(raw_text: str) -> str:
    pages = re.split(r'\n{3,}|\f', raw_text)
    pages = [p.strip() for p in pages if p.strip()]
    if not pages:
        return raw_text[:4000]
    sampled = []
    indices = [0]
    if len(pages) > 2:
        indices.append(len(pages) // 2)
    if len(pages) > 1:
        indices.append(len(pages) - 1)
    for idx in indices:
        sampled.append(f"[Page {idx + 1} of {len(pages)}]\n{pages[idx][:2000]}")
    return "\n\n".join(sampled)


def _format_golden_header(result: dict) -> str:
    lines = []
    for f in ['InvoiceNo', 'Date', 'InvoiceCurrency', 'FreightTerms', 'IncoTerms', 'TermsOfPayment']:
        v = result.get(f)
        if v is not None:
            lines.append(f"  {f}: {json.dumps(v)}")
    for party in ['Exporter', 'Importer']:
        p = result.get(party)
        if isinstance(p, dict):
            lines.append(f"  {party}.Name: {json.dumps(p.get('Name'))}")
            lines.append(f"  {party}.Address: {json.dumps(p.get('Address'))}")
    return "\n".join(lines) if lines else "  (no header)"


def _format_items(items: list, max_items: int = 10) -> str:
    if not items:
        return "  (no items)"
    return json.dumps(items[:max_items], indent=2, default=str)


def _call_llm(system: str, prompt: str, max_tokens: int = 8000) -> str:
    response = bedrock.converse(
        modelId=CODEGEN_MODEL,
        system=[{"text": system}],
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0}
    )
    return response['output']['message']['content'][0]['text']


def _parse_json_response(text: str) -> Optional[dict]:
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


# ─── REVIEW ENGINE ───────────────────────────────────────

def _review_output(
    golden_result: dict,
    generated_result: dict,
) -> dict:
    """Compare generated output vs golden answer. Returns review with score and fix instructions."""
    golden_items = golden_result.get("LineItems", [])
    generated_items = generated_result.get("LineItems", [])

    golden_header_str = _format_golden_header(golden_result)
    generated_header_str = _format_golden_header(generated_result)

    prompt = REVIEW_PROMPT.format(
        golden_item_count=len(golden_items),
        golden_header=golden_header_str,
        golden_items=_format_items(golden_items),
        generated_item_count=len(generated_items),
        generated_header=generated_header_str,
        generated_items=_format_items(generated_items),
    )

    try:
        text = _call_llm(REVIEWER_SYSTEM, prompt, max_tokens=4000)
        review = _parse_json_response(text)
        if review:
            return review
    except Exception as e:
        logger.warning(f"[CODEGEN] Review LLM call failed: {e}")

    # Fallback: simple count-based review
    match_rate = min(len(generated_items), len(golden_items)) / max(len(golden_items), 1)
    return {
        "passed": match_rate >= 0.8,
        "score": round(match_rate, 2),
        "item_match_rate": round(match_rate, 2),
        "fix_instructions": f"Generated {len(generated_items)} items but golden has {len(golden_items)}.",
        "header_issues": [],
        "missing_items": [],
        "wrong_values": [],
        "hallucinated_items": [],
    }


# ─── REWRITE ENGINE ─────────────────────────────────────

def _rewrite_code(
    previous_code: str,
    review_result: dict,
    raw_text_sample: str,
    golden_items: list,
    template_id: str,
    column_headers_json: str,
) -> Optional[dict]:
    """Send review feedback to LLM, get fixed code."""
    prompt = REWRITE_PROMPT.format(
        previous_code=previous_code,
        review_result=json.dumps(review_result, indent=2, default=str),
        raw_text_sample=raw_text_sample,
        golden_items=_format_items(golden_items, max_items=5),
        fix_instructions=review_result.get("fix_instructions", "Fix all issues listed above."),
        template_id=template_id,
        column_headers_json=column_headers_json,
    )

    try:
        text = _call_llm(CODEGEN_SYSTEM, prompt)
        parsed = _parse_json_response(text)
        if parsed and parsed.get("structured") and parsed.get("code"):
            return parsed
    except Exception as e:
        logger.warning(f"[CODEGEN] Rewrite LLM call failed: {e}")

    return None


# ─── MAIN ENTRY POINTS ──────────────────────────────────

def generate_extractor_from_golden(
    sample_text: str,
    column_headers: list,
    format_key: str,
    extraction_result: dict,
) -> dict:
    """
    Agentic code generation with review + rewrite loop.

    Flow:
    1. Generate code (LLM sees golden answer + raw text)
    2. Run code in sandbox
    3. Review output vs golden (LLM reviewer)
    4. If issues → rewrite code with fix instructions
    5. Repeat up to MAX_REWRITE_ROUNDS times
    6. Return best version

    Returns: {"template_id": ..., "structured": bool, "code": str, "column_headers": [...]}
    """
    from runner import run_extractor

    items = extraction_result.get("LineItems", [])
    golden_header = _format_golden_header(extraction_result)
    golden_items_str = _format_items(items, max_items=5)
    page_samples = _sample_pages(sample_text)
    col_str = " | ".join(column_headers) if column_headers else "unknown"
    col_json = json.dumps(column_headers)

    # ── STEP 1: GENERATE ─────────────────────────────────
    prompt = CODEGEN_PROMPT.format(
        column_headers=col_str,
        golden_header=golden_header,
        golden_item_count=len(items),
        golden_items=golden_items_str,
        raw_text_sample=page_samples,
        column_headers_json=col_json,
    )

    try:
        text = _call_llm(CODEGEN_SYSTEM, prompt)
        result = _parse_json_response(text)
    except Exception as e:
        logger.error(f"[CODEGEN] Initial generation failed: {e}")
        return {"template_id": None, "structured": False, "code": None, "column_headers": []}

    if not result or not result.get("structured") or not result.get("code"):
        logger.info("[CODEGEN] LLM determined text too unstructured")
        return {"template_id": None, "structured": False, "code": None, "column_headers": []}

    template_id = re.sub(r'[^a-z0-9_]', '', (result.get("template_id") or "tmpl").lower().replace('-', '_'))
    if format_key and not template_id.startswith(format_key):
        template_id = f"{format_key}_{template_id}"

    current_code = result["code"]
    best_code = current_code
    best_score = 0.0
    best_count = 0

    for round_num in range(1 + MAX_REWRITE_ROUNDS):
        round_label = "GENERATE" if round_num == 0 else f"REWRITE #{round_num}"

        # ── STEP 2: RUN ──────────────────────────────────
        try:
            test_result = run_extractor(current_code, sample_text)
        except Exception as e:
            logger.warning(f"[CODEGEN] {round_label} run failed: {e}")
            if round_num == 0:
                return {"template_id": None, "structured": False, "code": None, "column_headers": []}
            break

        if not test_result.get("success"):
            logger.warning(f"[CODEGEN] {round_label} extractor crashed: {test_result.get('error')}")
            if round_num >= MAX_REWRITE_ROUNDS:
                break
            review_for_rewrite = {
                "passed": False, "score": 0,
                "fix_instructions": f"Code crashed with error: {test_result.get('error')}. Fix the bug.",
            }
            rewritten = _rewrite_code(current_code, review_for_rewrite, page_samples, items, template_id, col_json)
            if rewritten and rewritten.get("code"):
                current_code = rewritten["code"]
                continue
            break

        generated_count = len(test_result.get("LineItems", []))
        logger.info(f"[CODEGEN] {round_label}: {generated_count}/{len(items)} items")

        # ── STEP 3: REVIEW ───────────────────────────────
        review = _review_output(extraction_result, test_result)
        score = review.get("score", 0)
        item_rate = review.get("item_match_rate", generated_count / max(len(items), 1))

        if score > best_score or generated_count > best_count:
            best_code = current_code
            best_score = score
            best_count = generated_count

        logger.info(f"[CODEGEN] {round_label} review: score={score}, items={generated_count}/{len(items)}, passed={review.get('passed')}")

        if review.get("passed") and item_rate >= 0.8:
            logger.info(f"[CODEGEN] PASSED after {round_label}")
            break

        # ── STEP 4: REWRITE ──────────────────────────────
        if round_num >= MAX_REWRITE_ROUNDS:
            logger.info(f"[CODEGEN] Max rewrites reached, using best version (score={best_score})")
            break

        rewritten = _rewrite_code(current_code, review, page_samples, items, template_id, col_json)
        if rewritten and rewritten.get("code"):
            current_code = rewritten["code"]
        else:
            logger.warning(f"[CODEGEN] Rewrite failed, keeping best version")
            break

    return {
        "template_id": template_id,
        "code": best_code,
        "column_headers": result.get("column_headers", column_headers),
        "structured": True,
        "score": best_score,
        "items_generated": best_count,
        "items_golden": len(items),
    }


def generate_extractor(sample_text: str, column_headers: list, format_key: str) -> dict:
    """Fallback: basic generation without golden answer or review loop."""
    page_samples = _sample_pages(sample_text)
    col_str = ", ".join(column_headers) if column_headers else "unknown"
    col_json = json.dumps(column_headers)

    prompt = f"""Write a Python `def extract(raw_text: str) -> dict` function that extracts invoice data using regex.

COLUMN HEADERS: {col_str}

SAMPLE TEXT:
{page_samples}

REQUIREMENTS:
1. Use ONLY `re` and `json` modules.
2. Detect table header row, parse data rows by column positions.
3. Strip commas/currency from numerics. Handle OCR noise. Never crash.
4. Return dict with: InvoiceNo, Date, InvoiceCurrency, FreightTerms, IncoTerms, TermsOfPayment, Exporter, Importer, LineItems.

Return JSON: {{"template_id": "<id>", "structured": true, "code": "<function>", "column_headers": {col_json}}}
If too messy: {{"template_id": null, "structured": false, "code": null, "column_headers": []}}"""

    try:
        text = _call_llm(CODEGEN_SYSTEM, prompt)
        parsed = _parse_json_response(text)
        if parsed and parsed.get("structured") and parsed.get("code"):
            tid = re.sub(r'[^a-z0-9_]', '', (parsed.get("template_id") or "tmpl").lower().replace('-', '_'))
            if format_key and format_key != 'unknown' and not tid.startswith(format_key):
                tid = f"{format_key}_{tid}"
            parsed["template_id"] = tid
            return parsed
    except Exception as e:
        logger.error(f"[CODEGEN] Basic generation failed: {e}")

    return {"template_id": None, "structured": False, "code": None, "column_headers": []}
