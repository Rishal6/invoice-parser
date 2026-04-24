"""
Code Generator — LLM writes Python/regex extraction code for invoice templates.

Given sample OCR text and detected column headers, calls Bedrock LLM
to generate a pure Python extractor function (regex + string parsing).
The generated code can run without LLM for repeat invoices.
"""
import json
import logging
import os
import re

import boto3
from botocore.config import Config

logger = logging.getLogger(__name__)

REGION = os.environ.get('AWS_REGION', 'us-east-1')
CODEGEN_MODEL = os.environ.get('CODEGEN_MODEL', 'us.anthropic.claude-sonnet-4-5-20250929-v1:0')

bedrock = boto3.client('bedrock-runtime', region_name=REGION, config=Config(
    read_timeout=300, retries={'max_attempts': 3}
))

CODEGEN_SYSTEM = (
    "You are an expert Python code generator specializing in invoice text parsing. "
    "You write clean, robust regex + string parsing code. Return ONLY valid JSON."
)

CODEGEN_PROMPT = """Write a Python function that extracts structured invoice data from raw OCR text.

The function signature MUST be:
    def extract(raw_text: str) -> dict

It must return a dict with these keys:
- "InvoiceNo": str or None
- "Date": str or None
- "InvoiceCurrency": str or None (ISO 4217)
- "FreightTerms": str or None
- "IncoTerms": str or None (FOB/CIF/CI/CF only)
- "TermsOfPayment": str or None
- "Exporter": {{"Name": str or None, "Address": str or None}}
- "Importer": {{"Name": str or None, "Address": str or None}}
- "LineItems": list of dicts, each with:
    - "PartNo": str or None
    - "ItemCode": str or None
    - "ItemDescription": str or None
    - "Quantity": str or None (digits + optional decimal only)
    - "UnitOfQty": str or None (PCS, EA, KG, etc.)
    - "UnitPrice": str or None (digits + optional decimal only)
    - "RITC": str or None (digits only)
    - "CountryOfOrigin": str or None (full country name)

RULES:
- Use ONLY `re` and `json` modules. No other imports.
- Use regex patterns, string splitting, and line-by-line parsing.
- Strip commas and currency symbols from numeric fields.
- Handle OCR noise: extra spaces, misaligned columns, minor typos.
- The function must not crash — wrap parsing in try/except where needed.
- Return the dict directly. No print statements, no file I/O.

COLUMN HEADERS DETECTED IN THIS INVOICE FORMAT:
{column_headers}

SAMPLE OCR TEXT (from this invoice template):
--- PAGE 1 ---
{page_samples}

Based on the text layout and column headers above, write the `extract` function.
Return JSON:
{{
    "template_id": "<short_snake_case_id based on company/format>",
    "structured": true,
    "code": "<the full Python function as a string>",
    "column_headers": {column_headers_json}
}}

If the text is too unstructured, messy, or inconsistent for reliable regex extraction, return:
{{
    "template_id": null,
    "structured": false,
    "code": null,
    "column_headers": []
}}"""


def _sample_pages(raw_text: str) -> str:
    """Sample page 1, middle, and last page from multi-page text.
    Pages are separated by double newlines or form feeds."""
    # Split on page boundaries
    pages = re.split(r'\n{3,}|\f', raw_text)
    pages = [p.strip() for p in pages if p.strip()]

    if not pages:
        return raw_text[:3000]

    sampled = []
    indices = [0]
    if len(pages) > 2:
        indices.append(len(pages) // 2)
    if len(pages) > 1:
        indices.append(len(pages) - 1)

    for idx in indices:
        page_text = pages[idx][:1500]  # Cap per page
        sampled.append(f"[Page {idx + 1} of {len(pages)}]\n{page_text}")

    return "\n\n".join(sampled)


def generate_extractor(sample_text: str, column_headers: list, format_key: str) -> dict:
    """
    Call Bedrock LLM to generate a Python extraction function.

    Args:
        sample_text: OCR text from the invoice (multi-page ok, will be sampled)
        column_headers: List of detected column header strings
        format_key: Format identifier (e.g. "us_commercial", "gst")

    Returns:
        {"template_id": "...", "code": "...", "column_headers": [...], "structured": true/false}
    """
    page_samples = _sample_pages(sample_text)
    column_headers_str = ", ".join(column_headers) if column_headers else "unknown"
    column_headers_json = json.dumps(column_headers)

    prompt = CODEGEN_PROMPT.format(
        column_headers=column_headers_str,
        page_samples=page_samples,
        column_headers_json=column_headers_json,
    )

    try:
        response = bedrock.converse(
            modelId=CODEGEN_MODEL,
            system=[{"text": CODEGEN_SYSTEM}],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 8000, "temperature": 0.0}
        )
        text = response['output']['message']['content'][0]['text']
    except Exception as e:
        logger.error(f"Codegen LLM call failed: {e}")
        return {"template_id": None, "structured": False, "code": None, "column_headers": []}

    # Parse JSON response
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse codegen response")
            return {"template_id": None, "structured": False, "code": None, "column_headers": []}
    else:
        logger.warning("No JSON found in codegen response")
        return {"template_id": None, "structured": False, "code": None, "column_headers": []}

    structured = parsed.get('structured', False)
    if not structured:
        logger.info("LLM determined invoice is not structured enough for codegen")
        return {"template_id": None, "structured": False, "code": None, "column_headers": []}

    template_id = parsed.get('template_id')
    code = parsed.get('code')

    if not template_id or not code:
        logger.warning("Codegen response missing template_id or code")
        return {"template_id": None, "structured": False, "code": None, "column_headers": []}

    # Clean template_id
    template_id = re.sub(r'[^a-z0-9_]', '', template_id.lower().replace('-', '_'))
    if format_key and format_key != 'unknown':
        template_id = f"{format_key}_{template_id}"

    logger.info(f"Codegen produced template: {template_id} (structured={structured})")
    return {
        "template_id": template_id,
        "code": code,
        "column_headers": parsed.get('column_headers', column_headers),
        "structured": True,
    }
