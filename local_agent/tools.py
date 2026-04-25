"""
Invoice extraction tools — all 6 tools for the orchestrator agent.

Tool 1: split_pdf       — Python: splits PDF locally with overlap
Tool 2: extract_chunk   — LLM: reads chunk → structured JSON
Tool 3: review_chunk    — LLM: compares source text vs extraction, finds gaps
Tool 4: re_extract      — LLM: focused re-extraction of missed/wrong items
Tool 5: deduplicate     — Python: exact + fuzzy + business logic dedup
Tool 6: verify_final    — Python + LLM: final completeness check
"""
import json
import os
import re
import logging
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

import base64

import boto3
import fitz as pymupdf
from botocore.config import Config
from pypdf import PdfReader, PdfWriter
from rapidfuzz import fuzz
from strands import tool

from knowledge import KnowledgeBase
from patterns import PatternLibrary
from registry import TemplateRegistry
from codegen import generate_extractor
from runner import run_extractor
from tracer import JobTracer

logger = logging.getLogger(__name__)

REGION = os.environ.get('AWS_REGION', 'us-east-1')
EXTRACT_MODEL_PRIMARY = os.environ.get('EXTRACT_MODEL', 'us.anthropic.claude-sonnet-4-5-20250929-v1:0')
EXTRACT_MODEL_FALLBACK = os.environ.get('EXTRACT_MODEL_FALLBACK', 'us.anthropic.claude-sonnet-4-5-20250929-v1:0')
REVIEW_MODEL = os.environ.get('REVIEW_MODEL', 'us.anthropic.claude-sonnet-4-5-20250929-v1:0')

bedrock = boto3.client('bedrock-runtime', region_name=REGION, config=Config(
    read_timeout=300, retries={'max_attempts': 3}
))

CHUNK_DIR = os.path.join(tempfile.gettempdir(), 'invoice_chunks')
os.makedirs(CHUNK_DIR, exist_ok=True)

# ─── KB + PATTERN + TEMPLATE INSTANCES ───────────────────
kb = KnowledgeBase()
pattern_library = PatternLibrary()
template_registry = TemplateRegistry()


# ─── ACCUMULATOR ───────────────────────────────────────────

@dataclass
class ExtractionAccumulator:
    items: List[Dict] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    processed_indices: set = field(default_factory=set)
    reviewed_indices: set = field(default_factory=set)
    total_chunks: int = 0
    chunk_paths: List[str] = field(default_factory=list)
    chunk_source_texts: Dict[int, str] = field(default_factory=dict)
    chunk_extractions: Dict[int, List[Dict]] = field(default_factory=dict)
    chunk_retry_count: Dict[int, int] = field(default_factory=dict)
    chunk_model_used: Dict[int, str] = field(default_factory=dict)
    invoice_header: Dict = field(default_factory=dict)
    review_summary: Dict = field(default_factory=dict)
    dedup_summary: Dict = field(default_factory=dict)
    tool_log: List[Dict] = field(default_factory=list)
    start_time: float = 0.0
    format_key: str = "unknown"
    format_confidence: float = 0.0
    kb_context: str = ""
    matched_patterns: List[Dict] = field(default_factory=list)
    tracer: Any = None


accumulator = ExtractionAccumulator()


def reset_accumulator():
    global accumulator
    accumulator = ExtractionAccumulator()


def init_tracer(job_id: str, filename: str = ""):
    """Create a tracer for this job and attach to the accumulator."""
    accumulator.tracer = JobTracer(job_id, filename)


# ─── EXTRACTION PROMPT ─────────────────────────────────────

EXTRACTION_PROMPT = """You are an expert OCR and data extraction AI specializing in invoice processing. Your objective is to deeply analyze the provided page, image, or text and extract structured data strictly adhering to the following instructions.

1. CLASSIFICATION & DOCUMENT SCOPE
- Classification: First, determine if the provided document is an invoice (not a packing list, purchase order, or related document). Provide your conclusion in one short sentence in the "Classification" field.
- Action based on Classification: If and only if the document is an invoice, extract all required fields. If it is NOT an invoice, return all extracted fields as null.
- Multi-Page Rule: For multiple pages, concatenate line items page-wise (Page 1 items, followed by Page 2 items). Maintain the original visual order within each page.
- Packing List Rule: The image may contain an invoice followed by a packing list. Extract item details ONLY from the invoice section. Ignore all packing list data.
- Deep Read Requirement: Carefully examine every part of every page—including headers, footers, margins, stamps, tables, and tiny print.

2. FIELD MAPPING & VARIANTS
Look for the following fields using these accepted variants:
- Exporter: Exporter, Shipper, Supplier, Seller.
- Importer: Importer, Consignee, Buyer, Receiver.
- Invoice Number: Invoice No., Bill Number, Invoice ID.
- Invoice Date: Date, Invoice Date, Bill Date. (Do not confuse Order Date, Issue Date, or Shipping Date with Invoice Date.)
- Freight Terms: Freight Terms, Delivery Terms, Shipment Terms.
- Payment Terms: Terms of Payment, Payment Terms, Payable Terms.
- Currency: Currency, Invoice Currency, Total Currency.
- Country of Origin: CountryOfOrigin, Origin Country, Made In, COO, CO, C/O.
- RITC: RITC, HSN, HS Code.
- PartNo: Check variants in this exact priority order: 1. Customer Material Description, 2. Part No., 3. Part Number, 4. Material No., 5. Item No., 6. Product Code, 7. Article No.

3. EXTRACTION & FORMATTING RULES
- Numeric Formatting: For Quantity and UnitPrice, return values as strings containing ONLY digits and an optional decimal point (e.g., "45485" or "1234.50"). Strip all commas, currency symbols, and thousand separators.
- Currency Normalization: Return currency using strict ISO 4217 three-letter codes (e.g., USD, EUR, INR).
- Unit of Measure (UnitOfQty): Normalize to short standardized forms (e.g., PCS, EA, KG, G, L, M, CM). PC should be normalized to PCS.
- Country Names: Return as full, official English country names (e.g., CHN -> China, JPN -> Japan).
- RITC / HSN: Return as a string of digits only. Only extract from a column explicitly labeled RITC, HSN, or HS Code. If no such column exists, return null.
- Spacing: Collapse repeated internal spaces into a single space.
- Missing Values: If a value cannot be found, confidently inferred, or is misaligned, set that field to null.

4. INCOTERMS EXTRACTION RULE
- Detection Strategy: Incoterms are often NOT explicitly labeled. Scan for headers like "Delivery Terms", "Terms of Delivery", "Freight Terms", "Shipment Terms", or look in the top header blocks for 3-letter trade terms (FCA, EXW, FOB, CIF, DAP).
- Standardized values allowed: FOB, CIF, CI, CF.
- Mapping Rules:
  - Map EXW, FCA, FCA/SUZHOU, EXWork, FOB, DAP, DDP to FOB.
  - Map CFR to CF.
  - Map CIF to CIF.
  - Map CIP to CI.
- Location Stripping: If Incoterms appear with locations (e.g., "FCA/SUZHOU", "FOB Shanghai"), extract the base term, apply mapping, ignore location. Return ONLY the normalized keyword.
- If Incoterms are not found anywhere, return null.

5. STRICT TABLE & ROW ALIGNMENT RULES
- Critical Order Preservation: Extract line items in the exact visual order they appear (top-to-bottom, left-to-right). Do not reorder, sort, merge, or deduplicate.
- Row-Locking & Headers: Always read an entire horizontal row. Distinguish between Quantity, Unit Price, and Total Amount. Do NOT extract total line amounts as the unit price.
- OCR Concatenation Guard: OCR engines often merge adjacent columns (e.g., "PRODUCTNAME500 PCS"). If a numeric value appears fused to the end of a word, split it: text part as ItemDescription, numeric part as Quantity.
- UOM as Boundary Marker: The UOM column often sits between Quantity and Unit Price. Use it as a visual boundary.
- Description Anchor Rule: Treat ItemDescription as the primary anchor. Extract Quantity, UnitOfQty, UnitPrice, RITC, and CountryOfOrigin ONLY from that same horizontal row.
- Row Alignment Verification: Do not combine values from different rows. If uncertain, set the field to null.
- Swap Guard: If neighboring rows appear to have swapped numeric values, do NOT swap them back. Leave uncertain fields as null.

6. OUTPUT SCHEMA ENFORCEMENT
- NO EXTRA FIELDS ALLOWED. Use ONLY the exact keys provided.
- Every field must appear even if null.
- Return ONLY the JSON object. No commentary, no markdown code blocks.

Use this exact JSON structure:
{
  "Classification": null,
  "InvoiceNo": null,
  "Date": null,
  "InvoiceCurrency": null,
  "FreightTerms": null,
  "IncoTerms": null,
  "TermsOfPayment": null,
  "Exporter": {
    "Name": null,
    "Address": null
  },
  "Importer": {
    "Name": null,
    "Address": null
  },
  "LineItems": [
    {
      "PartNo": null,
      "ItemCode": null,
      "ItemDescription": null,
      "Quantity": null,
      "UnitOfQty": null,
      "UnitPrice": null,
      "RITC": null,
      "CountryOfOrigin": null
    }
  ]
}"""

REVIEW_PROMPT = """You are a quality reviewer for invoice data extraction.

You will receive:
1. SOURCE: The original text or images from a PDF invoice chunk
2. EXTRACTION: The structured JSON extracted from that source

Your job: Compare them carefully and find EVERY discrepancy.

For each issue found, return:
- type: MISSING_ITEM | HALLUCINATED_ITEM | WRONG_VALUE | DUPLICATE_ITEM | WRONG_CLASSIFICATION
- severity: CRITICAL | WARNING | INFO
- description: What's wrong
- affectedItem: Which item (PartNo or description)
- expected: What the source shows
- actual: What the extraction has (or "MISSING")
- fixInstruction: Specific instruction to fix this

How to check:
1. Read through the source line by line
2. For each line item in the source, verify it exists in the extraction
3. For each item in the extraction, verify it exists in the source (anti-hallucination)
4. Check Classification — is the document actually an invoice? If not, all fields should be null
5. Check field accuracy: PartNo priority (Customer Material Desc > Part No > Material No > Item No), IncoTerms mapping, Currency ISO codes, Country full names, UOM normalization, Quantity/UnitPrice as digit-only strings
6. Check row alignment — are values from the correct horizontal row?
7. Check for duplicate items in the extraction

Return JSON:
{
  "passed": true/false,
  "score": 0.0-1.0,
  "expectedItemCount": number,
  "extractedItemCount": number,
  "issues": [...],
  "summary": "one sentence"
}

If there are 0 CRITICAL issues, set passed=true.
Return ONLY valid JSON."""

RE_EXTRACT_PROMPT = """You previously extracted line items from this invoice chunk but some were missed or wrong.

Here is the source again, and the specific issues found by the reviewer.

Fix ONLY the issues listed. Return the corrected/additional items as JSON:
{
  "additional_items": [
    {
      "PartNo": null,
      "ItemCode": null,
      "ItemDescription": null,
      "Quantity": null,
      "UnitOfQty": null,
      "UnitPrice": null,
      "RITC": null,
      "CountryOfOrigin": null
    }
  ],
  "corrected_items": [
    {
      "original_PartNo": "string — identifies which item to fix",
      "PartNo": null,
      "ItemCode": null,
      "ItemDescription": null,
      "Quantity": null,
      "UnitOfQty": null,
      "UnitPrice": null,
      "RITC": null,
      "CountryOfOrigin": null
    }
  ]
}

Follow the same formatting rules: Quantity/UnitPrice as digit-only strings, Currency as ISO 4217, UOM normalized (PCS, EA, KG), Country as full English name, RITC as digits only.
Return ONLY valid JSON."""


# ─── HELPER: Call Bedrock ──────────────────────────────────

def _call_bedrock(model_id: str, system: str, user_content: list, max_tokens: int = 10000) -> dict:
    """Call Bedrock Converse API. Single call, no continuation."""
    try:
        response = bedrock.converse(
            modelId=model_id,
            system=[{"text": system}],
            messages=[{"role": "user", "content": user_content}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0}
        )
    except Exception as e:
        logger.error(f"Bedrock call failed: {e}")
        return {'text': f'{{"error": "{str(e)}"}}', 'input_tokens': 0, 'output_tokens': 0}

    usage = response.get('usage', {})
    text = response['output']['message']['content'][0]['text']
    truncated = response.get('stopReason') == 'max_tokens'

    return {
        'text': text,
        'input_tokens': usage.get('inputTokens', 0),
        'output_tokens': usage.get('outputTokens', 0),
        'truncated': truncated,
    }


def _parse_json(text: str) -> dict:
    """Extract JSON from LLM response, including truncated JSON."""
    # Try clean parse first
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
    # Truncated JSON — try to close the array and object
    start = text.find('{')
    if start >= 0:
        fragment = text[start:]
        # Find last complete object in LineItems array
        last_close = fragment.rfind('}')
        if last_close > 0:
            attempt = fragment[:last_close + 1]
            # Close any open arrays/objects
            open_brackets = attempt.count('[') - attempt.count(']')
            open_braces = attempt.count('{') - attempt.count('}')
            attempt += ']' * open_brackets + '}' * open_braces
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                pass
    return {}


def _extract_text_from_pdf(pdf_path: str) -> str:
    """Extract raw text from a PDF file using pypdf."""
    reader = PdfReader(pdf_path)
    texts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            texts.append(text)
    return "\n\n".join(texts)


def _render_pdf_to_images(pdf_path: str, dpi: int = 200) -> list[dict]:
    """Render PDF pages to PNG images for vision-based extraction.
    Returns list of Bedrock Converse image content blocks."""
    doc = pymupdf.open(pdf_path)
    image_blocks = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        png_bytes = pix.tobytes("png")
        image_blocks.append({
            "image": {
                "format": "png",
                "source": {"bytes": png_bytes},
            }
        })
    doc.close()
    return image_blocks


def _render_page_strips(pdf_path: str, page_index: int, strips_per_page: int = 5,
                        overlap_pct: float = 0.2, dpi: int = 250) -> list[dict]:
    """Render a single PDF page as horizontal strips with overlap.
    Each strip is a Bedrock Converse image content block."""
    doc = pymupdf.open(pdf_path)
    page = doc[page_index]
    rect = page.rect
    zoom = dpi / 72

    strip_height = rect.height / strips_per_page
    overlap = strip_height * overlap_pct

    strips = []
    for i in range(strips_per_page):
        y0 = max(0, i * strip_height - (overlap if i > 0 else 0))
        y1 = min(rect.height, (i + 1) * strip_height + (overlap if i < strips_per_page - 1 else 0))

        clip = pymupdf.Rect(0, y0, rect.width, y1)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=clip)

        png_bytes = pix.tobytes("png")
        strips.append({
            "image": {
                "format": "png",
                "source": {"bytes": png_bytes},
            }
        })

    doc.close()
    return strips


# ─── TOOL 1: SPLIT PDF ────────────────────────────────────

@tool
def split_pdf(pdf_path: str, pages_per_chunk: int = 10, overlap_pages: int = 1) -> str:
    """Split a PDF into chunks with overlap for processing.

    Args:
        pdf_path: Local path to the PDF file
        pages_per_chunk: Pages per chunk (default 10)
        overlap_pages: Pages of overlap between chunks (default 1)

    Returns:
        Summary of chunks created with file paths
    """
    global accumulator

    logger.info(f"[TOOL] split_pdf called: {pdf_path}")
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)

    chunk_paths = []
    stride = max(1, pages_per_chunk - overlap_pages)

    for i in range(0, total_pages, stride):
        end = min(i + pages_per_chunk, total_pages)
        writer = PdfWriter()
        for p in range(i, end):
            writer.add_page(reader.pages[p])

        chunk_path = os.path.join(CHUNK_DIR, f"chunk_{len(chunk_paths)}.pdf")
        with open(chunk_path, 'wb') as f:
            writer.write(f)

        chunk_paths.append(chunk_path)

        if end == total_pages:
            break

    accumulator.total_chunks = len(chunk_paths)
    accumulator.chunk_paths = chunk_paths

    if accumulator.tracer:
        accumulator.tracer.step(
            'split_pdf',
            result=f"{len(chunk_paths)} chunks from {total_pages} pages ({pages_per_chunk}/chunk, {overlap_pages} overlap)",
            extra={'total_pages': total_pages, 'chunks': len(chunk_paths), 'overlap': overlap_pages},
        )

    listing = "\n".join(
        f"  Chunk {i}: pages {i*stride+1}-{min((i*stride)+pages_per_chunk, total_pages)}"
        for i in range(len(chunk_paths))
    )
    return (
        f"Split {total_pages}-page PDF into {len(chunk_paths)} chunks "
        f"({pages_per_chunk} pages each, {overlap_pages}-page overlap):\n{listing}\n\n"
        f"Now call extract_chunk for each chunk index 0 to {len(chunk_paths)-1}."
    )


# ─── TOOL 2: EXTRACT CHUNK ────────────────────────────────

@tool
def extract_chunk(chunk_index: int) -> str:
    """Extract all line items from a PDF chunk using LLM.

    Args:
        chunk_index: Zero-based chunk index from split_pdf

    Returns:
        Summary of extraction results. Full data stored in accumulator.
    """
    global accumulator

    logger.info(f"[TOOL] extract_chunk called: chunk_index={chunk_index}")
    if chunk_index >= len(accumulator.chunk_paths):
        return f"ERROR: chunk_index {chunk_index} out of range (0-{len(accumulator.chunk_paths)-1})"

    chunk_path = accumulator.chunk_paths[chunk_index]
    source_text = _extract_text_from_pdf(chunk_path)
    use_vision = len(source_text.strip()) < 50

    if use_vision:
        logger.info(f"[TOOL] chunk {chunk_index}: scanned PDF — using vision extraction")
        source_text = "[scanned PDF — extracted via vision]"

    accumulator.chunk_source_texts[chunk_index] = source_text

    # Detect format on first chunk and retrieve KB context
    if chunk_index == 0 and accumulator.format_key == "unknown":
        images_for_detect = None
        if use_vision:
            images_for_detect = _render_pdf_to_images(chunk_path)
        kb_result = kb.detect_and_retrieve(
            source_text=source_text if not use_vision else None,
            images=images_for_detect,
            invoice_header=accumulator.invoice_header or None,
        )
        accumulator.format_key = kb_result['format']
        accumulator.format_confidence = kb_result['confidence']
        accumulator.kb_context = kb_result['context']
        logger.info(f"[KB] Format: {kb_result['format_name']} (confidence={kb_result['confidence']:.2f})")

    # ─── TEMPLATE REGISTRY: check for saved extractor ────
    template_hit = False
    if not use_vision and accumulator.format_key != "unknown":
        saved_code = template_registry.lookup(accumulator.format_key)
        if saved_code:
            logger.info(f"[TEMPLATE] Found saved extractor for format: {accumulator.format_key}")
            try:
                codegen_result = run_extractor(saved_code, source_text)
                if codegen_result.get('success') and codegen_result.get('LineItems'):
                    items = codegen_result['LineItems']
                    template_hit = True
                    template_registry.record_success(accumulator.format_key)
                    logger.info(f"[TEMPLATE] Extractor returned {len(items)} items")

                    # Populate header from codegen result
                    if not accumulator.invoice_header:
                        header_fields = [
                            'Classification', 'InvoiceNo', 'Date', 'InvoiceCurrency',
                            'FreightTerms', 'IncoTerms', 'TermsOfPayment', 'Exporter', 'Importer',
                        ]
                        accumulator.invoice_header = {
                            k: codegen_result.get(k) for k in header_fields if codegen_result.get(k) is not None
                        }
                else:
                    logger.warning(f"[TEMPLATE] Extractor failed or returned no items, falling back to LLM")
            except Exception as e:
                logger.warning(f"[TEMPLATE] Extractor error: {e}, falling back to LLM")

    if template_hit:
        # Skip LLM — use template results directly
        for i, item in enumerate(items, 1):
            item['_chunk_index'] = chunk_index
            item['ItemNo'] = i

        accumulator.chunk_extractions[chunk_index] = items
        accumulator.processed_indices.add(chunk_index)
        accumulator.chunk_model_used[chunk_index] = 'template'

        if accumulator.tracer:
            accumulator.tracer.step(
                'extract_chunk', chunk=chunk_index,
                result=f"{len(items)} items (template, no LLM)",
                extra={'items_found': len(items), 'truncated': False, 'vision': False, 'model': 'template'},
            )

        return (
            f"Chunk {chunk_index}: {len(items)} items extracted (template — no LLM). "
            f"Progress: {len(accumulator.processed_indices)}/{accumulator.total_chunks} chunks. "
            f"Now call review_chunk({chunk_index}) to verify."
        )

    # ─── LLM EXTRACTION (existing path) ─────────────────
    # Build prompt: base + KB context (only relevant knowledge)
    prompt = EXTRACTION_PROMPT
    if accumulator.kb_context:
        prompt = f"{EXTRACTION_PROMPT}\n\n{accumulator.kb_context}"

    model_id = accumulator.chunk_model_used.get(chunk_index, EXTRACT_MODEL_PRIMARY)

    chunk_reader = PdfReader(chunk_path)
    chunk_pages = len(chunk_reader.pages)

    # ─── LLM EXTRACTION ──────────────────────────────────────
    try:
        if use_vision:
            image_blocks = _render_pdf_to_images(chunk_path)
            user_content = [{"text": f"{prompt}\n\nExtract all data from this invoice."}] + image_blocks
        else:
            user_content = [{"text": f"{prompt}\n\nINVOICE TEXT:\n{source_text}"}]

        result = _call_bedrock(
            model_id,
            "You are a precise invoice data extraction system. Return valid JSON only.",
            user_content,
        )
    except Exception as e:
        if accumulator.tracer:
            accumulator.tracer.step('extract_chunk', chunk=chunk_index, error=str(e))
        return f"Chunk {chunk_index}: Extraction failed: {e}"

    parsed = _parse_json(result['text'])
    items = parsed.get('LineItems', [])
    truncated = result.get('truncated', False)

    if not accumulator.invoice_header:
        header_fields = [
            'Classification', 'InvoiceNo', 'Date', 'InvoiceCurrency',
            'FreightTerms', 'IncoTerms', 'TermsOfPayment', 'Exporter', 'Importer',
        ]
        accumulator.invoice_header = {k: parsed.get(k) for k in header_fields if parsed.get(k) is not None}

    if truncated and chunk_pages >= 1:
        logger.info(f"[TOOL] chunk {chunk_index}: TRUNCATED — using strip-based extraction ({chunk_pages} pages)")
        items = []
        detected_columns = ""
        for page_idx in range(chunk_pages):
            strips = _render_page_strips(chunk_path, page_idx, strips_per_page=5, overlap_pct=0.2)
            for strip_idx, strip_block in enumerate(strips):
                try:
                    column_hint = ""
                    if detected_columns:
                        column_hint = (
                            f"\n\nIMPORTANT — Column order detected from header row: {detected_columns}. "
                            f"Use this EXACT column mapping. Do NOT swap PartNo and ItemCode."
                        )

                    strip_content = [
                        {"text": f"{prompt}\n\nThis is strip {strip_idx+1} of {len(strips)} from page {page_idx+1}. "
                                 f"Extract ALL line items visible in this strip. Return only items you can see."
                                 f"{column_hint}"},
                        strip_block,
                    ]
                    strip_result = _call_bedrock(
                        model_id,
                        "You are a precise invoice data extraction system. Return valid JSON only.",
                        strip_content,
                    )
                    strip_parsed = _parse_json(strip_result['text'])
                    strip_items = strip_parsed.get('LineItems', [])

                    if not accumulator.invoice_header and page_idx == 0 and strip_idx == 0:
                        header_fields = [
                            'Classification', 'InvoiceNo', 'Date', 'InvoiceCurrency',
                            'FreightTerms', 'IncoTerms', 'TermsOfPayment', 'Exporter', 'Importer',
                        ]
                        accumulator.invoice_header = {
                            k: strip_parsed.get(k) for k in header_fields if strip_parsed.get(k) is not None
                        }

                    if not detected_columns and strip_items and page_idx == 0 and strip_idx <= 1:
                        sample = strip_items[0]
                        cols = []
                        for col_name in ['PartNo', 'ItemCode', 'ItemDescription', 'Quantity', 'UnitOfQty', 'UnitPrice', 'RITC', 'CountryOfOrigin']:
                            val = sample.get(col_name)
                            if val is not None:
                                cols.append(f"{col_name}='{val}'")
                        if cols:
                            detected_columns = ", ".join(cols)
                            logger.info(f"[STRIP] Detected column mapping from strip {strip_idx}: {detected_columns}")

                    items.extend(strip_items)
                    logger.info(f"[TOOL] chunk {chunk_index} page {page_idx} strip {strip_idx}: {len(strip_items)} items")
                except Exception as e:
                    logger.warning(f"[TOOL] chunk {chunk_index} page {page_idx} strip {strip_idx}: failed: {e}")

        pre_dedup_count = len(items)
        items = _intra_chunk_dedup(items)
        logger.info(f"[TOOL] chunk {chunk_index}: intra-chunk dedup {pre_dedup_count} → {len(items)} items")
        truncated = False

    # ─── CODEGEN: generate template for structured invoices ──
    if (not use_vision and not truncated and chunk_index == 0
            and accumulator.format_key != "unknown"
            and not template_registry.lookup(accumulator.format_key)
            and len(items) >= 1):
        try:
            # Detect column headers from parsed response
            column_headers = []
            if items:
                column_headers = [k for k in items[0].keys() if not k.startswith('_')]

            codegen_out = generate_extractor(
                sample_text=source_text,
                column_headers=column_headers,
                format_key=accumulator.format_key,
            )
            if codegen_out.get('structured') and codegen_out.get('code'):
                # Validate the generated code runs without error on current text
                test_result = run_extractor(codegen_out['code'], source_text)
                if test_result.get('success') and test_result.get('LineItems'):
                    template_id = codegen_out['template_id'] or accumulator.format_key
                    template_registry.save(template_id, codegen_out['code'], {
                        'column_headers': codegen_out.get('column_headers', column_headers),
                        'format_key': accumulator.format_key,
                        'company': accumulator.invoice_header.get('Exporter', {}).get('Name')
                                   if isinstance(accumulator.invoice_header.get('Exporter'), dict)
                                   else accumulator.invoice_header.get('Exporter'),
                    })
                    logger.info(f"[CODEGEN] Saved template: {template_id}")
                else:
                    logger.warning(f"[CODEGEN] Generated code failed validation, not saving")
            else:
                logger.info(f"[CODEGEN] Invoice not structured enough for codegen")
        except Exception as e:
            logger.warning(f"[CODEGEN] Code generation failed: {e}")

    for i, item in enumerate(items, 1):
        item['_chunk_index'] = chunk_index
        item['ItemNo'] = i

    accumulator.chunk_extractions[chunk_index] = items
    accumulator.processed_indices.add(chunk_index)
    accumulator.chunk_model_used[chunk_index] = model_id

    model_name = "Claude Sonnet"
    if accumulator.tracer:
        strip_note = ""
        if not truncated and chunk_pages >= 1:
            strip_note = f" (strips: {chunk_pages}p x 5)"
        accumulator.tracer.step(
            'extract_chunk', chunk=chunk_index,
            result=f"{len(items)} items{strip_note}{' (vision)' if use_vision else ''}",
            tokens_in=result['input_tokens'], tokens_out=result['output_tokens'],
            extra={'items_found': len(items), 'truncated': truncated, 'vision': use_vision, 'model': model_name},
        )

    vision_note = " (vision mode — scanned PDF)" if use_vision else ""
    return (
        f"Chunk {chunk_index}: {len(items)} items extracted{vision_note}. "
        f"({result['input_tokens']} in, {result['output_tokens']} out). "
        f"Progress: {len(accumulator.processed_indices)}/{accumulator.total_chunks} chunks. "
        f"Now call review_chunk({chunk_index}) to verify."
    )


# ─── TOOL 3: REVIEW CHUNK ─────────────────────────────────

@tool
def review_chunk(chunk_index: int) -> str:
    """Review an extraction by comparing it against the source text.
    Uses a SEPARATE LLM call to cross-check. Finds missing items,
    hallucinations, math errors, and duplicates.

    Args:
        chunk_index: Chunk index to review

    Returns:
        Review result: passed/failed, issue count, specific issues found
    """
    global accumulator

    source_text = accumulator.chunk_source_texts.get(chunk_index)
    extraction = accumulator.chunk_extractions.get(chunk_index, [])

    if source_text is None:
        return f"ERROR: No source text for chunk {chunk_index}. Call extract_chunk first."

    extraction_json = json.dumps(extraction, indent=2, default=str)
    is_vision = source_text == "[scanned PDF — extracted via vision]"

    try:
        if is_vision:
            chunk_path = accumulator.chunk_paths[chunk_index]
            image_blocks = _render_pdf_to_images(chunk_path)
            user_content = [
                {"text": (
                    f"EXTRACTION ({len(extraction)} items):\n"
                    f"{'='*40}\n{extraction_json}\n{'='*40}\n\n"
                    f"The source PDF pages are attached as images.\n"
                    f"Review this extraction against the images. Find every discrepancy."
                )},
            ] + image_blocks
        else:
            user_content = [{"text": (
                f"SOURCE TEXT (from PDF chunk {chunk_index}):\n"
                f"{'='*40}\n{source_text}\n{'='*40}\n\n"
                f"EXTRACTION ({len(extraction)} items):\n"
                f"{'='*40}\n{extraction_json}\n{'='*40}\n\n"
                f"Review this extraction. Find every discrepancy."
            )}]

        result = _call_bedrock(
            REVIEW_MODEL,
            REVIEW_PROMPT,
            user_content,
        )
    except Exception as e:
        if accumulator.tracer:
            accumulator.tracer.step('review_chunk', chunk=chunk_index, error=str(e))
        return f"Chunk {chunk_index}: Review failed: {e}"

    review = _parse_json(result['text'])

    passed = review.get('passed', False)
    issues = review.get('issues', [])
    expected = review.get('expectedItemCount', '?')
    extracted = review.get('extractedItemCount', len(extraction))
    score = review.get('score', 0)
    summary = review.get('summary', '')

    critical_issues = [i for i in issues if i.get('severity') == 'CRITICAL']
    warnings = [i for i in issues if i.get('severity') == 'WARNING']

    if score:
        accumulator.scores.append(score)

    accumulator.reviewed_indices.add(chunk_index)
    accumulator.review_summary[f'chunk_{chunk_index}'] = {
        'passed': passed, 'score': score,
        'expected': expected, 'extracted': extracted,
        'critical': len(critical_issues), 'warnings': len(warnings),
    }

    if accumulator.tracer:
        status = "PASSED" if passed else f"FAILED ({len(critical_issues)} critical)"
        accumulator.tracer.step(
            'review_chunk', chunk=chunk_index,
            result=f"{status} score={score} ({extracted}/{expected} items)",
            tokens_in=result['input_tokens'], tokens_out=result.get('output_tokens', 0),
            extra={'passed': passed, 'score': score, 'critical': len(critical_issues)},
        )

    retries = accumulator.chunk_retry_count.get(chunk_index, 0)

    # Soft pass for scanned/vision docs: more lenient since OCR is inherently noisy
    missing_items = [i for i in critical_issues if i.get('type') == 'MISSING_ITEM']
    soft_pass = (score >= 0.6 and extracted == expected and not missing_items)

    if passed or soft_pass:
        label = "PASSED" if passed else "SOFT-PASSED"
        return (
            f"Chunk {chunk_index} {label} review. Score: {score}. "
            f"Expected: {expected}, Extracted: {extracted}. "
            f"{len(warnings)} warnings. {summary}"
        )

    issue_details = "\n".join(
        f"  - [{i.get('severity','?')}] {i.get('type','?')}: {i.get('description','')} "
        f"(fix: {i.get('fixInstruction','')})"
        for i in critical_issues[:5]
    )

    if retries >= 2:
        # Try pattern matching before giving up
        matched = pattern_library.match_errors(
            errors=issues,
            source_text=accumulator.chunk_source_texts.get(chunk_index, ''),
            invoice_header=accumulator.invoice_header,
            format_key=accumulator.format_key,
        )
        if matched:
            accumulator.matched_patterns = matched
            fix_prompt = pattern_library.build_fix_prompt(matched)
            pattern_names = ', '.join(m['pattern_id'] for m in matched)
            return (
                f"Chunk {chunk_index} FAILED review (retry {retries}). Score: {score}. "
                f"Expected: {expected}, Extracted: {extracted}. "
                f"{len(critical_issues)} CRITICAL issues:\n{issue_details}\n"
                f"PATTERN MATCH found: {pattern_names}. "
                f"Call re_extract({chunk_index}) with these pattern-based instructions:\n{fix_prompt}"
            )
        return (
            f"Chunk {chunk_index} FAILED review (retry {retries}). Score: {score}. "
            f"Expected: {expected}, Extracted: {extracted}. "
            f"{len(critical_issues)} CRITICAL issues:\n{issue_details}\n"
            f"Max retries reached. No matching patterns found. Move on to next chunk. "
            f"This invoice is flagged for human review."
        )

    return (
        f"Chunk {chunk_index} FAILED review (retry {retries}). Score: {score}. "
        f"Expected: {expected}, Extracted: {extracted}. "
        f"{len(critical_issues)} CRITICAL issues:\n{issue_details}\n"
        f"Call re_extract({chunk_index}) with these issues to fix."
    )


# ─── TOOL 4: RE-EXTRACT ───────────────────────────────────

@tool
def re_extract(chunk_index: int, fix_instructions: str) -> str:
    """Re-extract specific items that were missed or wrong.
    Takes the reviewer's fix instructions and does a focused extraction.

    Args:
        chunk_index: Chunk index to re-extract from
        fix_instructions: Specific issues and instructions from review_chunk

    Returns:
        Summary of additional/corrected items found
    """
    global accumulator

    source_text = accumulator.chunk_source_texts.get(chunk_index)
    if source_text is None:
        return f"ERROR: No source text for chunk {chunk_index}."

    existing = accumulator.chunk_extractions.get(chunk_index, [])
    existing_json = json.dumps(existing[:5], indent=2, default=str)

    is_vision = source_text == "[scanned PDF — extracted via vision]"
    model_id = accumulator.chunk_model_used.get(chunk_index, EXTRACT_MODEL_PRIMARY)

    # Inject matched pattern fix prompts if available
    pattern_section = ""
    if accumulator.matched_patterns:
        pattern_section = "\n\n" + pattern_library.build_fix_prompt(accumulator.matched_patterns) + "\n"

    # Inject KB context if available
    kb_section = ""
    if accumulator.kb_context:
        kb_section = f"\n\n{accumulator.kb_context}\n"

    try:
        text_block = {"text": (
            f"{RE_EXTRACT_PROMPT}\n\n"
            f"EXISTING EXTRACTION (first 5 items for reference):\n{existing_json}\n\n"
            f"REVIEWER ISSUES:\n{fix_instructions}\n"
            f"{pattern_section}{kb_section}\n"
            f"Find and return the missing/corrected items."
        )}

        if is_vision:
            chunk_path = accumulator.chunk_paths[chunk_index]
            image_blocks = _render_pdf_to_images(chunk_path)
            user_content = [text_block] + image_blocks
        else:
            text_block["text"] = (
                f"{RE_EXTRACT_PROMPT}\n\n"
                f"SOURCE TEXT:\n{'='*40}\n{source_text}\n{'='*40}\n\n"
                f"EXISTING EXTRACTION (first 5 items for reference):\n{existing_json}\n\n"
                f"REVIEWER ISSUES:\n{fix_instructions}\n"
                f"{pattern_section}{kb_section}\n"
                f"Find and return the missing/corrected items."
            )
            user_content = [text_block]

        result = _call_bedrock(
            model_id,
            "You fix invoice extraction errors. Return valid JSON only.",
            user_content,
        )
    except Exception as e:
        if accumulator.tracer:
            accumulator.tracer.step('re_extract', chunk=chunk_index, error=str(e))
        return f"Chunk {chunk_index}: Re-extraction failed: {e}"

    parsed = _parse_json(result['text'])
    additional = parsed.get('additional_items', [])
    corrected = parsed.get('corrected_items', [])

    for item in additional:
        item['_chunk_index'] = chunk_index
        item['_re_extracted'] = True

    # Add new items to chunk extraction
    accumulator.chunk_extractions[chunk_index].extend(additional)

    # Apply corrections to existing items
    corrections_applied = 0
    for fix in corrected:
        orig_part = fix.get('original_PartNo', '')
        for existing_item in accumulator.chunk_extractions[chunk_index]:
            if existing_item.get('PartNo') == orig_part:
                for k, v in fix.items():
                    if k != 'original_PartNo' and v is not None:
                        existing_item[k] = v
                corrections_applied += 1
                break

    accumulator.chunk_retry_count[chunk_index] = accumulator.chunk_retry_count.get(chunk_index, 0) + 1
    model_name = "Nova Pro" if "nova" in model_id else "Claude Sonnet"

    if accumulator.tracer:
        accumulator.tracer.step(
            're_extract', chunk=chunk_index,
            result=f"+{len(additional)} new, {corrections_applied} fixed (retry {accumulator.chunk_retry_count[chunk_index]})",
            tokens_in=result['input_tokens'], tokens_out=result.get('output_tokens', 0),
            extra={'additional': len(additional), 'corrected': corrections_applied,
                   'model': model_name, 'retry': accumulator.chunk_retry_count[chunk_index]},
        )

    return (
        f"Re-extraction for chunk {chunk_index} (model: {model_name}, retry {accumulator.chunk_retry_count[chunk_index]}): "
        f"{len(additional)} new items found, {corrections_applied} items corrected. "
        f"Chunk now has {len(accumulator.chunk_extractions[chunk_index])} total items. "
        f"Call review_chunk({chunk_index}) again to verify."
    )


# ─── TOOL 5: PROCESS ALL CHUNKS (PARALLEL) ───────────────

def _process_single_chunk(chunk_index: int, max_retries: int = 2) -> str:
    """Process one chunk: extract → review → retry if needed. Thread-safe."""
    extract_result = extract_chunk(chunk_index)
    if "Extraction failed" in extract_result:
        return extract_result

    review_result = review_chunk(chunk_index)

    retries = 0
    used_patterns = False
    while "FAILED" in review_result and retries < max_retries:
        if "PATTERN MATCH" in review_result:
            used_patterns = True
        fix_instructions = review_result.split("CRITICAL issues:\n")[-1] if "CRITICAL issues:" in review_result else review_result
        re_extract(chunk_index, fix_instructions)
        review_result = review_chunk(chunk_index)
        retries += 1

    if used_patterns and accumulator.matched_patterns:
        passed = "PASSED" in review_result or "SOFT-PASSED" in review_result
        for mp in accumulator.matched_patterns:
            pattern_library.update_confidence(mp['pattern_id'], worked=passed)

    return f"Chunk {chunk_index}: {len(accumulator.chunk_extractions.get(chunk_index, []))} items, review: {review_result[:80]}"


@tool
def process_all_chunks() -> str:
    """Process all chunks in parallel: extract + review + retry for each.
    Chunk 0 runs first (format detection), then remaining chunks run concurrently.

    Returns:
        Summary of all chunks processed with item counts
    """
    global accumulator

    total = accumulator.total_chunks
    if total == 0:
        return "ERROR: No chunks. Call split_pdf first."

    logger.info(f"[TOOL] process_all_chunks: {total} chunks, chunk 0 first then parallel")
    start = time.time()

    # Chunk 0 first — needs format detection, KB context
    result_0 = _process_single_chunk(0)
    logger.info(f"[PARALLEL] chunk 0 done: {result_0[:80]}")

    results = {0: result_0}

    # Remaining chunks in parallel
    if total > 1:
        max_workers = min(total - 1, 5)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_process_single_chunk, i): i
                for i in range(1, total)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                    logger.info(f"[PARALLEL] chunk {idx} done")
                except Exception as e:
                    results[idx] = f"Chunk {idx}: FAILED: {e}"
                    logger.error(f"[PARALLEL] chunk {idx} failed: {e}")

    elapsed = time.time() - start

    total_items = sum(len(accumulator.chunk_extractions.get(i, [])) for i in range(total))
    chunk_summary = "\n".join(
        f"  Chunk {i}: {len(accumulator.chunk_extractions.get(i, []))} items"
        for i in range(total)
    )

    workers = min(total - 1, 5) if total > 1 else 1
    if accumulator.tracer:
        accumulator.tracer.step(
            'process_all_chunks',
            result=f"{total} chunks done, {total_items} items total ({workers} parallel)",
            duration=elapsed,
            extra={'total_chunks': total, 'total_items': total_items,
                   'parallel_workers': workers, 'elapsed': round(elapsed, 1)},
        )

    return (
        f"All {total} chunks processed in {elapsed:.1f}s ({min(total-1, 5) if total > 1 else 1} parallel workers):\n"
        f"{chunk_summary}\n"
        f"Total items before dedup: {total_items}\n"
        f"Now call deduplicate() to remove overlaps."
    )


# ─── TOOL 6: DEDUPLICATE ──────────────────────────────────

@tool
def deduplicate() -> str:
    """Remove duplicate items across all chunks. Uses three-layer matching:
    Layer 1 — Exact: same PartNo + Qty + UnitPrice (overlap duplicates)
    Layer 2 — Fuzzy: OCR variations (e.g. O vs 0, spacing differences)
    Layer 3 — Business: same part, different price → KEEP (legitimate items)

    Returns:
        Dedup report with counts of exact dupes, fuzzy dupes, and kept items
    """
    global accumulator

    # Collect all items from all chunks
    all_items = []
    for idx in sorted(accumulator.chunk_extractions.keys()):
        all_items.extend(accumulator.chunk_extractions[idx])

    if not all_items:
        return "No items to deduplicate."

    # Layer 1: Exact match (overlap rows)
    seen_exact = {}
    exact_dupes = 0
    after_exact = []
    for item in all_items:
        key = (
            _normalize_str(item.get('PartNo', '')),
            _normalize_str(item.get('ItemDescription', ''))[:80],
            _normalize_num(item.get('Quantity')),
            _normalize_num(item.get('UnitPrice')),
            _normalize_str(item.get('RITC', '')),
        )
        if key in seen_exact:
            exact_dupes += 1
        else:
            seen_exact[key] = item
            after_exact.append(item)

    # Layer 2: Fuzzy match (OCR variations)
    fuzzy_dupes = 0
    fuzzy_groups = []
    after_fuzzy = []
    used = set()
    for i, item_a in enumerate(after_exact):
        if i in used:
            continue
        group = [i]
        for j, item_b in enumerate(after_exact):
            if j <= i or j in used:
                continue
            if _is_fuzzy_dupe(item_a, item_b):
                group.append(j)
                used.add(j)
                fuzzy_dupes += 1
        used.add(i)
        # Keep the first item from each fuzzy group
        after_fuzzy.append(after_exact[group[0]])
        if len(group) > 1:
            fuzzy_groups.append({
                'kept': _item_summary(after_exact[group[0]]),
                'removed': [_item_summary(after_exact[g]) for g in group[1:]],
            })

    # Layer 3: Business logic — same part, different price → already kept by Layer 1
    # (different UnitPrice means different exact key, so they survive)

    # Renumber
    for i, item in enumerate(after_fuzzy, 1):
        item['ItemNo'] = i
        item.pop('_chunk_index', None)
        item.pop('_re_extracted', None)

    accumulator.items = after_fuzzy
    accumulator.dedup_summary = {
        'before': len(all_items),
        'exact_dupes_removed': exact_dupes,
        'fuzzy_dupes_removed': fuzzy_dupes,
        'after': len(after_fuzzy),
        'fuzzy_groups': fuzzy_groups[:10],
    }

    if accumulator.tracer:
        accumulator.tracer.step(
            'deduplicate',
            result=f"{len(all_items)} -> {len(after_fuzzy)} (exact={exact_dupes}, fuzzy={fuzzy_dupes})",
            extra={'before': len(all_items), 'after': len(after_fuzzy),
                   'exact_dupes': exact_dupes, 'fuzzy_dupes': fuzzy_dupes},
        )

    return (
        f"Deduplication complete:\n"
        f"  Before: {len(all_items)} items\n"
        f"  Exact duplicates removed: {exact_dupes} (overlap rows)\n"
        f"  Fuzzy duplicates removed: {fuzzy_dupes} (OCR variations)\n"
        f"  After: {len(after_fuzzy)} unique items\n"
        f"Now call verify_final() for the completeness report."
    )


# ─── TOOL 6: VERIFY FINAL ─────────────────────────────────

@tool
def verify_final() -> str:
    """Final verification of the complete extraction.
    Checks: all chunks processed, all chunks reviewed, item count,
    math consistency, and overall quality.

    Returns:
        PASS or FAIL with detailed completeness report
    """
    global accumulator

    items = accumulator.items
    total_chunks = accumulator.total_chunks

    # Check all chunks processed
    unprocessed = [i for i in range(total_chunks) if i not in accumulator.processed_indices]
    unreviewed = [i for i in range(total_chunks) if i not in accumulator.reviewed_indices]

    # Quality checks
    null_desc = sum(1 for item in items if not item.get('ItemDescription'))
    null_qty = sum(1 for item in items if item.get('Quantity') is None)

    # Review scores
    chunk_scores = accumulator.review_summary
    failed_chunks = [k for k, v in chunk_scores.items() if not v.get('passed')]

    # Classification check
    classification = accumulator.invoice_header.get('Classification', '')
    is_invoice = classification and 'invoice' in str(classification).lower()

    # Overall assessment
    issues = []
    if unprocessed:
        issues.append(f"Unprocessed chunks: {unprocessed}")
    if unreviewed:
        issues.append(f"Unreviewed chunks: {unreviewed}")
    if null_desc > 0:
        issues.append(f"Items missing description: {null_desc}")
    if null_qty > len(items) * 0.5 and len(items) > 0:
        issues.append(f"Items missing quantity: {null_qty}/{len(items)}")
    if failed_chunks:
        issues.append(f"Failed review: {failed_chunks}")
    if classification and not is_invoice:
        issues.append(f"Document classified as: {classification} (not invoice)")

    passed = len(issues) == 0 or (
        len(unprocessed) == 0 and len(unreviewed) == 0
    )

    avg_score = (
        sum(accumulator.scores) / len(accumulator.scores)
        if accumulator.scores else 0
    )

    if accumulator.tracer:
        accumulator.tracer.step(
            'verify_final',
            result=f"{'PASS' if passed else 'FAIL'} — {len(items)} items, avg score {avg_score:.2f}",
            extra={'passed': passed, 'items': len(items)},
        )
        accumulator.tracer.finish(
            total_items=len(items),
            quality_score=avg_score,
            passed=passed,
        )

    report = (
        f"{'PASS' if passed else 'FAIL'}: Final Verification Report\n"
        f"{'='*50}\n"
        f"  Classification:     {classification or 'unknown'}\n"
        f"  Total items:        {len(items)}\n"
        f"  Chunks processed:   {len(accumulator.processed_indices)}/{total_chunks}\n"
        f"  Chunks reviewed:    {len(accumulator.reviewed_indices)}/{total_chunks}\n"
        f"  Average score:      {avg_score:.2f}\n"
        f"  Dedup:              {accumulator.dedup_summary.get('exact_dupes_removed', 0)} exact + "
        f"{accumulator.dedup_summary.get('fuzzy_dupes_removed', 0)} fuzzy removed\n"
    )
    if issues:
        report += f"  Issues: {'; '.join(issues)}\n"

    return report


# ─── DEDUP HELPERS ─────────────────────────────────────────

def _normalize_str(s) -> str:
    if s is None:
        return ''
    return re.sub(r'[^a-zA-Z0-9]', '', str(s)).lower()


def _normalize_num(v) -> str:
    f = _to_float(v)
    return f"{f:.4f}" if f is not None else str(v)


def _to_float(v):
    if v is None:
        return None
    try:
        return float(str(v).replace(',', '').replace('$', '').strip())
    except (ValueError, TypeError):
        return None


def _is_fuzzy_dupe(a: dict, b: dict) -> bool:
    """Check if two items are fuzzy duplicates (OCR noise)."""
    part_a = _normalize_str(a.get('PartNo', ''))
    part_b = _normalize_str(b.get('PartNo', ''))

    if part_a and part_b:
        if fuzz.ratio(part_a, part_b) < 85:
            return False
    elif part_a != part_b:
        return False

    qty_a = _normalize_num(a.get('Quantity'))
    qty_b = _normalize_num(b.get('Quantity'))
    if qty_a != qty_b:
        return False

    price_a = _normalize_num(a.get('UnitPrice'))
    price_b = _normalize_num(b.get('UnitPrice'))
    if price_a != price_b:
        return False

    desc_a = str(a.get('ItemDescription', ''))
    desc_b = str(b.get('ItemDescription', ''))
    if desc_a and desc_b:
        if fuzz.ratio(desc_a.lower(), desc_b.lower()) < 90:
            return False

    return True


def _item_summary(item: dict) -> str:
    return f"{item.get('PartNo', '?')} | {str(item.get('ItemDescription', ''))[:30]} | qty={item.get('Quantity')}"


def _intra_chunk_dedup(items: list) -> list:
    """Remove exact duplicates within a single chunk (from strip overlap)."""
    seen = {}
    result = []
    for item in items:
        key = (
            _normalize_str(item.get('PartNo', '')),
            _normalize_str(item.get('ItemDescription', ''))[:80],
            _normalize_num(item.get('Quantity')),
            _normalize_num(item.get('UnitPrice')),
        )
        if key not in seen:
            seen[key] = True
            result.append(item)
    return result
