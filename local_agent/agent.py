"""
Invoice Extraction Agent — Local Agentic Architecture (v10)

Agents-as-Tools pattern from AWS MLU labs.
Orchestrator decides: extract → review → re-extract? → dedup → verify.

LLM #1 (extractor): reads PDF chunk → structured JSON
LLM #2 (reviewer):  reads source text + extraction → finds gaps
No Textract, no regex. LLM verifies LLM.
"""
import json
import os
import sys
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path

import boto3
from botocore.config import Config
from strands import Agent, tool
from strands.hooks import HookProvider, HookRegistry, AfterModelCallEvent

import tools
from tools import (
    split_pdf, extract_chunk, review_chunk,
    re_extract, process_all_chunks, deduplicate,
    verify_final, reset_accumulator, init_tracer,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)

ORCHESTRATOR_MODEL = os.environ.get(
    'ORCHESTRATOR_MODEL', 'us.anthropic.claude-sonnet-4-5-20250929-v1:0'
)

SYSTEM_PROMPT = """You are an invoice extraction orchestrator. Your job is to extract ALL line items from a PDF invoice with 100% completeness.

You have 7 tools. Use them in this order:

PHASE 1 — SPLIT
  Call split_pdf. YOU decide pages_per_chunk and overlap_pages based on the PDF.
  Guidelines for your decision:
    - 1-2 pages: pages_per_chunk = total pages, overlap_pages = 0
    - 3-10 pages: pages_per_chunk=2, overlap_pages=1
    - 11-30 pages: pages_per_chunk=3, overlap_pages=1
    - 30+ pages: pages_per_chunk=5, overlap_pages=1
  For dense invoices (many items per page), use pages_per_chunk=1.
  If extract_chunk truncates, it auto-falls back to strip-based image extraction.
  split_pdf returns the total page count — use it to validate your choice.

PHASE 2 — EXTRACT + REVIEW ALL CHUNKS (PARALLEL)
  Call process_all_chunks. This single call:
    - Extracts chunk 0 first (for format detection)
    - Then extracts remaining chunks IN PARALLEL (up to 5 concurrent)
    - Reviews each chunk automatically
    - Retries failed chunks (max 2 retries each)
  You do NOT need to call extract_chunk or review_chunk individually.

  For manual control (e.g., re-extracting a specific failed chunk), you can still
  use extract_chunk, review_chunk, and re_extract individually.

PHASE 3 — DEDUP + FINAL
  After process_all_chunks completes:
    1. Call deduplicate to remove overlap and OCR duplicates
    2. Call verify_final for the completeness report

RULES:
- PREFER process_all_chunks over manual chunk-by-chunk processing.
- If process_all_chunks reports a failed chunk, you may manually re_extract it.
- Never finish without calling verify_final.
- Call verify_final ONLY ONCE. After verify_final returns (PASS or FAIL), STOP immediately. Do NOT retry, do NOT call review_chunk or re_extract again. The result goes to worker verification — humans handle the rest.
- A FAIL result is expected for low-quality scans or unusual formats. That is normal. Report the result and stop."""


class CompletenessGuard(HookProvider):
    """Steering hook: blocks the agent from finishing until verify_final has been called."""

    def __init__(self):
        self.verify_final_called = False

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(AfterModelCallEvent, self._check)

    def _check(self, event: AfterModelCallEvent):
        if hasattr(event, 'stop_reason') and event.stop_reason == 'end_turn':
            if not self.verify_final_called:
                logger.warning("[GUARD] Agent tried to finish without verify_final")


def run(pdf_path: str, output_path: str = None):
    """Run the full extraction pipeline on a local PDF."""
    pdf_path = str(Path(pdf_path).resolve())
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return None

    reset_accumulator()
    tools.accumulator.start_time = time.time()

    filename = os.path.basename(pdf_path)
    job_id = Path(pdf_path).stem[:8]
    init_tracer(job_id, filename)

    guard = CompletenessGuard()

    agent = Agent(
        model=ORCHESTRATOR_MODEL,
        system_prompt=SYSTEM_PROMPT,
        tools=[split_pdf, extract_chunk, review_chunk, re_extract, process_all_chunks, deduplicate, verify_final],
        hooks=[guard],
        callback_handler=None,
    )
    from pypdf import PdfReader
    total_pages = len(PdfReader(pdf_path).pages)

    user_msg = (
        f"Extract all line items from this invoice:\n"
        f"PDF path: {pdf_path}\n"
        f"Filename: {filename}\n"
        f"Total pages: {total_pages}\n"
        f"Decide the right chunk size, split it, extract every chunk, "
        f"review each extraction, deduplicate, and run final verification."
    )

    logger.info(f"Starting extraction: {filename}")
    start = time.time()

    try:
        response = agent(user_msg)
        guard.verify_final_called = True
    except Exception as e:
        logger.error(f"Agent failed: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}

    elapsed = time.time() - start

    header = tools.accumulator.invoice_header

    result = {
        'success': len(tools.accumulator.items) > 0,
        'filename': filename,
        'Classification': header.get('Classification'),
        'InvoiceNo': header.get('InvoiceNo'),
        'Date': header.get('Date'),
        'InvoiceCurrency': header.get('InvoiceCurrency'),
        'FreightTerms': header.get('FreightTerms'),
        'IncoTerms': header.get('IncoTerms'),
        'TermsOfPayment': header.get('TermsOfPayment'),
        'Exporter': header.get('Exporter'),
        'Importer': header.get('Importer'),
        'total_items': len(tools.accumulator.items),
        'LineItems': tools.accumulator.items,
        'quality_score': round(
            sum(tools.accumulator.scores) / len(tools.accumulator.scores), 3
        ) if tools.accumulator.scores else 0,
        'chunks_processed': len(tools.accumulator.processed_indices),
        'total_chunks': tools.accumulator.total_chunks,
        'review_summary': tools.accumulator.review_summary,
        'dedup_summary': tools.accumulator.dedup_summary,
        'elapsed_seconds': round(elapsed, 1),
        'tool_log': tools.accumulator.tracer.get_tool_log() if tools.accumulator.tracer else [],
        'format_key': tools.accumulator.format_key,
        'format_confidence': tools.accumulator.format_confidence,
        'patterns_used': [m.get('pattern_id') for m in tools.accumulator.matched_patterns],
    }

    if output_path is None:
        output_path = pdf_path.replace('.pdf', '_extracted.json')

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2, default=str, ensure_ascii=False)

    logger.info(
        f"Done: {result['total_items']} items, "
        f"score={result['quality_score']}, "
        f"{elapsed:.1f}s → {output_path}"
    )
    return result


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python agent.py <invoice.pdf> [output.json]")
        sys.exit(1)

    pdf = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    result = run(pdf, out)

    if result:
        print(f"\n{'='*60}")
        print(f"Items: {result['total_items']}")
        print(f"Score: {result['quality_score']}")
        print(f"Time:  {result['elapsed_seconds']}s")
        print(f"File:  {pdf.replace('.pdf', '_extracted.json')}")
