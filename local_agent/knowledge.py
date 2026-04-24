"""
Knowledge Base — Format detection and context retrieval.

Loads domain knowledge from markdown files.
Detects invoice format using LLM.
Returns ONLY relevant KB context to inject into extraction prompt.
"""
import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import boto3
from botocore.config import Config

logger = logging.getLogger(__name__)

REGION = os.environ.get('AWS_REGION', 'us-east-1')
KB_MODEL = os.environ.get('KB_MODEL', 'us.anthropic.claude-sonnet-4-5-20250929-v1:0')
KB_DIR = Path(__file__).parent / 'data' / 'kb'

bedrock = boto3.client('bedrock-runtime', region_name=REGION, config=Config(
    read_timeout=300, retries={'max_attempts': 3}
))

# Maps format_key to human-readable name
FORMAT_NAMES = {
    "cfdi": "Mexican CFDI",
    "fapiao": "Chinese Fapiao",
    "eu_vat": "European VAT Invoice",
    "japanese": "Japanese Invoice",
    "gst": "Indian GST Invoice",
    "us_commercial": "US Commercial Invoice",
    "unknown": "Unknown Format",
}

# Keywords used to match markdown section headers to format keys
FORMAT_KEYWORDS = {
    "cfdi": ["cfdi", "mexican", "mexico", "factura"],
    "fapiao": ["fapiao", "chinese", "china"],
    "eu_vat": ["eu_vat", "european", "vat", "eu "],
    "japanese": ["japanese", "japan", "請求書"],
    "gst": ["gst", "indian", "india"],
    "us_commercial": ["us_commercial", "us commercial", "american", "united states"],
}

DETECT_SYSTEM = (
    "You identify invoice formats. Return ONLY valid JSON, no commentary."
)

DETECT_PROMPT_TEMPLATE = """Identify the regional format of this invoice.

Possible formats:
- "cfdi" — Mexican CFDI (Comprobante Fiscal Digital, RFC numbers, SAT stamps)
- "fapiao" — Chinese Fapiao (发票, tax registration numbers, Chinese characters)
- "eu_vat" — European VAT invoice (VAT numbers, EU member state, Intra-Community)
- "japanese" — Japanese invoice (請求書, Japanese characters, consumption tax)
- "gst" — Indian GST invoice (GSTIN numbers, HSN codes, Indian rupees)
- "us_commercial" — US/generic commercial invoice (USD, US addresses, no special tax regime)
- "unknown" — Cannot determine

{header_context}
INVOICE CONTENT (first page):
{content}

Return JSON: {{"format": "<key>", "confidence": 0.0-1.0, "reasoning": "<one sentence>"}}"""


class KnowledgeBase:
    """Loads and retrieves domain knowledge for invoice extraction."""

    def __init__(self, kb_dir: Path = KB_DIR):
        self.kb_dir = kb_dir
        self.sections: Dict[str, str] = {}  # format_key -> content
        self.file_sections: Dict[str, Dict[str, str]] = {}  # filename -> {header -> content}
        self._load_all()

    def _load_all(self):
        """Load all KB markdown files and parse into sections."""
        if not self.kb_dir.exists():
            logger.warning(f"KB directory not found: {self.kb_dir}")
            return

        for md_file in sorted(self.kb_dir.glob('*.md')):
            try:
                text = md_file.read_text(encoding='utf-8')
            except Exception as e:
                logger.warning(f"Failed to read {md_file}: {e}")
                continue

            filename = md_file.stem  # e.g. "regional_formats", "document_types"
            parsed = self._parse_sections(text)
            self.file_sections[filename] = parsed

            # Map section headers to format keys
            for header, content in parsed.items():
                header_lower = header.lower()
                for fmt_key, keywords in FORMAT_KEYWORDS.items():
                    if any(kw in header_lower for kw in keywords):
                        # Append if key already exists (multiple files may contribute)
                        if fmt_key in self.sections:
                            self.sections[fmt_key] += f"\n\n{content}"
                        else:
                            self.sections[fmt_key] = content
                        break

        loaded = len(self.file_sections)
        mapped = len(self.sections)
        logger.info(f"KB loaded: {loaded} files, {mapped} format sections mapped")

    def _parse_sections(self, text: str) -> Dict[str, str]:
        """Split markdown by ## headers into {header: content} dict."""
        sections = {}
        # Split on ## headers (level 2)
        parts = re.split(r'^(## .+)$', text, flags=re.MULTILINE)

        # parts[0] is text before first ## header (preamble)
        # parts[1], parts[2] = header, content; parts[3], parts[4] = header, content; ...
        if parts[0].strip():
            sections['_preamble'] = parts[0].strip()

        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                header = parts[i].lstrip('#').strip()
                content = parts[i + 1].strip()
                sections[header] = content

        return sections

    def detect_format(self, source_text: str = None, images: list = None,
                      invoice_header: dict = None) -> Tuple[str, float]:
        """
        Detect invoice format using LLM.

        Sends first page text/image + available header info to LLM.
        LLM returns format identifier + confidence.

        Returns: (format_key, confidence) e.g. ("cfdi", 0.95)
        """
        # Build header context from available info
        header_context = ""
        if invoice_header:
            hints = []
            if invoice_header.get('Exporter', {}).get('Name'):
                hints.append(f"Exporter: {invoice_header['Exporter']['Name']}")
            if invoice_header.get('Exporter', {}).get('Address'):
                hints.append(f"Exporter Address: {invoice_header['Exporter']['Address']}")
            if invoice_header.get('InvoiceCurrency'):
                hints.append(f"Currency: {invoice_header['InvoiceCurrency']}")
            if invoice_header.get('InvoiceNo'):
                hints.append(f"Invoice No: {invoice_header['InvoiceNo']}")
            if hints:
                header_context = "HEADER INFO (already extracted):\n" + "\n".join(hints) + "\n\n"

        # Build content — first 2000 chars of text, or images
        content = ""
        if source_text and source_text.strip() and source_text != "[scanned PDF — extracted via vision]":
            content = source_text[:2000]

        prompt = DETECT_PROMPT_TEMPLATE.format(
            header_context=header_context,
            content=content if content else "(no text available — see attached images)",
        )

        # Build user content blocks
        user_content = [{"text": prompt}]
        if images and not content:
            # Add images for scanned PDFs (limit to first 2 pages for cost)
            for img_block in images[:2]:
                user_content.append(img_block)

        try:
            response = bedrock.converse(
                modelId=KB_MODEL,
                system=[{"text": DETECT_SYSTEM}],
                messages=[{"role": "user", "content": user_content}],
                inferenceConfig={"maxTokens": 500, "temperature": 0.0}
            )
            text = response['output']['message']['content'][0]['text']
        except Exception as e:
            logger.error(f"Format detection LLM call failed: {e}")
            return ("unknown", 0.0)

        # Parse response
        try:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
            else:
                parsed = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse format detection response: {text[:200]}")
            return ("unknown", 0.0)

        fmt = parsed.get('format', 'unknown')
        confidence = float(parsed.get('confidence', 0.0))
        reasoning = parsed.get('reasoning', '')

        # Validate format key
        if fmt not in FORMAT_NAMES:
            logger.warning(f"LLM returned unknown format key '{fmt}', falling back to 'unknown'")
            fmt = "unknown"

        logger.info(f"Format detected: {fmt} (confidence={confidence:.2f}) — {reasoning}")
        return (fmt, confidence)

    def get_context(self, format_key: str) -> str:
        """
        Get relevant KB context for the detected format.

        Returns a string to append to the extraction prompt.
        Only includes sections relevant to this format.
        """
        if format_key == "unknown" or format_key not in self.sections:
            return ""

        parts = [f"CONTEXT-SPECIFIC KNOWLEDGE FOR THIS INVOICE FORMAT ({FORMAT_NAMES.get(format_key, format_key)}):\n"]

        # Add the regional format section
        parts.append(self.sections[format_key])

        # Add relevant field knowledge from other KB files
        for filename, file_secs in self.file_sections.items():
            for header, content in file_secs.items():
                if header.startswith('_'):
                    continue
                header_lower = header.lower()
                # Skip if this is the same section we already added
                already_keywords = FORMAT_KEYWORDS.get(format_key, [])
                if any(kw in header_lower for kw in already_keywords):
                    continue
                # Include field knowledge sections that mention this format
                if format_key in content.lower() or any(kw in content.lower() for kw in already_keywords):
                    parts.append(f"\n--- {header} ---\n{content}")

        return "\n".join(parts)

    def detect_and_retrieve(self, source_text: str = None, images: list = None,
                            invoice_header: dict = None) -> Dict:
        """
        One-call convenience: detect format + retrieve context.

        Returns: {
            "format": "cfdi",
            "confidence": 0.95,
            "context": "CONTEXT-SPECIFIC KNOWLEDGE:\n...",
            "format_name": "Mexican CFDI"
        }
        """
        fmt, confidence = self.detect_format(source_text, images, invoice_header)
        context = self.get_context(fmt)

        return {
            "format": fmt,
            "confidence": confidence,
            "context": context,
            "format_name": FORMAT_NAMES.get(fmt, "Unknown Format"),
        }
