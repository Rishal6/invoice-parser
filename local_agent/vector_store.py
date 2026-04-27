"""
Vector Store — FAISS-based invoice format recognition.

Flow:
1. Render page 1 as image
2. Vision LLM describes the layout structure (not data)
3. Embed the description with Titan Embed v2 → 1024-dim vector
4. Search FAISS for similar layouts
5. On successful extraction (score >= 0.8): save format + thumbnail

Persistence: local disk + optional S3 sync.
"""
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import boto3
import faiss
import fitz as pymupdf
import numpy as np
from botocore.config import Config

logger = logging.getLogger(__name__)

REGION = os.environ.get("AWS_REGION", "us-east-1")
EMBED_MODEL = "amazon.titan-embed-text-v2:0"
LAYOUT_MODEL = os.environ.get("LAYOUT_MODEL", "us.amazon.nova-lite-v1:0")
EMBED_DIM = 1024
SIMILARITY_THRESHOLD = 0.85
DUPLICATE_THRESHOLD = 0.95
MIN_QUALITY_SCORE = 0.8

STORE_DIR = Path(os.environ.get(
    "VECTOR_STORE_DIR",
    os.path.join(os.path.dirname(__file__), "data", "formats"),
))

bedrock = boto3.client(
    "bedrock-runtime",
    region_name=REGION,
    config=Config(read_timeout=120, retries={"max_attempts": 3}),
)

LAYOUT_DESCRIBE_PROMPT = """Describe ONLY the structural layout of this invoice. Do NOT extract any data values.

Return a single paragraph covering:
1. Column headers from left to right (exact names as printed)
2. Table position on the page (top/middle/bottom, single table or split)
3. Tax section: inline per row, summary at bottom, or separate section
4. Header area: what fields appear (invoice no, date, exporter, importer positions)
5. Footer area: totals, signatures, stamps, notes
6. Any distinguishing structural features (borders, shading, logo position, multi-currency columns)

Be specific about column ORDER. For example: "S.No | Part No | Description | HSN | Qty | Unit | Rate | CGST | SGST | Amount"

Do NOT include any actual values (no names, numbers, dates). Describe structure only."""


class FormatEntry:
    """One known invoice format in the store."""

    __slots__ = (
        "format_id", "layout_description", "extraction_context",
        "source_job_id", "quality_score", "times_matched", "created_at",
        "updated_at", "thumbnail_key",
    )

    def __init__(self, **kw):
        for s in self.__slots__:
            setattr(self, s, kw.get(s))

    def to_dict(self) -> dict:
        return {s: getattr(self, s) for s in self.__slots__}

    @classmethod
    def from_dict(cls, d: dict) -> "FormatEntry":
        return cls(**{k: d.get(k) for k in cls.__slots__})


class VectorStore:
    """FAISS-backed vector store for invoice layout fingerprints."""

    def __init__(self, store_dir: Path = STORE_DIR):
        self.store_dir = store_dir
        self.store_dir.mkdir(parents=True, exist_ok=True)

        self.index: faiss.IndexFlatIP = None
        self.entries: list[FormatEntry] = []
        self._load()

    # ── persistence ──────────────────────────────────────────

    def _index_path(self) -> Path:
        return self.store_dir / "index.faiss"

    def _meta_path(self) -> Path:
        return self.store_dir / "metadata.json"

    def _thumb_dir(self) -> Path:
        d = self.store_dir / "thumbnails"
        d.mkdir(exist_ok=True)
        return d

    def _load(self):
        idx_path = self._index_path()
        meta_path = self._meta_path()

        if idx_path.exists() and meta_path.exists():
            self.index = faiss.read_index(str(idx_path))
            raw = json.loads(meta_path.read_text())
            self.entries = [FormatEntry.from_dict(e) for e in raw]
            logger.info(f"[VECTOR] Loaded {len(self.entries)} formats from {self.store_dir}")
        else:
            self.index = faiss.IndexFlatIP(EMBED_DIM)
            self.entries = []
            logger.info("[VECTOR] Initialized empty vector store")

    def _save(self):
        faiss.write_index(self.index, str(self._index_path()))
        raw = [e.to_dict() for e in self.entries]
        self._meta_path().write_text(json.dumps(raw, indent=2, default=str))
        logger.info(f"[VECTOR] Saved {len(self.entries)} formats to {self.store_dir}")

    def sync_to_s3(self, bucket: str, prefix: str = "formats/"):
        """Upload index + metadata to S3 for persistence across deploys."""
        try:
            s3 = boto3.client("s3")
            s3.upload_file(str(self._index_path()), bucket, f"{prefix}index.faiss")
            s3.upload_file(str(self._meta_path()), bucket, f"{prefix}metadata.json")
            logger.info(f"[VECTOR] Synced to s3://{bucket}/{prefix}")
        except Exception as e:
            logger.error(f"[VECTOR] S3 sync failed: {e}")

    def sync_from_s3(self, bucket: str, prefix: str = "formats/"):
        """Download index + metadata from S3 on startup."""
        try:
            s3 = boto3.client("s3")
            s3.download_file(bucket, f"{prefix}index.faiss", str(self._index_path()))
            s3.download_file(bucket, f"{prefix}metadata.json", str(self._meta_path()))
            self._load()
            logger.info(f"[VECTOR] Loaded from s3://{bucket}/{prefix}")
        except Exception as e:
            logger.warning(f"[VECTOR] S3 load failed (starting fresh): {e}")

    # ── step 1: render page 1 as image ───────────────────────

    def render_page1_image(self, pdf_path: str, dpi: int = 150) -> bytes:
        """Render first page of PDF as PNG bytes."""
        doc = pymupdf.open(pdf_path)
        page = doc[0]
        pix = page.get_pixmap(dpi=dpi)
        png_bytes = pix.tobytes("png")
        doc.close()
        return png_bytes

    def render_thumbnail(self, pdf_path: str, dpi: int = 72) -> bytes:
        """Render a small thumbnail for storage/debugging."""
        doc = pymupdf.open(pdf_path)
        page = doc[0]
        pix = page.get_pixmap(dpi=dpi)
        png_bytes = pix.tobytes("png")
        doc.close()
        return png_bytes

    # ── step 2: vision LLM describes layout ──────────────────

    def describe_layout(self, pdf_path: str) -> str:
        """Send page 1 image to vision LLM, get structural description."""
        image_bytes = self.render_page1_image(pdf_path)

        try:
            response = bedrock.converse(
                modelId=LAYOUT_MODEL,
                system=[{"text": "You describe invoice layouts. Return only the description, no JSON."}],
                messages=[{
                    "role": "user",
                    "content": [
                        {"text": LAYOUT_DESCRIBE_PROMPT},
                        {"image": {"format": "png", "source": {"bytes": image_bytes}}},
                    ],
                }],
                inferenceConfig={"maxTokens": 500, "temperature": 0.0},
            )
            description = response["output"]["message"]["content"][0]["text"]
            logger.info(f"[VECTOR] Layout described: {description[:100]}...")
            return description
        except Exception as e:
            logger.error(f"[VECTOR] Layout description failed: {e}")
            return ""

    # ── step 3: embed with Titan ─────────────────────────────

    def embed(self, text: str) -> np.ndarray:
        """Embed text using Bedrock Titan Embed v2. Returns normalized 1024-dim vector."""
        response = bedrock.invoke_model(
            modelId=EMBED_MODEL,
            body=json.dumps({
                "inputText": text,
                "dimensions": EMBED_DIM,
                "normalize": True,
            }),
        )
        body = json.loads(response["body"].read())
        vec = np.array(body["embedding"], dtype=np.float32)
        return vec

    # ── step 4: search ───────────────────────────────────────

    def search(self, layout_description: str, k: int = 3) -> list[dict]:
        """Search for similar layouts. Returns list of {entry, score}."""
        if not layout_description or self.index.ntotal == 0:
            return []

        query_vec = self.embed(layout_description).reshape(1, -1)
        distances, indices = self.index.search(query_vec, min(k, self.index.ntotal))

        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.entries):
                continue
            results.append({
                "entry": self.entries[idx],
                "score": float(score),
            })
        return results

    def find_matching_format(self, pdf_path: str) -> Optional[dict]:
        """
        Full search pipeline: render → describe → embed → search.

        Returns {entry, score, layout_description} if match found, else None.
        """
        layout_desc = self.describe_layout(pdf_path)
        if not layout_desc:
            return None

        results = self.search(layout_desc)
        if results and results[0]["score"] >= SIMILARITY_THRESHOLD:
            best = results[0]
            entry = best["entry"]
            entry.times_matched = (entry.times_matched or 0) + 1
            entry.updated_at = time.time()
            self._save()
            logger.info(
                f"[VECTOR] Match found: {entry.format_id} "
                f"(score={best['score']:.3f}, matched {entry.times_matched} times)"
            )
            return {
                "entry": entry,
                "score": best["score"],
                "layout_description": layout_desc,
            }

        logger.info(f"[VECTOR] No match (best={results[0]['score']:.3f})" if results else "[VECTOR] No match (empty store)")
        return None

    # ── step 8: save on success ──────────────────────────────

    def save_format(
        self,
        pdf_path: str,
        layout_description: str,
        extraction_result: dict,
        quality_score: float,
        job_id: str = "",
    ) -> Optional[str]:
        """
        Save a new format after successful extraction.

        Gates:
        - quality_score >= 0.8
        - Not a near-duplicate (score < 0.95 against existing)
        """
        if quality_score < MIN_QUALITY_SCORE:
            logger.info(f"[VECTOR] Skip save: quality {quality_score:.2f} < {MIN_QUALITY_SCORE}")
            return None

        # Check for near-duplicate
        results = self.search(layout_description)
        if results and results[0]["score"] >= DUPLICATE_THRESHOLD:
            existing = results[0]["entry"]
            existing.times_matched = (existing.times_matched or 0) + 1
            existing.updated_at = time.time()
            if quality_score > (existing.quality_score or 0):
                existing.extraction_context = self._build_context(extraction_result)
                existing.quality_score = quality_score
                existing.source_job_id = job_id
            self._save()
            logger.info(f"[VECTOR] Near-duplicate of {existing.format_id}, updated stats")
            return existing.format_id

        # Build extraction context from successful result
        context = self._build_context(extraction_result)

        # Generate format ID
        format_id = "fmt_" + hashlib.md5(layout_description.encode()).hexdigest()[:12]

        # Save thumbnail
        thumb_key = ""
        try:
            thumb_bytes = self.render_thumbnail(pdf_path)
            thumb_path = self._thumb_dir() / f"{format_id}.png"
            thumb_path.write_bytes(thumb_bytes)
            thumb_key = f"thumbnails/{format_id}.png"
        except Exception as e:
            logger.warning(f"[VECTOR] Thumbnail save failed: {e}")

        entry = FormatEntry(
            format_id=format_id,
            layout_description=layout_description,
            extraction_context=context,
            source_job_id=job_id,
            quality_score=quality_score,
            times_matched=1,
            created_at=time.time(),
            updated_at=time.time(),
            thumbnail_key=thumb_key,
        )

        # Add to FAISS
        vec = self.embed(layout_description).reshape(1, -1)
        self.index.add(vec)
        self.entries.append(entry)
        self._save()

        logger.info(f"[VECTOR] Saved new format: {format_id} (score={quality_score:.2f})")
        return format_id

    def _build_context(self, extraction_result: dict) -> dict:
        """Build extraction context from a successful result for prompt injection."""
        items = extraction_result.get("LineItems", [])

        # Detect column order from populated fields
        column_order = []
        field_priority = [
            "PartNo", "ItemCode", "ItemDescription", "Quantity",
            "UnitOfQty", "UnitPrice", "RITC", "CountryOfOrigin",
        ]
        if items:
            for field in field_priority:
                non_null = sum(1 for it in items if it.get(field) is not None)
                if non_null > len(items) * 0.3:
                    column_order.append(field)

        # Detect tax structure
        tax_structure = "unknown"
        if items:
            has_ritc = any(it.get("RITC") for it in items)
            tax_structure = "HSN/RITC per item" if has_ritc else "no tax codes in items"

        # Sample output: first 3 items
        sample = items[:3] if items else []

        # Header info
        header = {}
        for k in ["InvoiceCurrency", "IncoTerms", "FreightTerms"]:
            v = extraction_result.get(k)
            if v:
                header[k] = v

        return {
            "column_order": column_order,
            "total_items": len(items),
            "tax_structure": tax_structure,
            "header_fields": header,
            "sample_output": sample,
        }

    # ── prompt builder ───────────────────────────────────────

    def build_augmented_prompt(self, base_prompt: str, match: dict) -> str:
        """Inject retrieved format context into the extraction prompt."""
        ctx = match["entry"].extraction_context
        if not ctx:
            return base_prompt

        parts = [base_prompt, "\n\nREFERENCE FORMAT (from a similar invoice extracted before):"]

        if ctx.get("column_order"):
            parts.append(f"- Column order: {' | '.join(ctx['column_order'])}")
        if ctx.get("tax_structure"):
            parts.append(f"- Tax structure: {ctx['tax_structure']}")
        if ctx.get("total_items"):
            parts.append(f"- Expected item count (approximate): {ctx['total_items']}")
        if ctx.get("header_fields"):
            parts.append(f"- Header fields: {json.dumps(ctx['header_fields'])}")
        if ctx.get("sample_output"):
            parts.append(f"- Sample items (first 3 from prior extraction):")
            parts.append(f"  {json.dumps(ctx['sample_output'], indent=2)}")

        parts.append(
            "\nUse this reference to guide field mapping and column alignment. "
            "If this invoice differs from the reference, trust what you see over the reference."
        )

        return "\n".join(parts)

    # ── stats ────────────────────────────────────────────────

    @property
    def total_formats(self) -> int:
        return len(self.entries)

    def get_stats(self) -> dict:
        return {
            "total_formats": self.total_formats,
            "index_size": self.index.ntotal if self.index else 0,
            "store_dir": str(self.store_dir),
            "formats": [
                {
                    "format_id": e.format_id,
                    "quality_score": e.quality_score,
                    "times_matched": e.times_matched,
                    "layout_preview": (e.layout_description or "")[:80],
                }
                for e in self.entries
            ],
        }
