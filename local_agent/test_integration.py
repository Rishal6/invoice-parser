"""
Integration tests for the invoice parser pipeline.

Tests all modules without hitting real AWS Bedrock.
Mocks LLM calls, tests real logic: registry, runner, verification,
feedback, metrics, patterns, knowledge, dedup, strip rendering.
"""
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ─── FIXTURES ─────────────────────────────────────────────

@pytest.fixture
def tmp_data_dir(tmp_path, monkeypatch):
    """Create temp data directories and redirect storage to use them."""
    (tmp_path / "templates").mkdir()
    (tmp_path / "kb").mkdir()
    (tmp_path / "verifications").mkdir()
    (tmp_path / "traces").mkdir()
    (tmp_path / "feedback").mkdir()
    import storage
    monkeypatch.setattr(storage, "LOCAL_DOC_ROOT", str(tmp_path))
    return tmp_path


@pytest.fixture
def sample_agent_result():
    """Minimal agent extraction result for testing."""
    return {
        "success": True,
        "filename": "test_invoice.pdf",
        "Classification": "Commercial Invoice",
        "InvoiceNo": "INV-2026-001",
        "Date": "2026-03-15",
        "InvoiceCurrency": "USD",
        "FreightTerms": "Prepaid",
        "IncoTerms": "FOB",
        "TermsOfPayment": "Net 30",
        "Exporter": {"Name": "ACME Corp", "Address": "123 Export St"},
        "Importer": {"Name": "Global Import LLC", "Address": "456 Import Ave"},
        "total_items": 3,
        "quality_score": 0.85,
        "LineItems": [
            {"PartNo": "A100", "ItemCode": "IC001", "ItemDescription": "Widget Alpha",
             "Quantity": "100", "UnitOfQty": "PCS", "UnitPrice": "10.50",
             "RITC": "84719000", "CountryOfOrigin": "China"},
            {"PartNo": "B200", "ItemCode": "IC002", "ItemDescription": "Gadget Beta",
             "Quantity": "50", "UnitOfQty": "EA", "UnitPrice": "25.00",
             "RITC": "85423100", "CountryOfOrigin": "Japan"},
            {"PartNo": "C300", "ItemCode": "IC003", "ItemDescription": "Sprocket Gamma",
             "Quantity": "200", "UnitOfQty": "KG", "UnitPrice": "5.75",
             "RITC": "73181500", "CountryOfOrigin": "India"},
        ],
    }


@pytest.fixture
def sample_worker_result(sample_agent_result):
    """Worker-corrected version: fixes PartNo on item 1, adds item 4."""
    result = json.loads(json.dumps(sample_agent_result))
    result["LineItems"][1]["PartNo"] = "B200-REV2"
    result["LineItems"][1]["UnitPrice"] = "27.50"
    result["LineItems"].append({
        "PartNo": "D400", "ItemCode": "IC004", "ItemDescription": "Bolt Delta",
        "Quantity": "500", "UnitOfQty": "PCS", "UnitPrice": "0.25",
        "RITC": "73181500", "CountryOfOrigin": "China",
    })
    return result


# ─── TEST: RUNNER (sandboxed exec) ───────────────────────

class TestRunner:
    def test_basic_extraction(self):
        from runner import run_extractor

        code = """
import re
def extract(raw_text):
    inv = re.search(r'Invoice[:\\s]+(\\S+)', raw_text)
    return {
        "InvoiceNo": inv.group(1) if inv else None,
        "LineItems": [{"PartNo": "TEST1", "Quantity": "10"}],
    }
"""
        result = run_extractor(code, "Invoice: INV-001\nSome items here")
        assert result["success"] is True
        assert result["InvoiceNo"] == "INV-001"
        assert len(result["LineItems"]) == 1

    def test_syntax_error(self):
        from runner import run_extractor
        result = run_extractor("def extract(raw_text)\n  return {}", "text")
        assert result["success"] is False
        assert "SyntaxError" in result["error"]

    def test_runtime_error(self):
        from runner import run_extractor
        code = """
def extract(raw_text):
    return 1 / 0
"""
        result = run_extractor(code, "text")
        assert result["success"] is False
        assert "ZeroDivisionError" in result["error"]

    def test_no_extract_function(self):
        from runner import run_extractor
        result = run_extractor("x = 1", "text")
        assert result["success"] is False
        assert "extract" in result["error"]

    def test_returns_non_dict(self):
        from runner import run_extractor
        code = """
def extract(raw_text):
    return [1, 2, 3]
"""
        result = run_extractor(code, "text")
        assert result["success"] is False

    def test_restricted_imports(self):
        from runner import run_extractor
        code = """
import os
def extract(raw_text):
    return {"files": os.listdir(".")}
"""
        result = run_extractor(code, "text")
        assert result["success"] is False


# ─── TEST: REGISTRY ──────────────────────────────────────

class TestRegistry:
    def test_save_and_lookup(self, tmp_data_dir):
        from registry import TemplateRegistry

        reg = TemplateRegistry(registry_dir=tmp_data_dir / "templates")

        code = 'def extract(raw_text):\n    return {"LineItems": []}\n'
        reg.save("test_template", code, {
            "column_headers": ["PartNo", "Qty"],
            "format_key": "us_commercial",
        })

        found = reg.lookup("test_template")
        assert found is not None
        assert "def extract" in found

    def test_lookup_miss(self, tmp_data_dir):
        from registry import TemplateRegistry

        reg = TemplateRegistry(registry_dir=tmp_data_dir / "templates")
        assert reg.lookup("nonexistent") is None

    def test_record_success(self, tmp_data_dir):
        from registry import TemplateRegistry

        reg = TemplateRegistry(registry_dir=tmp_data_dir / "templates")
        code = 'def extract(raw_text):\n    return {"LineItems": []}\n'
        reg.save("tmpl_a", code, {"format_key": "gst"})

        reg.record_success("tmpl_a")
        assert reg.entries["tmpl_a"]["success_count"] == 1

        reg.record_success("tmpl_a")
        assert reg.entries["tmpl_a"]["success_count"] == 2

    def test_list_templates(self, tmp_data_dir):
        from registry import TemplateRegistry

        reg = TemplateRegistry(registry_dir=tmp_data_dir / "templates")
        code = 'def extract(raw_text):\n    return {"LineItems": []}\n'
        reg.save("tmpl_x", code, {"format_key": "cfdi"})
        reg.save("tmpl_y", code, {"format_key": "eu_vat"})

        templates = reg.list_templates()
        assert len(templates) == 2
        ids = {t["template_id"] for t in templates}
        assert ids == {"tmpl_x", "tmpl_y"}


# ─── TEST: VERIFICATION ─────────────────────────────────

class TestVerification:
    def test_create_verification_routing(self, tmp_data_dir, sample_agent_result):
        import verification as v

        vm = v.VerificationManager()

        # quality_score 0.85 → quick_review
        result = vm.create_verification("job_001", sample_agent_result)
        assert result["review_level"] == "quick_review"
        assert result["status"] == "pending"

        # high score → auto_approve
        high = dict(sample_agent_result, quality_score=0.97)
        result2 = vm.create_verification("job_002", high)
        assert result2["review_level"] == "auto_approve"

        # low score → full_review
        low = dict(sample_agent_result, quality_score=0.5)
        result3 = vm.create_verification("job_003", low)
        assert result3["review_level"] == "full_review"

    def test_submit_and_diff(self, tmp_data_dir, sample_agent_result, sample_worker_result):
        import verification as v

        vm = v.VerificationManager()
        vm.create_verification("job_diff", sample_agent_result)

        submit = vm.submit_verification("job_diff", sample_worker_result, worker_id="worker_1")
        assert submit["status"] == "verified"
        assert submit["accuracy"]["corrected_fields"] > 0
        assert submit["accuracy"]["field_accuracy"] < 1.0

    def test_list_pending(self, tmp_data_dir, sample_agent_result):
        import verification as v

        vm = v.VerificationManager()
        vm.create_verification("job_p1", sample_agent_result)
        vm.create_verification("job_p2", sample_agent_result)

        pending = vm.list_pending()
        assert len(pending) == 2
        ids = {p["job_id"] for p in pending}
        assert ids == {"job_p1", "job_p2"}

    def test_accuracy_calculation(self, tmp_data_dir, sample_agent_result):
        import verification as v

        vm = v.VerificationManager()

        # Identical results → 100% accuracy
        vm.create_verification("job_perfect", sample_agent_result)
        submit = vm.submit_verification("job_perfect", sample_agent_result)
        assert submit["accuracy"]["field_accuracy"] == 1.0
        assert submit["accuracy"]["corrected_fields"] == 0


# ─── TEST: METRICS ───────────────────────────────────────

class TestMetrics:
    def test_record_and_query(self, tmp_data_dir):
        from metrics import MetricsTracker

        mt = MetricsTracker()

        accuracy = {
            "total_fields": 100,
            "correct_fields": 85,
            "corrected_fields": 15,
            "field_accuracy": 0.85,
            "per_field_accuracy": {"PartNo": 0.9, "RITC": 0.7},
        }

        mt.record_invoice(
            job_id="j1", filename="inv1.pdf",
            format_key="us_commercial", company="ACME",
            accuracy=accuracy, quality_score=0.85,
            review_level="quick_review", elapsed_seconds=30.0,
        )

        overall = mt.get_overall_accuracy(days=1)
        assert overall["total_invoices"] == 1
        assert overall["avg_field_accuracy"] == 0.85

    def test_format_accuracy(self, tmp_data_dir):
        from metrics import MetricsTracker

        mt = MetricsTracker()

        for i in range(3):
            mt.record_invoice(
                job_id=f"j{i}", filename=f"inv{i}.pdf",
                format_key="cfdi", company="MexCorp",
                accuracy={"total_fields": 50, "correct_fields": 45,
                          "corrected_fields": 5, "field_accuracy": 0.9,
                          "per_field_accuracy": {}},
                quality_score=0.9, review_level="quick_review",
                elapsed_seconds=20.0,
            )

        fmt = mt.get_format_accuracy("cfdi")
        assert "cfdi" in fmt
        assert fmt["cfdi"]["invoices"] == 3
        assert fmt["cfdi"]["avg_accuracy"] == 0.9

    def test_confidence_routing(self, tmp_data_dir):
        from metrics import MetricsTracker

        mt = MetricsTracker()

        # No history → relies on quality_score
        conf = mt.get_confidence_for_invoice("unknown", "unknown", 0.95)
        assert conf["review_level"] == "auto_approve"

        conf_low = mt.get_confidence_for_invoice("unknown", "unknown", 0.5)
        assert conf_low["review_level"] == "full_review"

    def test_improvement_trend(self, tmp_data_dir):
        from metrics import MetricsTracker

        mt = MetricsTracker()

        mt.record_invoice(
            job_id="t1", filename="t1.pdf",
            format_key="gst", company="IndCo",
            accuracy={"total_fields": 50, "correct_fields": 40,
                      "corrected_fields": 10, "field_accuracy": 0.8,
                      "per_field_accuracy": {}},
            quality_score=0.8, review_level="quick_review",
            elapsed_seconds=25.0,
        )

        trend = mt.get_improvement_trend(days=7)
        assert len(trend) >= 1
        assert "week" in trend[0]
        assert "avg_accuracy" in trend[0]


# ─── TEST: PATTERNS ──────────────────────────────────────

class TestPatterns:
    def test_add_and_confidence(self, tmp_data_dir):
        from patterns import PatternLibrary

        pl = PatternLibrary()

        pid = pl.add_pattern({
            "pattern_id": "test_missing_ritc",
            "description": "RITC column missed on CFDI invoices",
            "fix_prompt": "Look for HSN/RITC column explicitly",
            "affected_fields": ["RITC"],
        })

        assert pid == "test_missing_ritc"
        assert len(pl.patterns) == 1
        assert pl.patterns[0]["confidence"] == 0.2

    def test_update_confidence(self, tmp_data_dir):
        from patterns import PatternLibrary

        pl = PatternLibrary()
        pl.add_pattern({
            "pattern_id": "p1",
            "description": "test",
            "fix_prompt": "fix it",
            "affected_fields": ["PartNo"],
        })

        # add_pattern sets times_used=1, so after 5 updates = 6 total
        for _ in range(5):
            pl.update_confidence("p1", worked=True)

        p = next(p for p in pl.patterns if p["pattern_id"] == "p1")
        assert p["times_used"] == 6
        assert p["success_rate"] == round(5 / 6, 3)
        assert p["confidence"] == p["success_rate"]  # times_used >= 5

    def test_promotable_patterns(self, tmp_data_dir):
        from patterns import PatternLibrary

        pl = PatternLibrary()
        pl.add_pattern({
            "pattern_id": "promo_test",
            "description": "test promotion",
            "fix_prompt": "fix",
            "affected_fields": ["RITC"],
            "company": None,
        })

        # Not promotable yet (not enough uses)
        assert len(pl.get_promotable_patterns()) == 0

        # add_pattern starts with times_used=1. Need 4 more updates for times_used=5
        # and all must succeed for success_rate >= 0.9
        # With times_used=1 (initial, times_worked=0) + 5 updates (all worked):
        # times_used=6, times_worked=5, success_rate=5/6=0.833 < 0.9
        # Need all 6 to work: set times_worked=1 on initial or update 9 times
        for _ in range(9):
            pl.update_confidence("promo_test", worked=True)

        # times_used=10, times_worked=9, success_rate=0.9
        promotable = pl.get_promotable_patterns()
        assert len(promotable) == 1
        assert promotable[0]["pattern_id"] == "promo_test"

    def test_company_specific_not_promotable(self, tmp_data_dir):
        from patterns import PatternLibrary

        pl = PatternLibrary()
        pl.add_pattern({
            "pattern_id": "company_specific",
            "description": "ACME-only issue",
            "fix_prompt": "fix",
            "affected_fields": ["PartNo"],
            "company": "ACME Corp",
        })

        for _ in range(6):
            pl.update_confidence("company_specific", worked=True)

        # Company-specific patterns should not be promotable
        assert len(pl.get_promotable_patterns()) == 0


# ─── TEST: KNOWLEDGE BASE ────────────────────────────────

class TestKnowledge:
    def test_load_kb(self):
        from knowledge import KnowledgeBase, KB_DIR

        if not KB_DIR.exists():
            pytest.skip("KB directory not found")

        kb = KnowledgeBase()
        assert len(kb.file_sections) > 0

    def test_get_context_unknown(self):
        from knowledge import KnowledgeBase

        kb = KnowledgeBase()
        ctx = kb.get_context("unknown")
        assert ctx == ""

    def test_format_names(self):
        from knowledge import FORMAT_NAMES
        assert "cfdi" in FORMAT_NAMES
        assert "fapiao" in FORMAT_NAMES
        assert "eu_vat" in FORMAT_NAMES


# ─── TEST: FEEDBACK ──────────────────────────────────────

class TestFeedback:
    def test_diff_results(self, sample_agent_result, sample_worker_result):
        from feedback import FeedbackProcessor

        fp = FeedbackProcessor()
        corrections = fp.diff_results(sample_agent_result, sample_worker_result)

        assert len(corrections) > 0
        # Worker changed PartNo B200 → B200-REV2, UnitPrice 25.00 → 27.50
        part_corrections = [c for c in corrections if c["field"] == "PartNo" and c.get("item_index") == 1]
        assert len(part_corrections) == 1
        assert part_corrections[0]["worker_value"] == "B200-REV2"

    def test_diff_identical(self, sample_agent_result):
        from feedback import FeedbackProcessor

        fp = FeedbackProcessor()
        corrections = fp.diff_results(sample_agent_result, sample_agent_result)
        assert len(corrections) == 0

    def test_accuracy_for_job(self, sample_agent_result, sample_worker_result):
        from feedback import FeedbackProcessor

        fp = FeedbackProcessor()
        accuracy = fp.get_accuracy_for_job(sample_agent_result, sample_worker_result)
        assert accuracy["field_accuracy"] < 1.0
        assert accuracy["corrected_fields"] > 0


# ─── TEST: DEDUP HELPERS ─────────────────────────────────

class TestDedup:
    def test_exact_dedup(self):
        from tools import _normalize_str, _normalize_num, _is_fuzzy_dupe

        a = {"PartNo": "A100", "ItemDescription": "Widget", "Quantity": "100", "UnitPrice": "10.50"}
        b = {"PartNo": "A100", "ItemDescription": "Widget", "Quantity": "100", "UnitPrice": "10.50"}
        assert _is_fuzzy_dupe(a, b) is True

    def test_different_price_not_dupe(self):
        from tools import _is_fuzzy_dupe

        a = {"PartNo": "A100", "ItemDescription": "Widget", "Quantity": "100", "UnitPrice": "10.50"}
        b = {"PartNo": "A100", "ItemDescription": "Widget", "Quantity": "100", "UnitPrice": "15.00"}
        # Same part, different price — NOT a fuzzy dupe (price must match)
        assert _is_fuzzy_dupe(a, b) is False

    def test_different_part_not_dupe(self):
        from tools import _is_fuzzy_dupe

        a = {"PartNo": "A100", "ItemDescription": "Widget Alpha", "Quantity": "100", "UnitPrice": "10.50"}
        b = {"PartNo": "Z999", "ItemDescription": "Totally Different", "Quantity": "100", "UnitPrice": "10.50"}
        assert _is_fuzzy_dupe(a, b) is False

    def test_normalize_helpers(self):
        from tools import _normalize_str, _normalize_num, _to_float

        assert _normalize_str("  Test-123! ") == "test123"
        assert _normalize_str(None) == ""
        assert _to_float("1,234.50") == 1234.50
        assert _to_float(None) is None
        assert _normalize_num("1234.5") == "1234.5000"


# ─── TEST: CODEGEN (mocked LLM) ─────────────────────────

class TestCodegen:
    @patch("codegen.bedrock")
    def test_generate_extractor_structured(self, mock_bedrock):
        from codegen import generate_extractor

        mock_bedrock.converse.return_value = {
            "output": {"message": {"content": [{"text": json.dumps({
                "template_id": "acme_8col",
                "structured": True,
                "code": "import re\ndef extract(raw_text):\n    return {'LineItems': []}",
                "column_headers": ["PartNo", "Description", "Qty"],
            })}]}},
        }

        result = generate_extractor(
            sample_text="Invoice No: INV-001\nPartNo Description Qty\nA100 Widget 10",
            column_headers=["PartNo", "Description", "Qty"],
            format_key="us_commercial",
        )

        assert result["structured"] is True
        assert result["code"] is not None
        assert "us_commercial" in result["template_id"]

    @patch("codegen.bedrock")
    def test_generate_extractor_unstructured(self, mock_bedrock):
        from codegen import generate_extractor

        mock_bedrock.converse.return_value = {
            "output": {"message": {"content": [{"text": json.dumps({
                "template_id": None,
                "structured": False,
                "code": None,
                "column_headers": [],
            })}]}},
        }

        result = generate_extractor(
            sample_text="messy handwritten text...",
            column_headers=[],
            format_key="unknown",
        )

        assert result["structured"] is False
        assert result["code"] is None

    @patch("codegen.bedrock")
    def test_generate_extractor_llm_failure(self, mock_bedrock):
        from codegen import generate_extractor

        mock_bedrock.converse.side_effect = Exception("Throttled")

        result = generate_extractor("text", [], "unknown")
        assert result["structured"] is False


# ─── TEST: STRIP RENDERING ───────────────────────────────

class TestStripRendering:
    def test_render_page_strips(self):
        """Test strip rendering with a real tiny PDF."""
        import fitz as pymupdf

        # Create a minimal PDF
        doc = pymupdf.open()
        page = doc.new_page(width=595, height=842)  # A4
        page.insert_text((50, 100), "Line item 1: Widget A100 qty 10")
        page.insert_text((50, 300), "Line item 2: Gadget B200 qty 20")
        page.insert_text((50, 500), "Line item 3: Sprocket C300 qty 30")
        page.insert_text((50, 700), "Line item 4: Bolt D400 qty 40")

        tmp_pdf = tempfile.mktemp(suffix=".pdf")
        doc.save(tmp_pdf)
        doc.close()

        try:
            from tools import _render_page_strips
            strips = _render_page_strips(tmp_pdf, 0, strips_per_page=4, overlap_pct=0.2)

            assert len(strips) == 4
            for s in strips:
                assert "image" in s
                assert s["image"]["format"] == "png"
                assert len(s["image"]["source"]["bytes"]) > 0
        finally:
            os.unlink(tmp_pdf)

    def test_render_pdf_to_images(self):
        """Test full page rendering."""
        import fitz as pymupdf

        doc = pymupdf.open()
        doc.new_page(width=595, height=842)
        tmp_pdf = tempfile.mktemp(suffix=".pdf")
        doc.save(tmp_pdf)
        doc.close()

        try:
            from tools import _render_pdf_to_images
            images = _render_pdf_to_images(tmp_pdf)
            assert len(images) == 1
            assert images[0]["image"]["format"] == "png"
        finally:
            os.unlink(tmp_pdf)


# ─── TEST: PROCESS ALL CHUNKS (mocked) ──────────────────

class TestParallelProcessing:
    @patch("tools._call_bedrock")
    def test_process_all_chunks_basic(self, mock_bedrock, tmp_path):
        """Test parallel processing with mocked Bedrock calls."""
        import fitz as pymupdf
        from tools import process_all_chunks, split_pdf, reset_accumulator, accumulator as acc
        import tools

        reset_accumulator()

        # Create a 3-page PDF
        doc = pymupdf.open()
        for i in range(3):
            page = doc.new_page(width=595, height=842)
            page.insert_text((50, 100), f"Invoice No: INV-001\nPage {i+1}")
            page.insert_text((50, 200), f"PartNo: ITEM{i+1}\nQuantity: {(i+1)*10}\nUnitPrice: {(i+1)*5.0}")
        pdf_path = str(tmp_path / "test.pdf")
        doc.save(pdf_path)
        doc.close()

        # Mock Bedrock to return extraction + review results
        extract_response = {
            "text": json.dumps({
                "Classification": "Invoice",
                "InvoiceNo": "INV-001",
                "LineItems": [{"PartNo": "ITEM1", "Quantity": "10", "UnitPrice": "5.0",
                               "ItemDescription": "Test item", "UnitOfQty": "PCS",
                               "RITC": None, "CountryOfOrigin": None, "ItemCode": None}],
            }),
            "input_tokens": 100,
            "output_tokens": 200,
            "truncated": False,
        }
        review_response = {
            "text": json.dumps({
                "passed": True,
                "score": 0.95,
                "expectedItemCount": 1,
                "extractedItemCount": 1,
                "issues": [],
                "summary": "All good",
            }),
            "input_tokens": 100,
            "output_tokens": 100,
            "truncated": False,
        }

        # Mock format detection
        with patch("tools.kb") as mock_kb:
            mock_kb.detect_and_retrieve.return_value = {
                "format": "us_commercial",
                "confidence": 0.9,
                "context": "",
                "format_name": "US Commercial Invoice",
            }

            # Mock template registry
            with patch("tools.template_registry") as mock_reg:
                mock_reg.lookup.return_value = None

                # Alternate between extract and review responses
                mock_bedrock.side_effect = [
                    extract_response, review_response,  # chunk 0
                    extract_response, review_response,  # chunk 1
                    extract_response, review_response,  # chunk 2
                ]

                split_pdf(pdf_path, pages_per_chunk=1, overlap_pages=0)
                assert tools.accumulator.total_chunks == 3

                result = process_all_chunks()
                assert "3 chunks processed" in result
                assert tools.accumulator.processed_indices == {0, 1, 2}


# ─── TEST: END-TO-END FLOW (all modules) ────────────────

class TestEndToEnd:
    def test_full_verification_flow(self, tmp_data_dir, sample_agent_result, sample_worker_result):
        """
        End-to-end: extraction result → verification → corrections →
        feedback → pattern created → metrics recorded.
        """
        import verification as v
        from feedback import FeedbackProcessor
        from patterns import PatternLibrary
        from metrics import MetricsTracker

        pl = PatternLibrary()
        fp = FeedbackProcessor(pattern_library=pl)
        mt = MetricsTracker()
        vm = v.VerificationManager()

        # 1. Create verification
        job_id = "e2e_test_001"
        verif = vm.create_verification(job_id, sample_agent_result, format_key="us_commercial")
        assert verif["review_level"] == "quick_review"

        # 2. Worker submits corrections
        submit = vm.submit_verification(job_id, sample_worker_result, worker_id="tester")
        assert submit["status"] == "verified"
        corrections = submit["corrections"]
        accuracy = submit["accuracy"]
        assert accuracy["corrected_fields"] > 0

        # 3. Process feedback (mock LLM for pattern generation)
        with patch.object(fp, "_generate_pattern_from_corrections") as mock_gen:
            mock_gen.return_value = {
                "pattern_id": "e2e_test_pattern",
                "description": "B200 part number revision",
                "fix_prompt": "Check for revised part numbers",
                "affected_fields": ["PartNo"],
                "format": "us_commercial",
            }

            fb_result = fp.process_feedback(
                job_id=job_id,
                corrections=corrections,
                source_text="Invoice text here...",
                invoice_header=sample_agent_result,
                format_key="us_commercial",
            )

            assert fb_result["patterns_created"] == 1
            assert len(pl.patterns) == 1

        # 4. Record metrics
        mt.record_invoice(
            job_id=job_id,
            filename="test.pdf",
            format_key="us_commercial",
            company="ACME Corp",
            accuracy=accuracy,
            quality_score=0.85,
            review_level="quick_review",
            elapsed_seconds=30.0,
        )

        overall = mt.get_overall_accuracy(days=1)
        assert overall["total_invoices"] == 1

        fmt = mt.get_format_accuracy("us_commercial")
        assert fmt["us_commercial"]["invoices"] == 1

    def test_template_registry_round_trip(self, tmp_data_dir):
        """Create template → save → lookup → run → verify output."""
        from registry import TemplateRegistry
        from runner import run_extractor

        reg = TemplateRegistry(registry_dir=tmp_data_dir / "templates")

        extractor_code = '''
import re

def extract(raw_text):
    inv_match = re.search(r'Invoice No[.:]?\\s*(\\S+)', raw_text)
    date_match = re.search(r'Date[:]?\\s*(\\d{4}-\\d{2}-\\d{2})', raw_text)

    items = []
    for m in re.finditer(r'(\\w+)\\s+(\\d+)\\s+PCS\\s+([\\d.]+)', raw_text):
        items.append({
            "PartNo": m.group(1),
            "Quantity": m.group(2),
            "UnitOfQty": "PCS",
            "UnitPrice": m.group(3),
            "ItemDescription": None,
            "ItemCode": None,
            "RITC": None,
            "CountryOfOrigin": None,
        })

    return {
        "InvoiceNo": inv_match.group(1) if inv_match else None,
        "Date": date_match.group(1) if date_match else None,
        "LineItems": items,
    }
'''

        reg.save("test_round_trip", extractor_code, {
            "column_headers": ["PartNo", "Qty", "UOM", "Price"],
            "format_key": "us_commercial",
        })

        # Lookup
        code = reg.lookup("test_round_trip")
        assert code is not None

        # Run
        test_text = (
            "Invoice No. INV-2026-100\n"
            "Date: 2026-04-15\n"
            "WIDGET1 50 PCS 12.50\n"
            "GADGET2 100 PCS 8.75\n"
            "BOLT3 200 PCS 1.25\n"
        )

        result = run_extractor(code, test_text)
        assert result["success"] is True
        assert result["InvoiceNo"] == "INV-2026-100"
        assert result["Date"] == "2026-04-15"
        assert len(result["LineItems"]) == 3
        assert result["LineItems"][0]["PartNo"] == "WIDGET1"
        assert result["LineItems"][2]["Quantity"] == "200"

    def test_pattern_lifecycle(self, tmp_data_dir):
        """Pattern: create → use → succeed → promote."""
        from patterns import PatternLibrary

        pl = PatternLibrary()

        # Create (starts with times_used=1, times_worked=0)
        pl.add_pattern({
            "pattern_id": "lifecycle_test",
            "description": "RITC missed in CFDI",
            "fix_prompt": "Look for HSN column",
            "affected_fields": ["RITC"],
            "company": None,
            "format": "cfdi",
        })

        # Need 9 successful updates: times_used=10, times_worked=9, rate=0.9
        for _ in range(9):
            pl.update_confidence("lifecycle_test", worked=True)

        p = next(p for p in pl.patterns if p["pattern_id"] == "lifecycle_test")
        assert p["times_used"] == 10
        assert p["success_rate"] == 0.9
        assert p["confidence"] == 0.9

        # Should be promotable (times_used>=5, success_rate>=0.9, company=None)
        promotable = pl.get_promotable_patterns()
        assert len(promotable) == 1

        # Promote
        kb_dir = tmp_data_dir / "kb"
        kb_dir.mkdir(exist_ok=True)

        pl.promote_to_kb("lifecycle_test", "test_kb.md")

        p = next(p for p in pl.patterns if p["pattern_id"] == "lifecycle_test")
        assert p.get("promoted_to_kb") is True


# ─── TEST: TRACER ────────────────────────────────────────

class TestTracer:
    def test_step_and_trace(self, tmp_data_dir):
        from tracer import JobTracer
        import storage

        t = JobTracer("test123", "invoice.pdf")
        t.step('split_pdf', result="5 chunks from 50 pages")
        t.step('extract_chunk', chunk=0, result="140 items", tokens_in=15000, tokens_out=8000, duration=12.1)
        t.step('review_chunk', chunk=0, result="PASSED score=0.82", tokens_in=12000, tokens_out=2000)
        t.step('deduplicate', result="305 -> 271 (exact=20, fuzzy=14)")
        t.step('verify_final', result="PASS — 271 items")
        t.finish(total_items=271, quality_score=0.82, passed=True)

        trace = t.get_trace()
        assert trace['job_id'] == 'test123'
        assert len(trace['steps']) == 5
        assert trace['total_tokens']['in'] == 27000
        assert trace['total_tokens']['out'] == 10000
        assert trace['estimated_cost'] > 0

        log = t.get_tool_log()
        assert len(log) == 5
        assert log[0]['tool'] == 'split_pdf'
        assert log[1]['chunk_index'] == 0

        saved = storage.get_doc('traces', 'test123')
        assert saved is not None
        assert saved['job_id'] == 'test123'

    def test_empty_tracer(self, tmp_data_dir):
        from tracer import JobTracer

        t = JobTracer("empty_job", "")
        trace = t.get_trace()
        assert trace['steps'] == []
        assert trace['total_tokens']['total'] == 0


# ─── TEST: API ENDPOINTS ─────────────────────────────────

class TestAPI:
    @pytest.fixture
    def client(self):
        """Create FastAPI test client."""
        from fastapi.testclient import TestClient
        from api import app
        return TestClient(app)

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_verify_dashboard(self, client):
        resp = client.get("/verify/ui/dashboard")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_verify_pending_list(self, client):
        resp = client.get("/verify/pending/list")
        assert resp.status_code == 200

    def test_patterns_endpoint(self, client):
        resp = client.get("/patterns")
        assert resp.status_code == 200

    def test_metrics_endpoint(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
