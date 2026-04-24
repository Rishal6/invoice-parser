"""
Metrics -- Track extraction accuracy over time.

Records accuracy per invoice, aggregates by week/format/company.
Used for confidence routing (auto-approve vs full review).
"""
import time
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta

import storage

logger = logging.getLogger(__name__)

MAX_INVOICE_RECORDS = 500


class MetricsTracker:
    """Tracks extraction accuracy over time."""

    def __init__(self):
        self.data = self._load()

    def _load(self) -> Dict:
        """Load metrics from storage."""
        doc = storage.get_doc('metrics', 'aggregate')
        if doc:
            return doc
        return {"invoices": [], "weekly": [], "by_format": {}, "by_company": {}}

    def _save(self):
        """Save metrics to storage."""
        storage.put_doc('metrics', 'aggregate', self.data)

    # ------------------------------------------------------------------
    # Aggregate helpers
    # ------------------------------------------------------------------

    def _update_aggregate(self, bucket: Dict, accuracy: Dict):
        """
        Update a running aggregate bucket (format or company) with new accuracy data.

        bucket schema:
            total_invoices: int
            total_correct: int
            total_fields: int
            field_errors: {field_name: error_count, ...}
        """
        bucket["total_invoices"] = bucket.get("total_invoices", 0) + 1
        bucket["total_correct"] = bucket.get("total_correct", 0) + accuracy.get("correct_fields", 0)
        bucket["total_fields"] = bucket.get("total_fields", 0) + accuracy.get("total_fields", 0)

        field_errors = bucket.get("field_errors", {})
        per_field = accuracy.get("per_field_accuracy", {})
        for field_name, correct in per_field.items():
            if not correct:
                field_errors[field_name] = field_errors.get(field_name, 0) + 1
        bucket["field_errors"] = field_errors

    def _week_key(self, ts: float) -> str:
        """Return ISO week string like '2026-W17' for a Unix timestamp."""
        dt = datetime.fromtimestamp(ts)
        iso = dt.isocalendar()
        return f"{iso[0]}-W{iso[1]:02d}"

    # ------------------------------------------------------------------
    # Record
    # ------------------------------------------------------------------

    def record_invoice(self, job_id: str, filename: str,
                       format_key: str, company: str,
                       accuracy: Dict, quality_score: float,
                       review_level: str, elapsed_seconds: float):
        """
        Record accuracy for one invoice.

        Args:
            job_id: extraction job ID
            filename: PDF filename
            format_key: detected format (cfdi, fapiao, eu_vat, etc.)
            company: exporter company name
            accuracy: from verification._calculate_accuracy()
                      {total_fields, correct_fields, corrected_fields,
                       field_accuracy, per_field_accuracy}
            quality_score: agent's quality score
            review_level: "auto_approve", "quick_review", "full_review"
            elapsed_seconds: extraction time

        Appends to invoices list, updates by_format and by_company aggregates.
        Keep only last 500 invoice records to prevent file bloat.
        """
        ts = time.time()
        record = {
            "job_id": job_id,
            "filename": filename,
            "format_key": format_key,
            "company": company,
            "field_accuracy": accuracy.get("field_accuracy", 0.0),
            "total_fields": accuracy.get("total_fields", 0),
            "correct_fields": accuracy.get("correct_fields", 0),
            "corrected_fields": accuracy.get("corrected_fields", 0),
            "per_field_accuracy": accuracy.get("per_field_accuracy", {}),
            "quality_score": quality_score,
            "review_level": review_level,
            "elapsed_seconds": elapsed_seconds,
            "timestamp": ts,
        }

        # Append and trim to cap
        self.data["invoices"].append(record)
        if len(self.data["invoices"]) > MAX_INVOICE_RECORDS:
            self.data["invoices"] = self.data["invoices"][-MAX_INVOICE_RECORDS:]

        # Update format aggregate
        fmt_key = format_key or "unknown"
        if fmt_key not in self.data["by_format"]:
            self.data["by_format"][fmt_key] = {
                "total_invoices": 0, "total_correct": 0,
                "total_fields": 0, "field_errors": {}
            }
        self._update_aggregate(self.data["by_format"][fmt_key], accuracy)

        # Update company aggregate
        comp_key = (company or "unknown").strip()
        if comp_key not in self.data["by_company"]:
            self.data["by_company"][comp_key] = {
                "total_invoices": 0, "total_correct": 0,
                "total_fields": 0, "field_errors": {}
            }
        self._update_aggregate(self.data["by_company"][comp_key], accuracy)

        self._save()
        logger.info(
            "Recorded metrics for %s — accuracy %.1f%%, quality %.2f, review=%s",
            filename, accuracy.get("field_accuracy", 0) * 100, quality_score, review_level
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def _invoices_in_window(self, days: int) -> List[Dict]:
        """Return invoice records from the last N days."""
        cutoff = time.time() - (days * 86400)
        return [inv for inv in self.data["invoices"] if inv.get("timestamp", 0) >= cutoff]

    def get_overall_accuracy(self, days: int = 30) -> Dict:
        """
        Get overall accuracy for the last N days.

        Returns: {
            "period": "last 30 days",
            "total_invoices": 150,
            "avg_field_accuracy": 0.89,
            "avg_quality_score": 0.82,
            "by_review_level": {
                "auto_approve": 45,
                "quick_review": 80,
                "full_review": 25
            },
            "trend": "improving"  # or "stable", "declining"
        }
        """
        invoices = self._invoices_in_window(days)
        if not invoices:
            return {
                "period": f"last {days} days",
                "total_invoices": 0,
                "avg_field_accuracy": 0.0,
                "avg_quality_score": 0.0,
                "by_review_level": {},
                "trend": "stable",
            }

        total = len(invoices)
        avg_acc = sum(inv.get("field_accuracy", 0) for inv in invoices) / total
        avg_qs = sum(inv.get("quality_score", 0) for inv in invoices) / total

        by_review: Dict[str, int] = {}
        for inv in invoices:
            lvl = inv.get("review_level", "unknown")
            by_review[lvl] = by_review.get(lvl, 0) + 1

        trend = self._detect_trend(invoices)

        return {
            "period": f"last {days} days",
            "total_invoices": total,
            "avg_field_accuracy": round(avg_acc, 4),
            "avg_quality_score": round(avg_qs, 4),
            "by_review_level": by_review,
            "trend": trend,
        }

    def get_format_accuracy(self, format_key: str = None) -> Dict:
        """
        Get accuracy breakdown by format.

        If format_key given, return details for that format.
        Otherwise return summary for all formats.

        Returns: {
            "cfdi": {"invoices": 30, "avg_accuracy": 0.75, "common_errors": ["PartNo", "RITC"]},
            "eu_vat": {"invoices": 50, "avg_accuracy": 0.92, "common_errors": []},
            ...
        }
        """
        by_fmt = self.data.get("by_format", {})

        def _summarize(key: str, bucket: Dict) -> Dict:
            total_inv = bucket.get("total_invoices", 0)
            total_fields = bucket.get("total_fields", 0)
            total_correct = bucket.get("total_correct", 0)
            avg_acc = total_correct / total_fields if total_fields > 0 else 0.0
            # Top error fields sorted by count descending
            field_errors = bucket.get("field_errors", {})
            common = sorted(field_errors.keys(), key=lambda f: field_errors[f], reverse=True)[:5]
            return {
                "invoices": total_inv,
                "avg_accuracy": round(avg_acc, 4),
                "common_errors": common,
            }

        if format_key:
            if format_key not in by_fmt:
                return {format_key: {"invoices": 0, "avg_accuracy": 0.0, "common_errors": []}}
            return {format_key: _summarize(format_key, by_fmt[format_key])}

        result = {}
        for key, bucket in by_fmt.items():
            result[key] = _summarize(key, bucket)
        return result

    def get_company_accuracy(self, company: str = None) -> Dict:
        """
        Get accuracy breakdown by company.
        Similar to get_format_accuracy but grouped by exporter company.
        """
        by_comp = self.data.get("by_company", {})

        def _summarize(bucket: Dict) -> Dict:
            total_inv = bucket.get("total_invoices", 0)
            total_fields = bucket.get("total_fields", 0)
            total_correct = bucket.get("total_correct", 0)
            avg_acc = total_correct / total_fields if total_fields > 0 else 0.0
            field_errors = bucket.get("field_errors", {})
            common = sorted(field_errors.keys(), key=lambda f: field_errors[f], reverse=True)[:5]
            return {
                "invoices": total_inv,
                "avg_accuracy": round(avg_acc, 4),
                "common_errors": common,
            }

        if company:
            comp_key = company.strip()
            if comp_key not in by_comp:
                return {comp_key: {"invoices": 0, "avg_accuracy": 0.0, "common_errors": []}}
            return {comp_key: _summarize(by_comp[comp_key])}

        result = {}
        for key, bucket in by_comp.items():
            result[key] = _summarize(bucket)
        return result

    def get_field_accuracy(self, days: int = 30) -> Dict:
        """
        Get per-field accuracy across all invoices in the window.

        Returns: {
            "PartNo": 0.72,
            "ItemDescription": 0.85,
            ...
        }

        Helps identify which fields are weakest across all invoices.
        """
        invoices = self._invoices_in_window(days)
        if not invoices:
            return {}

        # Accumulate per-field: {field: [correct_count, total_count]}
        field_stats: Dict[str, List[int]] = {}
        for inv in invoices:
            per_field = inv.get("per_field_accuracy", {})
            for field_name, correct in per_field.items():
                if field_name not in field_stats:
                    field_stats[field_name] = [0, 0]
                field_stats[field_name][1] += 1
                if correct:
                    field_stats[field_name][0] += 1

        result = {}
        for field_name, (correct_count, total_count) in field_stats.items():
            result[field_name] = round(correct_count / total_count, 4) if total_count > 0 else 0.0

        # Sort weakest first
        return dict(sorted(result.items(), key=lambda kv: kv[1]))

    def get_confidence_for_invoice(self, format_key: str, company: str,
                                   quality_score: float) -> Dict:
        """
        Determine review level for a new invoice based on historical accuracy.

        Logic:
        1. Check format accuracy history
        2. Check company accuracy history
        3. Factor in current quality_score

        confidence = weighted average:
          - 40% format historical accuracy (if >= 10 invoices)
          - 30% company historical accuracy (if >= 5 invoices)
          - 30% current quality_score

        If not enough history, default to quality_score only.

        Returns: {
            "confidence": 0.85,
            "review_level": "quick_review",
            "reason": "Format cfdi: 75% accuracy over 30 invoices, ..."
        }
        """
        fmt_bucket = self.data.get("by_format", {}).get(format_key or "unknown", {})
        comp_bucket = self.data.get("by_company", {}).get((company or "unknown").strip(), {})

        fmt_inv = fmt_bucket.get("total_invoices", 0)
        comp_inv = comp_bucket.get("total_invoices", 0)

        fmt_total_fields = fmt_bucket.get("total_fields", 0)
        fmt_total_correct = fmt_bucket.get("total_correct", 0)
        fmt_acc = fmt_total_correct / fmt_total_fields if fmt_total_fields > 0 else 0.0

        comp_total_fields = comp_bucket.get("total_fields", 0)
        comp_total_correct = comp_bucket.get("total_correct", 0)
        comp_acc = comp_total_correct / comp_total_fields if comp_total_fields > 0 else 0.0

        has_format_history = fmt_inv >= 10
        has_company_history = comp_inv >= 5

        reasons = []

        if has_format_history and has_company_history:
            # Full weighted average: 40% format, 30% company, 30% quality
            confidence = 0.4 * fmt_acc + 0.3 * comp_acc + 0.3 * quality_score
            reasons.append(f"Format {format_key}: {fmt_acc:.0%} accuracy over {fmt_inv} invoices")
            reasons.append(f"Company {company}: {comp_acc:.0%} accuracy over {comp_inv} invoices")
            reasons.append(f"Quality score: {quality_score:.2f}")
        elif has_format_history:
            # 55% format, 45% quality
            confidence = 0.55 * fmt_acc + 0.45 * quality_score
            reasons.append(f"Format {format_key}: {fmt_acc:.0%} accuracy over {fmt_inv} invoices")
            reasons.append(f"Quality score: {quality_score:.2f}")
            reasons.append(f"Company {company}: insufficient history ({comp_inv} invoices)")
        elif has_company_history:
            # 40% company, 60% quality
            confidence = 0.4 * comp_acc + 0.6 * quality_score
            reasons.append(f"Company {company}: {comp_acc:.0%} accuracy over {comp_inv} invoices")
            reasons.append(f"Quality score: {quality_score:.2f}")
            reasons.append(f"Format {format_key}: insufficient history ({fmt_inv} invoices)")
        else:
            # No meaningful history — rely on quality score
            confidence = quality_score
            reasons.append(f"Quality score: {quality_score:.2f}")
            reasons.append("No significant format or company history yet")

        confidence = round(max(0.0, min(1.0, confidence)), 4)

        if confidence >= 0.95:
            review_level = "auto_approve"
        elif confidence >= 0.7:
            review_level = "quick_review"
        else:
            review_level = "full_review"

        return {
            "confidence": confidence,
            "review_level": review_level,
            "reason": "; ".join(reasons),
        }

    # ------------------------------------------------------------------
    # Trends
    # ------------------------------------------------------------------

    def _detect_trend(self, invoices: List[Dict]) -> str:
        """
        Compare average accuracy of the last 2 weeks.
        improving if delta > 0.02, declining if < -0.02, stable otherwise.
        """
        if len(invoices) < 2:
            return "stable"

        now = time.time()
        one_week_ago = now - 7 * 86400
        two_weeks_ago = now - 14 * 86400

        this_week = [inv for inv in invoices if inv.get("timestamp", 0) >= one_week_ago]
        last_week = [inv for inv in invoices
                     if two_weeks_ago <= inv.get("timestamp", 0) < one_week_ago]

        if not this_week or not last_week:
            return "stable"

        avg_this = sum(inv.get("field_accuracy", 0) for inv in this_week) / len(this_week)
        avg_last = sum(inv.get("field_accuracy", 0) for inv in last_week) / len(last_week)
        delta = avg_this - avg_last

        if delta > 0.02:
            return "improving"
        elif delta < -0.02:
            return "declining"
        return "stable"

    def get_improvement_trend(self, days: int = 60) -> List[Dict]:
        """
        Get weekly accuracy trend.

        Returns list of weekly data points:
        [
            {"week": "2026-W15", "invoices": 25, "avg_accuracy": 0.78},
            {"week": "2026-W16", "invoices": 30, "avg_accuracy": 0.82},
            ...
        ]
        """
        invoices = self._invoices_in_window(days)
        if not invoices:
            return []

        # Group by ISO week
        weeks: Dict[str, List[Dict]] = {}
        for inv in invoices:
            wk = self._week_key(inv.get("timestamp", 0))
            if wk not in weeks:
                weeks[wk] = []
            weeks[wk].append(inv)

        result = []
        for wk in sorted(weeks.keys()):
            group = weeks[wk]
            avg_acc = sum(inv.get("field_accuracy", 0) for inv in group) / len(group)
            result.append({
                "week": wk,
                "invoices": len(group),
                "avg_accuracy": round(avg_acc, 4),
            })

        return result
