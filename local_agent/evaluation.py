"""
Evaluation Engine — deterministic quality checks at every pipeline step.

Runs BEFORE LLM review (cheap, instant, catches obvious errors)
and AFTER verify_final (business rule validation).

No LLM calls — pure Python validation.
"""
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

VALID_CURRENCIES = {
    'USD', 'EUR', 'GBP', 'INR', 'JPY', 'CNY', 'CAD', 'AUD', 'CHF',
    'SGD', 'HKD', 'MXN', 'BRL', 'KRW', 'THB', 'MYR', 'TWD', 'VND',
    'IDR', 'PHP', 'AED', 'SAR', 'ZAR', 'SEK', 'NOK', 'DKK', 'PLN',
    'CZK', 'HUF', 'TRY', 'RUB', 'NZD', 'ILS', 'CLP', 'ARS', 'COP',
    'PEN', 'EGP', 'PKR', 'BDT', 'LKR', 'QAR', 'KWD', 'BHD', 'OMR',
}

VALID_INCOTERMS = {'FOB', 'CIF', 'CI', 'CF'}

VALID_UOMS = {
    'PCS', 'EA', 'KG', 'G', 'LB', 'OZ', 'L', 'ML', 'GAL',
    'M', 'CM', 'MM', 'FT', 'IN', 'YD', 'SET', 'PKG', 'BOX',
    'ROLL', 'SHT', 'PR', 'DOZ', 'CTN', 'PAL', 'MT', 'NOS',
    'PAIR', 'LOT', 'BAG', 'UNIT', 'SQFT', 'SQM',
}

DATE_PATTERNS = [
    r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
    r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',
    r'\d{1,2}\s+\w+\s+\d{2,4}',
    r'\w+\s+\d{1,2},?\s+\d{4}',
]


class EvaluationResult:
    """Collects evaluation findings with severity levels."""

    def __init__(self, step: str):
        self.step = step
        self.checks: List[Dict] = []
        self.score = 1.0

    def add(self, check: str, passed: bool, severity: str = "WARNING",
            detail: str = "", penalty: float = 0.0):
        self.checks.append({
            "check": check,
            "passed": passed,
            "severity": severity,
            "detail": detail,
        })
        if not passed:
            self.score = max(0.0, self.score - penalty)

    @property
    def passed(self) -> bool:
        critical_fails = [c for c in self.checks if not c["passed"] and c["severity"] == "CRITICAL"]
        return len(critical_fails) == 0

    @property
    def issues(self) -> List[Dict]:
        return [c for c in self.checks if not c["passed"]]

    def summary(self) -> str:
        total = len(self.checks)
        passed = sum(1 for c in self.checks if c["passed"])
        failed = total - passed
        lines = [f"[EVAL:{self.step}] {passed}/{total} checks passed (score={self.score:.2f})"]
        for c in self.issues:
            lines.append(f"  [{c['severity']}] {c['check']}: {c['detail']}")
        return "\n".join(lines)


# ─── STEP 1: PRE-REVIEW EXTRACTION CHECKS ───────────────

def evaluate_extraction(parsed: dict, items: list, source_text: str = "") -> EvaluationResult:
    """Run after LLM extraction, BEFORE sending to review_chunk.
    Catches garbage responses before wasting an LLM review call."""
    ev = EvaluationResult("extraction")

    # 1. JSON structure valid
    ev.add("json_has_lineitems", "LineItems" in parsed or len(items) > 0,
           severity="CRITICAL", detail="No LineItems key in response", penalty=0.5)

    # 2. Not empty
    ev.add("items_not_empty", len(items) > 0,
           severity="CRITICAL", detail="0 items extracted", penalty=0.5)

    # 3. No all-null items
    all_null_items = []
    for i, item in enumerate(items):
        fields = ['PartNo', 'ItemCode', 'ItemDescription', 'Quantity', 'UnitPrice']
        if all(item.get(f) is None for f in fields):
            all_null_items.append(i)
    ev.add("no_all_null_items", len(all_null_items) == 0,
           severity="WARNING", detail=f"{len(all_null_items)} items have all fields null: {all_null_items[:5]}",
           penalty=0.1 * len(all_null_items))

    # 4. Quantity format (digits only)
    bad_qty = []
    for i, item in enumerate(items):
        qty = item.get('Quantity')
        if qty is not None and not re.match(r'^[\d.]+$', str(qty)):
            bad_qty.append((i, qty))
    ev.add("quantity_format", len(bad_qty) == 0,
           severity="WARNING", detail=f"Non-numeric Quantity: {bad_qty[:3]}", penalty=0.05 * len(bad_qty))

    # 5. UnitPrice format (digits only)
    bad_price = []
    for i, item in enumerate(items):
        price = item.get('UnitPrice')
        if price is not None and not re.match(r'^[\d.]+$', str(price)):
            bad_price.append((i, price))
    ev.add("unitprice_format", len(bad_price) == 0,
           severity="WARNING", detail=f"Non-numeric UnitPrice: {bad_price[:3]}", penalty=0.05 * len(bad_price))

    # 6. Duplicate detection (exact same item repeated)
    seen_keys = set()
    exact_dupes = 0
    for item in items:
        key = (str(item.get('PartNo', '')), str(item.get('ItemDescription', ''))[:50],
               str(item.get('Quantity', '')), str(item.get('UnitPrice', '')))
        if key in seen_keys:
            exact_dupes += 1
        seen_keys.add(key)
    ev.add("no_exact_duplicates", exact_dupes == 0,
           severity="WARNING", detail=f"{exact_dupes} exact duplicate items in single extraction",
           penalty=0.05 * exact_dupes)

    # 7. Reasonable item count (not suspiciously high)
    ev.add("item_count_reasonable", len(items) <= 500,
           severity="WARNING", detail=f"{len(items)} items — suspiciously high, may include non-item rows",
           penalty=0.1)

    # 8. Header: InvoiceNo not null (most invoices have one)
    invoice_no = parsed.get('InvoiceNo')
    ev.add("has_invoice_no", invoice_no is not None,
           severity="INFO", detail="InvoiceNo is null", penalty=0.02)

    logger.info(ev.summary())
    return ev


# ─── STEP 2: BUSINESS RULE VALIDATION ───────────────────

def evaluate_business_rules(header: dict, items: list) -> EvaluationResult:
    """Run in verify_final. Checks business logic that LLM review can't enforce."""
    ev = EvaluationResult("business_rules")

    # 1. Currency is valid ISO 4217
    currency = header.get('InvoiceCurrency')
    if currency:
        ev.add("currency_iso4217", str(currency).upper() in VALID_CURRENCIES,
               severity="WARNING", detail=f"'{currency}' not a recognized ISO 4217 code", penalty=0.05)

    # 2. IncoTerms valid
    incoterms = header.get('IncoTerms')
    if incoterms:
        ev.add("incoterms_valid", str(incoterms).upper() in VALID_INCOTERMS,
               severity="WARNING", detail=f"'{incoterms}' not in valid set {VALID_INCOTERMS}", penalty=0.03)

    # 3. Date is parseable
    date_str = header.get('Date')
    if date_str:
        date_valid = any(re.search(p, str(date_str)) for p in DATE_PATTERNS)
        ev.add("date_parseable", date_valid,
               severity="WARNING", detail=f"Date '{date_str}' doesn't match any known format", penalty=0.03)

    # 4. Exporter/Importer have names
    for party in ['Exporter', 'Importer']:
        p = header.get(party)
        has_name = isinstance(p, dict) and p.get('Name')
        ev.add(f"{party.lower()}_has_name", bool(has_name),
               severity="INFO", detail=f"{party} name is missing", penalty=0.02)

    # 5. UOM normalization check
    bad_uoms = []
    for i, item in enumerate(items):
        uom = item.get('UnitOfQty')
        if uom and str(uom).upper() not in VALID_UOMS:
            bad_uoms.append((i, uom))
    ev.add("uom_normalized", len(bad_uoms) == 0,
           severity="INFO", detail=f"Non-standard UOM: {bad_uoms[:5]}", penalty=0.02)

    # 6. RITC format (digits only)
    bad_ritc = []
    for i, item in enumerate(items):
        ritc = item.get('RITC')
        if ritc and not re.match(r'^\d+$', str(ritc)):
            bad_ritc.append((i, ritc))
    ev.add("ritc_digits_only", len(bad_ritc) == 0,
           severity="WARNING", detail=f"Non-digit RITC values: {bad_ritc[:3]}", penalty=0.03)

    # 7. Country of Origin — full name check
    bad_countries = []
    for i, item in enumerate(items):
        coo = item.get('CountryOfOrigin')
        if coo and len(str(coo)) <= 3:
            bad_countries.append((i, coo))
    ev.add("country_full_name", len(bad_countries) == 0,
           severity="WARNING",
           detail=f"Country codes instead of full names: {bad_countries[:3]}",
           penalty=0.03)

    # 8. No item with description but null quantity AND null price
    phantom_items = []
    for i, item in enumerate(items):
        if item.get('ItemDescription') and item.get('Quantity') is None and item.get('UnitPrice') is None:
            phantom_items.append(i)
    ev.add("no_phantom_items", len(phantom_items) == 0,
           severity="WARNING",
           detail=f"{len(phantom_items)} items have description but no Qty and no Price: {phantom_items[:5]}",
           penalty=0.05 * len(phantom_items))

    logger.info(ev.summary())
    return ev


# ─── STEP 3: MATH CONSISTENCY ───────────────────────────

def evaluate_math_consistency(items: list) -> EvaluationResult:
    """Check Quantity × UnitPrice math where both values exist."""
    ev = EvaluationResult("math_consistency")

    checkable = 0
    mismatches = []

    for i, item in enumerate(items):
        qty_str = item.get('Quantity')
        price_str = item.get('UnitPrice')
        if qty_str is None or price_str is None:
            continue

        try:
            qty = float(str(qty_str).replace(',', ''))
            price = float(str(price_str).replace(',', ''))
        except (ValueError, TypeError):
            continue

        checkable += 1

        # If there's a total amount field, check math
        # Even without total, check for obviously wrong values
        if qty <= 0:
            mismatches.append({"item": i, "issue": f"Quantity <= 0: {qty}"})
        if price < 0:
            mismatches.append({"item": i, "issue": f"UnitPrice < 0: {price}"})
        if qty > 1_000_000:
            mismatches.append({"item": i, "issue": f"Quantity suspiciously high: {qty}"})
        if price > 10_000_000:
            mismatches.append({"item": i, "issue": f"UnitPrice suspiciously high: {price}"})

    ev.add("values_checkable", checkable > 0,
           severity="INFO", detail="No items with both Quantity and UnitPrice", penalty=0.0)

    ev.add("no_negative_values", not any("< 0" in m["issue"] for m in mismatches),
           severity="CRITICAL", detail=str([m for m in mismatches if "< 0" in m["issue"]][:3]),
           penalty=0.2)

    ev.add("no_zero_quantity", not any("<= 0" in m["issue"] for m in mismatches),
           severity="WARNING", detail=str([m for m in mismatches if "<= 0" in m["issue"]][:3]),
           penalty=0.1)

    suspicious = [m for m in mismatches if "suspiciously" in m["issue"]]
    ev.add("no_suspicious_values", len(suspicious) == 0,
           severity="WARNING", detail=str(suspicious[:3]), penalty=0.05)

    logger.info(ev.summary())
    return ev


# ─── STEP 4: DEDUP QUALITY ──────────────────────────────

def evaluate_dedup(before_count: int, after_count: int, expected_count: Optional[int] = None) -> EvaluationResult:
    """Check if dedup removed too many or too few items."""
    ev = EvaluationResult("dedup")

    if before_count == 0:
        ev.add("has_items", False, severity="CRITICAL", detail="0 items before dedup", penalty=0.5)
        return ev

    removal_rate = (before_count - after_count) / before_count

    # Flag if more than 40% removed — aggressive dedup
    ev.add("removal_rate_sane", removal_rate <= 0.4,
           severity="WARNING",
           detail=f"Dedup removed {removal_rate:.0%} of items ({before_count}→{after_count}). Possible false positives.",
           penalty=0.1)

    # If we have expected count from review, check alignment
    if expected_count and expected_count > 0:
        ratio = after_count / expected_count
        ev.add("matches_expected_count", 0.8 <= ratio <= 1.2,
               severity="WARNING",
               detail=f"After dedup: {after_count} items, expected ~{expected_count} (ratio={ratio:.2f})",
               penalty=0.1)

    logger.info(ev.summary())
    return ev


# ─── STEP 5: CODEGEN FIELD-LEVEL ACCURACY ───────────────

def evaluate_codegen_output(golden_items: list, generated_items: list) -> EvaluationResult:
    """Field-level comparison between golden (LLM) and generated (regex) extraction.
    Goes deeper than just item count — checks if values actually match."""
    ev = EvaluationResult("codegen")

    max_compare = min(len(golden_items), len(generated_items), 20)
    if max_compare == 0:
        ev.add("has_items", False, severity="CRITICAL", detail="No items to compare", penalty=1.0)
        return ev

    fields_to_check = ['PartNo', 'ItemDescription', 'Quantity', 'UnitOfQty', 'UnitPrice']
    total_fields = 0
    matching_fields = 0
    mismatches_by_field = {f: 0 for f in fields_to_check}

    for i in range(max_compare):
        golden = golden_items[i]
        generated = generated_items[i]
        for field in fields_to_check:
            g_val = _norm(golden.get(field))
            r_val = _norm(generated.get(field))
            if not g_val and not r_val:
                continue
            total_fields += 1
            if g_val == r_val:
                matching_fields += 1
            else:
                mismatches_by_field[field] = mismatches_by_field.get(field, 0) + 1

    field_accuracy = matching_fields / max(total_fields, 1)

    ev.add("field_accuracy_above_80", field_accuracy >= 0.8,
           severity="CRITICAL",
           detail=f"Field accuracy: {field_accuracy:.2f} ({matching_fields}/{total_fields})",
           penalty=0.3 if field_accuracy < 0.8 else 0.0)

    # Per-field breakdown
    for field, mismatch_count in mismatches_by_field.items():
        if mismatch_count > 0:
            ev.add(f"{field}_accuracy", mismatch_count <= max_compare * 0.2,
                   severity="WARNING",
                   detail=f"{field}: {mismatch_count}/{max_compare} mismatches",
                   penalty=0.05)

    # Item count match
    count_ratio = len(generated_items) / max(len(golden_items), 1)
    ev.add("item_count_match", count_ratio >= 0.8,
           severity="CRITICAL",
           detail=f"Generated {len(generated_items)} vs golden {len(golden_items)} (ratio={count_ratio:.2f})",
           penalty=0.2 if count_ratio < 0.8 else 0.0)

    ev.score = field_accuracy

    logger.info(ev.summary())
    return ev


# ─── STEP 6: TEMPLATE HEALTH ────────────────────────────

def evaluate_template_health(entry: dict) -> EvaluationResult:
    """Check if a template is still reliable based on usage history."""
    ev = EvaluationResult("template_health")

    times_used = entry.get('times_used', 0)
    success_count = entry.get('success_count', 0)

    if times_used == 0:
        ev.add("has_usage", False, severity="INFO", detail="Template never used", penalty=0.0)
        return ev

    success_rate = success_count / times_used

    ev.add("success_rate_above_70", success_rate >= 0.7,
           severity="CRITICAL",
           detail=f"Template success rate: {success_rate:.2f} ({success_count}/{times_used})",
           penalty=0.5 if success_rate < 0.7 else 0.0)

    ev.add("sufficient_usage", times_used >= 3,
           severity="INFO",
           detail=f"Only {times_used} uses — not enough data for confidence",
           penalty=0.0)

    ev.score = success_rate
    logger.info(ev.summary())
    return ev


# ─── STEP 7: PATTERN HEALTH ─────────────────────────────

def evaluate_pattern_health(patterns: list, max_age_days: int = 30) -> EvaluationResult:
    """Check pattern library for stale/underperforming patterns."""
    import time
    ev = EvaluationResult("pattern_health")

    now = time.time()
    stale = []
    underperforming = []

    for p in patterns:
        pid = p.get('pattern_id', '?')
        last_used = p.get('last_used') or p.get('created_at', 0)
        days_since = (now - last_used) / 86400 if last_used else 999

        if days_since > max_age_days and p.get('times_used', 0) > 0:
            stale.append(pid)

        success_rate = p.get('success_rate', 0)
        times_used = p.get('times_used', 0)
        if times_used >= 5 and success_rate < 0.5:
            underperforming.append((pid, success_rate, times_used))

    ev.add("no_stale_patterns", len(stale) == 0,
           severity="INFO",
           detail=f"{len(stale)} patterns unused for {max_age_days}+ days: {stale[:5]}",
           penalty=0.0)

    ev.add("no_underperforming_patterns", len(underperforming) == 0,
           severity="WARNING",
           detail=f"{len(underperforming)} patterns below 50% success: {underperforming[:3]}",
           penalty=0.05 * len(underperforming))

    logger.info(ev.summary())
    return ev


# ─── STEP 8: KB CONFIDENCE GATING ───────────────────────

def should_use_kb_context(format_confidence: float, threshold: float = 0.5) -> bool:
    """Returns False if KB format detection confidence is too low.
    Injecting wrong format rules makes extraction worse."""
    if format_confidence < threshold:
        logger.info(f"[EVAL] KB confidence {format_confidence:.2f} < {threshold} — skipping KB context injection")
        return False
    return True


# ─── HELPERS ─────────────────────────────────────────────

def _norm(val) -> str:
    if val is None:
        return ''
    return str(val).strip().lower().replace(',', '').replace(' ', '')
