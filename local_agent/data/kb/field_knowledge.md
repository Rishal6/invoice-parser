# Field-Specific Extraction Knowledge

This document covers the nuances, aliases, and regional variations for key invoice fields that are commonly misidentified or missed during extraction.

---

## Part Number (PartNo)

Part numbers appear under many different labels depending on the supplier's country, industry, and ERP system. Not every code on an invoice is a part number — tax codes and classification codes are frequently confused with part numbers.

**Priority order for extraction (highest to lowest):**

1. Customer Material Description — used by German automotive suppliers (Bosch, Continental, ZF, Schaeffler). This is the buyer's own part number and takes highest priority because it maps directly to the buyer's inventory system.
2. Part No. / Part Number
3. Material No. / Material Number
4. Item No. / Item Number
5. Product Code
6. Article No. / Article Number

**Regional variants:**

- German invoices: "Kundenmaterialnummer" or "Customer Material Description" — highest priority, always prefer this over the supplier's own part number
- Japanese invoices: 品番 = Part Number
- Chinese invoices: 物料号 = Material Number, 规格型号 = Specification/Model (sometimes used as part number)
- Korean invoices: 품번 = Part Number

**Common traps — these are NOT part numbers:**

- CLAVEPROD SERV on Mexican CFDI invoices — this is the SAT tax product classification code. It looks like a part number but is a fiscal catalog code. Ignore it as a part number source.
- HSN/SAC codes on Indian invoices — these are tax classification codes, not part numbers.
- HS Code / Tariff Code on any invoice — these are customs classification codes.
- UNSPSC codes — United Nations product classification, not part numbers.

When multiple number-like fields appear on a line item, use context to determine which is the actual part number: part numbers are typically alphanumeric with dashes or dots (e.g., "A2C-3920-1045", "BX.450.331"), while tax/tariff codes are purely numeric with a fixed digit count (e.g., "84719090", "8708.80").

---

## RITC / HS Code

RITC (Regional Import Tariff Code) and HS (Harmonized System) codes classify goods for customs purposes. The first 6 digits follow the international HS standard; digits beyond 6 are country-specific extensions.

**Digit structure:**

- 6 digits = international HS code (recognized worldwide)
- 8 digits = most common country-specific extension (EU CN code, US HTS code, India ITC-HS)
- 10 digits = some countries extend further (US HTS uses 10 digits)

**Regional equivalents — all map to RITC/HS Code:**

- Mexico: Fraccion Arancelaria (F.A.) — the Mexican customs tariff. Extract as HS/RITC code. Example: F.A.:8708809900
- India: HSN Code (Harmonized System Nomenclature) — same underlying HS system. Example: HSN:84719090
- EU: CN Code (Combined Nomenclature) — HS code + 2 EU-specific digits
- US: HTS Code (Harmonized Tariff Schedule) — HS code + 4 US-specific digits
- Japan: 関税番号 = Tariff number

**Where to find HS codes on invoices:**

Sometimes the HS code appears in a dedicated labeled column. Other times it is buried within the product description text. When embedded in descriptions, look for these patterns:

- "F.A.:8708809900" or "FRACC. ARANC.: 8708.80.99.00"
- "HSN:84719090" or "HSN Code: 8471.90.90"
- "HS Code: 8471.90" or "HS: 847190"
- "Tariff: 8708.80.9900"
- "RITC: 87088099"

HS codes may include dots as separators (8708.80.99) or appear as a continuous number (8708809900). Normalize to the continuous format without dots for output.

---

## Incoterms (Terms of Delivery)

Incoterms define the responsibilities of buyer and seller for shipping, insurance, and customs. They are rarely labeled explicitly as "Incoterms" on invoices.

**Where to look on an invoice:**

- "Delivery Terms" or "Terms of Delivery"
- "Freight Terms" or "Shipping Terms"
- "Shipment Terms"
- "Trade Terms"
- Near the port of loading/discharge section
- In the header area alongside payment terms

Incoterms almost always appear with a location: "FOB Shanghai", "CIF Hamburg", "FCA Suzhou Industrial Park", "DDP Los Angeles". Extract both the term and the location.

**Standard Incoterms 2020 and their mapping:**

- EXW (Ex Works) → maps to FOB
- FCA (Free Carrier) → maps to FOB
- FOB (Free on Board) → maps to FOB
- CFR (Cost and Freight) → maps to CF
- CIF (Cost, Insurance, Freight) → maps to CIF
- CIP (Carriage and Insurance Paid To) → maps to CI
- DAP (Delivered at Place) → maps to FOB
- DDP (Delivered Duty Paid) → maps to FOB
- FAS (Free Alongside Ship) → maps to FOB
- CPT (Carriage Paid To) → maps to CF

**Common abbreviations on invoices:**

Some invoices use shorthand: "C&F" = CFR, "C&I" = CIF, "C.I.F." = CIF, "F.O.B." = FOB. Normalize these to standard Incoterm codes before applying the mapping.

---

## Currency

Every invoice has a transaction currency. Always normalize to the 3-letter ISO 4217 code.

**Common currencies and their symbols/representations:**

- USD — $, US$, U.S. Dollars
- EUR — euro sign, Euros
- INR — Rs, Rs., Rupees, Indian Rupees
- JPY — yen sign, Japanese Yen (no decimal places)
- CNY — RMB, Chinese Yuan, Renminbi, yuan sign
- MXN — Mex$, Mexican Pesos, M.N. (Moneda Nacional)
- KRW — Korean Won, W (no decimal places)
- GBP — pound sign, British Pounds, Sterling
- TWD — NT$, New Taiwan Dollars
- THB — Thai Baht, B

**Dual currency invoices:**

Common in electronics and semiconductor industries. A single invoice may show amounts in two currencies:
- KRW + USD (Korean electronics suppliers)
- TWD + USD (Taiwanese component suppliers)
- CNY + USD (Chinese export invoices)

When dual currencies appear, extract the primary invoice currency (the one used for the total amount due). Note the secondary currency if present but do not confuse it with the billing currency.

**Currency vs. payment currency:**

Some invoices specify a different payment currency from the invoice currency (e.g., invoice in EUR, payment in USD at a stated exchange rate). Extract the invoice currency — the one associated with the line item prices and totals.

---

## Country of Origin

The country where goods were manufactured or substantially transformed. Required for customs clearance on international invoices.

**Code normalization — always convert short codes to full country names:**

- MEX → Mexico
- CHN → China
- JPN → Japan
- DEU → Germany
- KOR → South Korea (not just "Korea")
- USA → United States
- IND → India
- TWN → Taiwan
- THA → Thailand
- VNM → Vietnam
- MYS → Malaysia
- IDN → Indonesia
- GBR → United Kingdom
- FRA → France
- ITA → Italy
- BRA → Brazil

**Where to find on invoices:**

- Dedicated "Country of Origin" or "Origin" column per line item
- A single declaration for the entire invoice (applies to all line items)
- Embedded at the end of a product description block: "COUNTRY OF ORIGIN MEX", "ORIGIN: CHINA", "Made in Japan"
- In a separate origin declaration section near the bottom of the invoice
- As part of a customs data block alongside HS codes

**Per-item vs. per-invoice:**

Some invoices ship goods from multiple origins. When a Country of Origin column exists per line item, extract it per line. When a single origin statement appears once on the invoice, apply it to all line items.
