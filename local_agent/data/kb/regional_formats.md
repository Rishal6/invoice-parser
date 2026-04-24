# Regional Invoice Formats

Different countries have distinct invoice standards with specific fiscal requirements, tax structures, and field layouts. This document covers the major regional formats and what to watch for during extraction.

---

## Mexican CFDI (Comprobante Fiscal Digital por Internet)

Mexico mandates electronic invoicing through the SAT (Servicio de Administracion Tributaria). CFDI invoices carry digital tax certification and contain dense fiscal metadata mixed into the document.

**Key identifiers:**

- SAT digital certificate and digital seal (long alphanumeric strings)
- UUID fiscal folio — a unique 36-character identifier (e.g., 3F2504E0-4F89-11D3-9A0C-0305E82C3301) that serves as the official invoice reference
- RFC (Registro Federal de Contribuyentes) — Mexican tax ID for both emisor (issuer) and receptor (receiver)

**Critical field traps:**

- CLAVEPROD SERV is the SAT tax product classification code (e.g., H87, 84719090). This is a fiscal catalog code, NOT a part number. Never extract it as PartNo.
- Fraccion Arancelaria (often abbreviated F.A.) is the Mexican customs tariff code. This IS the equivalent of an HS code or RITC code and should be extracted as such. Example: F.A.:8708809900.
- Unidad Aduana is the customs unit of measure, not the sales unit.

**Description field challenges:**

Product descriptions in CFDI invoices often contain a dense mix of commercial and fiscal data in a single block. A typical description block might include the product name, P.O. reference number, customs tariff (F.A.), country of origin, SAT product code, tax breakdown (Codigo Impuesto, Impuesto Traslado), Marca (brand), and Mercancias (goods classification) all concatenated together.

When extracting, separate the actual product description from the inline fiscal metadata. Look for the P.O. reference embedded in the description — it often appears as "PO: XXXXX" or "O.C.: XXXXX". Country of origin frequently appears at the end of the description block as "COUNTRY OF ORIGIN MEX" or "PAIS DE ORIGEN: MEXICO".

---

## Chinese Fapiao (发票)

China uses a government-regulated invoice system. Export invoices from Chinese suppliers often appear in dual language format.

**Key identifiers:**

- Red official stamp (circular seal) indicating tax bureau authentication
- 发票代码 (Invoice Code) and 发票号码 (Invoice Number) — the official fiscal identifiers
- 纳税人识别号 — Taxpayer Identification Number (for both buyer and seller)
- Total amount written in Chinese characters at the bottom of the invoice as a fraud-prevention measure

**Column headers in Chinese:**

- 品名 or 货物名称 = Product name / Description
- 规格型号 = Specification / Model
- 数量 = Quantity
- 单价 = Unit price
- 金额 = Amount
- 税率 = Tax rate
- 税额 = Tax amount
- 物料号 = Material number (this IS a part number equivalent)

**Extraction notes:**

Dual-language invoices may have Chinese above and English below (or vice versa). Prefer the English description when available, but use Chinese headers to identify column positions. The 物料号 field maps to PartNo in extraction.

---

## EU VAT Invoice

European Union invoices follow VAT (Value Added Tax) directive requirements. Intra-community transactions (between EU member states) have specific rules.

**Key identifiers:**

- VAT registration number for the seller (mandatory)
- VAT registration number for the buyer (mandatory for B2B intra-community supply)
- VAT number format varies by country: DE + 9 digits (Germany), FR + 11 characters (France), NL + 12 characters (Netherlands), etc.

**Tax complexity:**

- Multiple VAT rates may apply on a single invoice — different line items can carry different rates (e.g., 19% standard, 7% reduced in Germany)
- Reverse charge mechanism: On intra-community B2B supplies, the invoice may show 0% VAT with a note like "Reverse charge — VAT to be accounted for by the recipient" or "Innergemeinschaftliche Lieferung". The buyer self-assesses VAT. Do not treat 0% reverse charge as tax-free.
- Some invoices show both net amount and gross amount per line item

**Currency:**

Usually EUR, but not always. UK invoices use GBP, Swiss invoices use CHF, Scandinavian countries may use SEK, NOK, or DKK. Always check the stated currency rather than assuming EUR.

**Common field labels by language:**

- German: Rechnungsnummer (Invoice No.), Rechnungsdatum (Invoice Date), Menge (Quantity), Einzelpreis (Unit Price), Kundenmaterialnummer (Customer Material Number)
- French: Numero de facture, Date de facture, Quantite, Prix unitaire
- Spanish: Numero de factura, Fecha de factura, Cantidad, Precio unitario

---

## Japanese Invoice (適格請求書)

Since October 2023, Japan's Qualified Invoice System (インボイス制度) requires specific fields for consumption tax credit claims.

**Key identifiers:**

- Registration number: "T" followed by 13 digits (e.g., T1234567890123) — identifies the seller as a qualified invoice issuer
- Title may read 適格請求書 (Qualified Invoice) or simply 請求書 (Invoice)

**Tax structure:**

Japan has two consumption tax rates applied simultaneously:
- 8% reduced rate (軽減税率) — applies to food and beverages, newspapers
- 10% standard rate (標準税率) — applies to most other goods and services

Invoices must show the tax amount calculated separately for each rate. Look for a tax summary section at the bottom showing the breakdown.

**Common field labels:**

- 品番 = Part Number (this IS a part number, extract as PartNo)
- 品名 = Product Name / Description
- 数量 = Quantity
- 単価 = Unit Price
- 金額 = Amount
- 消費税 = Consumption Tax
- 合計 = Total

**Date format:** Japanese invoices often use the format YYYY年MM月DD日 (e.g., 2024年03月15日) or the Japanese era calendar (令和6年3月15日). Convert to standard date format during extraction.

---

## Indian Tax Invoice / GST Invoice

India's Goods and Services Tax (GST) system requires specific fields on every tax invoice.

**Key identifiers:**

- GSTIN (Goods and Services Tax Identification Number) for both seller and buyer — exactly 15 alphanumeric characters (e.g., 27AAPFU0939F1ZV). The first two digits represent the state code.
- HSN code (Harmonized System Nomenclature) for goods or SAC code (Service Accounting Code) for services — mandatory on every line item. HSN codes at 4, 6, or 8 digits depending on the supplier's turnover threshold.
- Invoice titled "Tax Invoice" (mandatory wording under GST law)

**Tax structure:**

The tax split depends on whether the transaction is intra-state or inter-state:
- Intra-state (seller and buyer in same state): Tax splits into CGST (Central) + SGST (State), each at half the total rate. Example: 18% GST = 9% CGST + 9% SGST.
- Inter-state (seller and buyer in different states): Tax is IGST (Integrated) at the full rate. Example: 18% IGST.

These are NOT separate taxes — they are the same GST amount split differently based on geography. Extract the total tax rate and the component breakdown.

**Additional fields:**

- E-way bill number: Required for transport of goods exceeding INR 50,000 in value. May appear on the invoice or as a separate document reference.
- Place of supply: Determines whether CGST+SGST or IGST applies. Usually a state name or state code.
- IRN (Invoice Reference Number): For e-invoicing compliance, a unique hash generated by the GST portal.
- QR code: Mandatory for B2C invoices above a threshold, contains invoice summary data.

**Currency:** Almost always INR. Export invoices from India may show USD or other foreign currency alongside INR equivalent.
