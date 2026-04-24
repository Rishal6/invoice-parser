# Document Types — Classification Guide

When processing a document, first determine its type before attempting extraction. Only Commercial Invoices should be fully extracted. All other document types should be classified and skipped.

---

## Commercial Invoice

A commercial invoice is the primary billing document in international trade, issued by the seller (exporter) to the buyer (importer) for goods shipped. It serves as the basis for customs clearance, payment, and tax assessment.

**Required fields that confirm a document is a commercial invoice:**

- Invoice number and invoice date
- Exporter name and address (seller, shipper, or supplier)
- Importer name and address (buyer, consignee, or bill-to party)
- Line items with product descriptions AND unit prices AND quantities AND line totals
- Total invoice value with currency
- Payment terms or trade terms

**How to confirm it is an invoice:**

The defining characteristic is the presence of line items with monetary values (unit price, amount, total). If the document has itemized products with prices and a total amount due, it is almost certainly an invoice. Look for headings like "Invoice", "Commercial Invoice", "Tax Invoice", "Factura", "Rechnung", or "請求書".

---

## Bill of Lading

A bill of lading is a shipping and transport document issued by the carrier or freight forwarder. It is proof of shipment, not a billing document.

**Distinguishing features:**

- Has shipper, consignee, and notify party fields
- Contains container numbers, seal numbers, vessel name, voyage number, port of loading, port of discharge
- Describes goods in general terms (number of packages, gross weight, volume)
- Does NOT contain unit prices or line item totals
- Often titled "Bill of Lading", "B/L", "Ocean Bill of Lading", "House Bill of Lading", or "Waybill"

**Action:** Classify as BILL_OF_LADING and skip extraction. The absence of prices is the clearest signal.

---

## Packing List

A packing list accompanies the shipment and describes how goods are packed. It looks very similar to an invoice because it lists the same products, but it contains no pricing information.

**Distinguishing features:**

- Lists products with quantities, but NO unit prices and NO monetary totals
- Includes physical details: net weight, gross weight, dimensions (L x W x H), carton counts, pallet counts
- May reference the related invoice number
- Often titled "Packing List", "Packing Slip", or "Delivery Note"

**Action:** Classify as PACKING_LIST and skip extraction. The key difference from an invoice is the absence of unit prices and the presence of weight/dimension data instead.

---

## Purchase Order

A purchase order is issued BY the buyer TO the seller, requesting goods or services. It may contain prices and quantities, making it look like an invoice at first glance.

**Distinguishing features:**

- The issuing party is the buyer, not the seller
- Titled "Purchase Order", "P.O.", or "Order Confirmation"
- Contains a P.O. number (not an invoice number)
- Prices listed are requested or agreed prices, not billed amounts
- May have a "Ship To" address separate from the buyer address
- Often includes requested delivery dates

**Action:** Classify as PURCHASE_ORDER and skip extraction. The direction of the document (buyer to seller) is the opposite of an invoice (seller to buyer).

---

## Proforma Invoice

A proforma invoice is a preliminary or estimated invoice sent before goods are shipped. It is not a final commercial invoice and should not be treated as one.

**Distinguishing features:**

- Explicitly titled "Proforma Invoice", "Pro Forma Invoice", or "PI"
- Prices and totals are estimates or quotations, not final charges
- Often used for customs pre-clearance or advance payment requests
- May not have a final invoice number (uses PI number instead)

**Action:** Classify as PROFORMA_INVOICE and skip extraction. It is not a final billing document.

---

## Credit Note / Debit Note

A credit note (or debit note) is an adjustment document issued after the original invoice. It may contain line items with amounts, but it corrects or adjusts an existing invoice rather than billing for new goods.

**Distinguishing features:**

- Titled "Credit Note", "Credit Memo", "Debit Note", or "Debit Memo"
- References an original invoice number
- Amounts may be negative (credits) or positive (additional charges)
- Reason for adjustment is usually stated (returns, pricing corrections, quantity discrepancies)

**Action:** Classify as CREDIT_DEBIT_NOTE and skip extraction. It is a correction, not a standalone invoice.
