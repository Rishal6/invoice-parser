# Invoice Parser

AI-powered invoice extraction system built on **AWS Bedrock + Strands Agents**. Extracts every line item from PDF invoices with LLM-verifies-LLM architecture, mandatory worker verification, and a self-improving feedback loop.

**Live URL:** (populated after deployment)

## Architecture

```
PDF Upload → Split → Extract (parallel) → Review → Dedup → Verify → Worker Review → Complete
                ↑                              ↓
            Knowledge Base ←── Pattern Library ←── Feedback Processor
```

### Key Design Decisions

| Decision | Why |
|---|---|
| **LLM extracts, LLM reviews** | No regex/Textract. Handles any invoice format worldwide |
| **Strip-based vision** | Dense pages split into overlapping vertical strips for OCR accuracy |
| **Mandatory worker verification** | Every extraction goes through human review before completion |
| **RLHF feedback loop** | Worker corrections become patterns, patterns graduate to Knowledge Base |
| **Storage abstraction** | Same code runs locally (files) or on AWS (S3 + DynamoDB) |

### Tech Stack

- **Orchestrator**: Strands SDK Agent (AWS Bedrock Claude Sonnet)
- **Backend**: FastAPI + Gunicorn
- **Infrastructure**: ECS Fargate, ALB, S3, DynamoDB, CodeBuild
- **Deployment**: CloudFormation (single `deploy.sh`)

## Quick Start (Local)

```bash
cd local_agent
pip install -r requirements.txt

# Set AWS credentials (needs Bedrock access)
export AWS_REGION=us-east-1

# Run API server
python api.py
# → http://localhost:8000

# Or run CLI extraction
python agent.py invoice.pdf
```

## Quick Start (AWS)

```bash
# Prerequisites: AWS CLI configured with Bedrock-enabled account
./deploy.sh us-east-1

# Outputs the ALB URL — share this with your manager
```

The deploy script:
1. Creates all AWS resources (S3, DynamoDB, ECR, ECS, ALB, CodeBuild)
2. Packages source code and uploads to S3
3. Builds Docker image on AWS via CodeBuild (no local Docker needed)
4. Deploys container to ECS Fargate
5. Returns the public URL

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Web UI for upload and results |
| `POST` | `/parse/async` | Upload PDF, returns job_id |
| `GET` | `/jobs/{job_id}` | Poll job status |
| `GET` | `/result/{job_id}` | Get extraction result |
| `GET` | `/jobs/{job_id}/trace` | Step-by-step execution trace |
| `GET` | `/verify/ui/dashboard` | Worker verification dashboard |
| `GET` | `/verify/ui/{job_id}` | Verify a specific extraction |
| `POST` | `/verify/{job_id}` | Submit worker corrections |
| `GET` | `/metrics` | Accuracy metrics over time |
| `GET` | `/patterns` | Learned error patterns |
| `GET` | `/health` | Health check |

## Extraction Pipeline

### Phase 1: Split
PDF split into chunks (pages_per_chunk auto-decided by orchestrator). Dense invoices use strip-based vision — each page cut into overlapping vertical strips for accurate OCR.

### Phase 2: Extract + Review (Parallel)
- **LLM #1 (Extractor)**: Reads PDF chunk → structured JSON (InvoiceNo, Date, Exporter, LineItems, etc.)
- **LLM #2 (Reviewer)**: Cross-checks extraction against source text, scores quality, flags issues
- Auto-retries failed chunks up to 2x with reviewer's fix instructions
- Up to 5 chunks processed concurrently

### Phase 3: Dedup + Verify
- Exact + fuzzy deduplication across chunk boundaries (overlap items)
- Final verification counts items and produces quality score

### Phase 4: Worker Verification (Mandatory)
- Job moves to `AWAITING_VERIFICATION` (not `COMPLETED`)
- Worker reviews in side-by-side UI (extraction on left, source PDF on right)
- Can approve all, correct individual fields, add/remove line items
- Accuracy calculated automatically (field-level diff)

### Phase 5: Feedback Loop
```
Worker corrections → FeedbackProcessor → LLM generates patterns
                                              ↓
                                        Pattern Library (confidence tracking)
                                              ↓
                                  Auto-promote to Knowledge Base
                                  (when success_rate >= 0.9 over 5+ uses)
```

## Extracted Fields

### Header
InvoiceNo, Date, InvoiceCurrency, FreightTerms, IncoTerms, TermsOfPayment, Exporter (Name + Address), Importer (Name + Address), Classification

### Line Items
ItemNo, PartNo, ItemCode, ItemDescription, Quantity, UnitOfQty, UnitPrice, RITC, CountryOfOrigin

## Project Structure

```
invoice-parser/
├── local_agent/           # Core application
│   ├── api.py             # FastAPI app + viewer UI
│   ├── agent.py           # Strands orchestrator agent
│   ├── tools.py           # 7 extraction tools (split, extract, review, etc.)
│   ├── storage.py         # Storage abstraction (local files ↔ S3+DynamoDB)
│   ├── verification.py    # Worker verification workflow
│   ├── feedback.py        # Correction → pattern generation (LLM)
│   ├── patterns.py        # Pattern library + confidence tracking
│   ├── metrics.py         # Accuracy tracking over time
│   ├── tracer.py          # Step-by-step job tracing
│   ├── knowledge.py       # Knowledge Base (format-specific extraction rules)
│   ├── codegen.py         # Template code generation
│   ├── registry.py        # Template registry
│   ├── runner.py           # Sandboxed code execution
│   ├── data/kb/           # Knowledge Base markdown files
│   └── test_integration.py # 48 integration tests
├── infra/
│   └── template.yaml      # CloudFormation (S3, DDB, ECR, ECS, ALB, CodeBuild)
├── Dockerfile             # Multi-stage build
├── deploy.sh              # One-command AWS deployment
└── README.md
```

## AWS Resources Created

| Resource | Purpose | Cost |
|---|---|---|
| S3 Bucket | PDF uploads, source packages | ~$0.02/GB/month |
| DynamoDB Table | Verifications, patterns, metrics, traces | Pay-per-request |
| ECR Repository | Docker images | ~$0.10/GB/month |
| ECS Fargate (1 task) | Application runtime (2 vCPU, 4GB RAM) | ~$70/month |
| ALB | Load balancer + public URL | ~$16/month |
| CodeBuild | Docker builds (on-demand) | ~$0.01/build |
| CloudWatch Logs | Application logs (14-day retention) | ~$0.50/GB |

Estimated monthly cost: **~$90/month** at low volume.

## Monitoring

- **CloudWatch Logs**: `/ecs/invoice-parser` — all application logs
- **Traces**: `GET /jobs/{job_id}/trace` — per-job step-by-step execution with token counts and cost
- **Metrics**: `GET /metrics` — accuracy trends, per-format and per-company breakdown
- **Health**: `GET /health` — active job count

## Testing

```bash
cd local_agent
python -m pytest test_integration.py -v
# 48 tests — covers all modules without hitting Bedrock
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `STORAGE_BACKEND` | `local` | `local` (filesystem) or `aws` (S3 + DynamoDB) |
| `S3_BUCKET` | — | S3 bucket name (required when `aws`) |
| `DYNAMO_TABLE` | — | DynamoDB table name (required when `aws`) |
| `AWS_REGION` | `us-east-1` | AWS region |
| `ORCHESTRATOR_MODEL` | `us.anthropic.claude-sonnet-4-5-*` | Bedrock model for orchestrator |
| `FEEDBACK_MODEL` | `us.anthropic.claude-sonnet-4-5-*` | Bedrock model for feedback processing |
| `PORT` | `8000` | API port |
