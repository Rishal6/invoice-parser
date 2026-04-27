"""
Invoice Parser — FastAPI REST API

Endpoints:
  POST /parse          — Upload PDF, run extraction (sync)
  POST /parse/async    — Upload PDF, start extraction job (background)
  GET  /jobs/{job_id}  — Poll job status
  GET  /result/{job_id} — Get extraction result
  GET  /health         — Health check
"""
import json
import os
import re
import uuid
import time
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

import agent
import storage
from verification import VerificationManager
from feedback import FeedbackProcessor
from patterns import PatternLibrary
from metrics import MetricsTracker

logger = logging.getLogger(__name__)

verification_mgr = VerificationManager()
pattern_lib = PatternLibrary()
feedback_proc = FeedbackProcessor(pattern_library=pattern_lib)
metrics_tracker = MetricsTracker()

UPLOAD_DIR = Path("/tmp/invoice_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR = Path("/tmp/invoice_results")
RESULT_DIR.mkdir(exist_ok=True)

executor = ThreadPoolExecutor(max_workers=2)
jobs: Dict[str, "JobStatus"] = {}


# ─── Pydantic Models ──────────────────────────────────────

class JobState(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    AWAITING_VERIFICATION = "AWAITING_VERIFICATION"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ExporterImporter(BaseModel):
    Name: Optional[str] = None
    Address: Optional[str] = None


class LineItem(BaseModel):
    ItemNo: Optional[int] = None
    PartNo: Optional[str] = None
    ItemCode: Optional[str] = None
    ItemDescription: Optional[str] = None
    Quantity: Optional[str] = None
    UnitOfQty: Optional[str] = None
    UnitPrice: Optional[str] = None
    RITC: Optional[str] = None
    CountryOfOrigin: Optional[str] = None


class DedupSummary(BaseModel):
    before: int = 0
    exact_dupes_removed: int = 0
    fuzzy_dupes_removed: int = 0
    after: int = 0


class ExtractionResult(BaseModel):
    success: bool
    filename: str
    Classification: Optional[str] = None
    InvoiceNo: Optional[str] = None
    Date: Optional[str] = None
    InvoiceCurrency: Optional[str] = None
    FreightTerms: Optional[str] = None
    IncoTerms: Optional[str] = None
    TermsOfPayment: Optional[str] = None
    Exporter: Optional[ExporterImporter] = None
    Importer: Optional[ExporterImporter] = None
    total_items: int = 0
    LineItems: List[LineItem] = []
    quality_score: float = 0.0
    chunks_processed: int = 0
    total_chunks: int = 0
    review_summary: Dict = {}
    dedup_summary: DedupSummary = DedupSummary()
    elapsed_seconds: float = 0.0
    tool_log: List[Dict] = []
    error: Optional[str] = None


class JobProgress(BaseModel):
    items_found: int = 0
    chunks_done: int = 0
    total_chunks: int = 0
    current_phase: str = ""


class JobStatus(BaseModel):
    job_id: str
    status: JobState = JobState.PENDING
    filename: str = ""
    created_at: float = 0.0
    completed_at: Optional[float] = None
    result: Optional[ExtractionResult] = None
    progress: Optional[JobProgress] = None


class ParseResponse(BaseModel):
    job_id: str
    status: JobState
    filename: str


class HealthResponse(BaseModel):
    status: str = "ok"
    active_jobs: int = 0


# ─── FastAPI App ───────────────────────────────────────────

app = FastAPI(
    title="Invoice Parser API",
    description="Agentic invoice extraction — LLM extracts, LLM reviews, Python deduplicates",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _load_stores_from_s3():
    """Load FAISS index + template registry from S3 on container startup."""
    bucket = os.environ.get("S3_BUCKET")
    if not bucket:
        return
    try:
        from vector_store import VectorStore
        vs = VectorStore()
        vs.sync_from_s3(bucket, prefix="formats/")
        logger.info(f"[STARTUP] Vector store loaded from s3://{bucket}/formats/")
    except Exception as e:
        logger.warning(f"[STARTUP] Vector store load skipped: {e}")
    try:
        from registry import TemplateRegistry
        tr = TemplateRegistry()
        tr.sync_from_s3(bucket, prefix="templates/")
        logger.info(f"[STARTUP] Template registry loaded from s3://{bucket}/templates/")
    except Exception as e:
        logger.warning(f"[STARTUP] Template registry load skipped: {e}")


# ─── Viewer HTML ──────────────────────────────────────────

VIEWER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Invoice Parser</title>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#f5f7fa;color:#1a1a2e}
  .header{background:linear-gradient(135deg,#232f3e 0%,#1a1a2e 100%);color:#fff;padding:24px 32px;display:flex;justify-content:space-between;align-items:center}
  .header h1{font-size:22px;font-weight:600}
  .header p{font-size:13px;color:#a0aec0;margin-top:4px}
  .header-nav{display:flex;gap:12px}
  .header-nav a{color:#ff9900;text-decoration:none;font-size:14px;font-weight:600;padding:6px 14px;border:1px solid #ff9900;border-radius:6px;transition:all .2s}
  .header-nav a:hover{background:#ff9900;color:#fff}
  .controls{display:flex;gap:12px;align-items:center;padding:20px 32px;background:#fff;border-bottom:1px solid #e2e8f0;flex-wrap:wrap}
  .controls input,.controls select{padding:10px 14px;border:1px solid #cbd5e0;border-radius:8px;font-size:14px}
  .controls input[type="text"]{flex:1;min-width:200px}
  .controls input[type="file"]{border:none}
  .btn{padding:10px 20px;border:none;border-radius:8px;font-size:14px;font-weight:600;cursor:pointer;transition:all .2s}
  .btn-primary{background:#ff9900;color:#fff}
  .btn-primary:hover{background:#ec8f00}
  .btn-secondary{background:#edf2f7;color:#4a5568}
  .btn-secondary:hover{background:#e2e8f0}
  .btn-export{background:#38a169;color:#fff}
  .btn-export:hover{background:#2f855a}
  .btn-danger{background:#e53e3e;color:#fff}
  .btn-danger:hover{background:#c53030}
  .status-bar{display:flex;gap:20px;padding:16px 32px;background:#fff;border-bottom:1px solid #e2e8f0;flex-wrap:wrap}
  .stat{text-align:center}
  .stat-value{font-size:24px;font-weight:700;color:#232f3e}
  .stat-label{font-size:11px;color:#718096;text-transform:uppercase;letter-spacing:.5px}
  .badge{display:inline-block;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600}
  .badge-green{background:#c6f6d5;color:#276749}
  .badge-yellow{background:#fefcbf;color:#975a16}
  .badge-red{background:#fed7d7;color:#9b2c2c}
  .badge-blue{background:#bee3f8;color:#2a4365}
  .content{padding:20px 32px}
  .tabs{display:flex;gap:4px;margin-bottom:16px}
  .tab{padding:8px 16px;border-radius:8px 8px 0 0;cursor:pointer;font-size:13px;font-weight:600;background:#edf2f7;color:#4a5568;border:1px solid #e2e8f0;border-bottom:none}
  .tab.active{background:#fff;color:#232f3e}
  .table-wrap{background:#fff;border-radius:12px;border:1px solid #e2e8f0;overflow-x:auto;max-height:70vh;overflow-y:auto}
  table{width:100%;border-collapse:collapse;font-size:13px}
  thead{position:sticky;top:0;z-index:2}
  th{background:#f7fafc;padding:12px 14px;text-align:left;font-weight:600;color:#4a5568;border-bottom:2px solid #e2e8f0;white-space:nowrap}
  td{padding:10px 14px;border-bottom:1px solid #f0f0f0}
  tr:hover td{background:#f7fafc}
  .num{text-align:right;font-variant-numeric:tabular-nums}
  .json-view{background:#1a1a2e;color:#a0e8af;padding:20px;border-radius:12px;overflow:auto;max-height:70vh;font-family:'SF Mono',Monaco,monospace;font-size:13px;line-height:1.6;white-space:pre-wrap}
  .search-box{padding:12px 16px;background:#f7fafc;border-bottom:1px solid #e2e8f0}
  .search-box input{width:100%;padding:8px 12px;border:1px solid #e2e8f0;border-radius:6px;font-size:13px}
  .empty-state{text-align:center;padding:80px 32px;color:#a0aec0}
  .empty-state h2{font-size:18px;color:#4a5568;margin-bottom:8px}
  .loader{display:inline-block;width:20px;height:20px;border:3px solid #e2e8f0;border-top-color:#ff9900;border-radius:50%;animation:spin .6s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}
  .progress-bar{width:100%;height:6px;background:#e2e8f0;border-radius:3px;overflow:hidden;margin-top:8px}
  .progress-fill{height:100%;background:#ff9900;transition:width .3s}
  .drop-zone{border:2px dashed #cbd5e0;border-radius:12px;padding:40px;text-align:center;cursor:pointer;transition:all .2s;margin:20px 32px}
  .drop-zone.dragover{border-color:#ff9900;background:#fff8ee}
  .drop-zone p{color:#718096;font-size:14px}
  .log-panel{background:#1a1a2e;color:#e2e8f0;padding:16px;border-radius:8px;margin-top:12px;font-family:monospace;font-size:12px;max-height:200px;overflow-y:auto}
  .log-entry{padding:2px 0;border-bottom:1px solid #2d3748}
  .log-tool{color:#ff9900;font-weight:600}
  .log-pass{color:#68d391}
  .log-fail{color:#fc8181}
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>Invoice Parser</h1>
    <p>Agentic extraction — upload a PDF, LLM extracts + reviews + deduplicates</p>
  </div>
  <div class="header-nav">
    <a href="/verify/ui/dashboard">Worker Dashboard</a>
    <a href="/docs">API Docs</a>
  </div>
</div>

<div class="drop-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
  <p><strong>Drop PDF here</strong> or click to select</p>
  <input type="file" id="fileInput" accept=".pdf" style="display:none" onchange="handleFile(this.files[0])">
  <div id="uploadProgress" style="display:none">
    <div class="loader" style="margin:12px auto"></div>
    <p id="progressText" style="margin-top:8px;color:#4a5568">Uploading...</p>
    <div class="progress-bar"><div class="progress-fill" id="progressFill" style="width:0%"></div></div>
  </div>
</div>

<div class="controls">
  <input type="text" id="jobIdInput" placeholder="Enter Job ID (e.g. a1b2c3d4)">
  <button class="btn btn-primary" onclick="pollJob()">Load Result</button>
  <span style="color:#a0aec0">|</span>
  <input type="file" id="jsonFileInput" accept=".json">
  <button class="btn btn-secondary" onclick="loadLocalJSON()">Load Local JSON</button>
</div>

<div class="status-bar" id="statusBar" style="display:none">
  <div class="stat"><div class="stat-value" id="statItems">-</div><div class="stat-label">Items</div></div>
  <div class="stat"><div class="stat-value" id="statScore">-</div><div class="stat-label">Quality</div></div>
  <div class="stat"><div class="stat-value" id="statStatus">-</div><div class="stat-label">Status</div></div>
  <div class="stat"><div class="stat-value" id="statTime">-</div><div class="stat-label">Time</div></div>
  <div class="stat"><div class="stat-value" id="statChunks">-</div><div class="stat-label">Chunks</div></div>
  <div class="stat"><div class="stat-value" id="statDedup">-</div><div class="stat-label">Dedup</div></div>
</div>

<div class="status-bar" id="invoiceHeader" style="display:none;background:#f7fafc;font-size:13px;gap:24px;padding:12px 32px">
  <div><strong>Invoice:</strong> <span id="hdrInvNo">-</span></div>
  <div><strong>Date:</strong> <span id="hdrDate">-</span></div>
  <div><strong>Currency:</strong> <span id="hdrCurrency">-</span></div>
  <div><strong>IncoTerms:</strong> <span id="hdrInco">-</span></div>
  <div><strong>Exporter:</strong> <span id="hdrExporter">-</span></div>
  <div><strong>Importer:</strong> <span id="hdrImporter">-</span></div>
  <div><strong>Classification:</strong> <span id="hdrClassification">-</span></div>
</div>

<div class="content">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;flex-wrap:wrap;gap:8px">
    <div class="tabs">
      <div class="tab active" onclick="switchTab('table')">Table View</div>
      <div class="tab" onclick="switchTab('json')">JSON View</div>
      <div class="tab" onclick="switchTab('log')">Trace</div>
    </div>
    <div style="display:flex;gap:8px">
      <button class="btn btn-export" onclick="exportCSV()">Export CSV</button>
      <button class="btn btn-secondary" onclick="exportJSON()">Export JSON</button>
    </div>
  </div>

  <div id="tableView">
    <div class="table-wrap">
      <div class="search-box">
        <input type="text" id="searchInput" placeholder="Search by Part No, Description, Country..." oninput="filterTable()">
      </div>
      <table>
        <thead><tr>
          <th>#</th><th>Part No</th><th>Item Code</th><th>Description</th><th class="num">Qty</th>
          <th>UOM</th><th class="num">Unit Price</th><th>RITC</th><th>Country</th>
        </tr></thead>
        <tbody id="tableBody"></tbody>
      </table>
    </div>
  </div>

  <div id="jsonView" style="display:none">
    <div class="json-view" id="jsonContent"></div>
  </div>

  <div id="logView" style="display:none">
    <div class="log-panel" id="logContent">No tool log available</div>
  </div>

  <div class="empty-state" id="emptyState">
    <h2>No data loaded</h2>
    <p>Drop a PDF above, enter a Job ID, or load a local JSON file</p>
  </div>
</div>

<script>
const API = '';
let currentData = null;
let currentItems = [];
let pollTimer = null;

// Drag and drop
const dz = document.getElementById('dropZone');
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('dragover'); });
dz.addEventListener('dragleave', () => dz.classList.remove('dragover'));
dz.addEventListener('drop', e => { e.preventDefault(); dz.classList.remove('dragover'); if(e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]); });

async function handleFile(file) {
  if (!file || !file.name.toLowerCase().endsWith('.pdf')) return alert('Please select a PDF file');
  const prog = document.getElementById('uploadProgress');
  const txt = document.getElementById('progressText');
  const fill = document.getElementById('progressFill');
  prog.style.display = 'block';
  txt.textContent = 'Uploading...';
  fill.style.width = '10%';

  try {
    const form = new FormData();
    form.append('file', file);
    fill.style.width = '30%';
    txt.textContent = 'Starting extraction...';

    const res = await fetch(API + '/parse/async', { method: 'POST', body: form });
    const json = await res.json();

    if (json.job_id) {
      document.getElementById('jobIdInput').value = json.job_id;
      fill.style.width = '50%';
      txt.textContent = 'Processing: ' + file.name + ' (Job: ' + json.job_id + ')';
      startPolling(json.job_id);
    } else {
      throw new Error(JSON.stringify(json));
    }
  } catch(e) {
    alert('Upload error: ' + e.message);
    prog.style.display = 'none';
  }
}

function startPolling(jobId) {
  if (pollTimer) clearInterval(pollTimer);
  const txt = document.getElementById('progressText');
  const fill = document.getElementById('progressFill');
  let dots = 0;

  pollTimer = setInterval(async () => {
    dots = (dots + 1) % 4;
    try {
      const res = await fetch(API + '/jobs/' + jobId);
      const job = await res.json();

      if (job.status === 'PROCESSING') {
        const p = job.progress;
        if (p && p.total_chunks > 0) {
          const pct = Math.min(85, 20 + Math.round((p.chunks_done / p.total_chunks) * 60));
          fill.style.width = pct + '%';
          let msg = p.current_phase;
          if (p.current_phase === 'Extracting') {
            msg = p.items_found > 0
              ? p.items_found + ' items found (' + p.chunks_done + '/' + p.total_chunks + ' chunks)'
              : 'Extracting chunk ' + (p.chunks_done + 1) + ' of ' + p.total_chunks;
          } else if (p.current_phase === 'Finalizing') {
            msg = p.items_found + ' items found — deduplicating & verifying';
          }
          txt.textContent = msg + '.'.repeat(dots + 1);
        } else {
          fill.style.width = '25%';
          txt.textContent = 'Analyzing document' + '.'.repeat(dots + 1);
        }
      } else if (job.status === 'AWAITING_VERIFICATION') {
        clearInterval(pollTimer);
        fill.style.width = '100%';
        fill.style.background = '#38a169';
        const items = job.result ? (job.result.total_items || job.result.LineItems?.length || 0) : 0;
        const score = job.result?.quality_score ? (job.result.quality_score * 100).toFixed(0) + '% quality' : '';
        const summary = items > 0 ? items + ' items extracted' + (score ? ' · ' + score : '') : 'Extraction complete';
        txt.innerHTML = summary + ' — <a href="/verify/ui/' + jobId + '" style="color:#fff;font-weight:600;text-decoration:underline">Review & Verify</a>';
        if (job.result) {
          currentData = job.result;
          currentItems = job.result.LineItems || [];
          renderAll();
        }
      } else if (job.status === 'COMPLETED') {
        clearInterval(pollTimer);
        fill.style.width = '100%';
        fill.style.background = '#38a169';
        const items = job.result ? (job.result.total_items || job.result.LineItems?.length || 0) : 0;
        txt.textContent = items > 0 ? items + ' items extracted — verified!' : 'Done!';
        document.getElementById('uploadProgress').style.display = 'none';
        if (job.result) {
          currentData = job.result;
          currentItems = job.result.LineItems || [];
          renderAll();
        } else {
          pollResult(jobId);
        }
      } else if (job.status === 'FAILED') {
        clearInterval(pollTimer);
        fill.style.width = '100%';
        fill.style.background = '#e53e3e';
        txt.textContent = 'Failed: ' + (job.result?.error || 'Unknown error');
      }
    } catch(e) {
      console.error('Poll error:', e);
    }
  }, 3000);
}

async function pollJob() {
  const jobId = document.getElementById('jobIdInput').value.trim();
  if (!jobId) return alert('Enter a Job ID');
  try {
    const res = await fetch(API + '/jobs/' + jobId);
    if (res.status === 404) return alert('Job not found: ' + jobId);
    const job = await res.json();
    if (job.status === 'AWAITING_VERIFICATION') {
      currentData = job.result;
      currentItems = (job.result && job.result.LineItems) || [];
      if (currentData) renderAll();
      const prog = document.getElementById('uploadProgress');
      prog.style.display = 'block';
      const fill = document.getElementById('progressFill');
      fill.style.width = '90%';
      fill.style.background = '#d69e2e';
      document.getElementById('progressText').innerHTML = 'Awaiting verification — <a href="/verify/ui/' + jobId + '" style="color:#ff9900;font-weight:600">Open Verification</a>';
    } else if (job.status === 'COMPLETED' && job.result) {
      currentData = job.result;
      currentItems = job.result.LineItems || [];
      renderAll();
    } else if (job.status === 'PROCESSING' || job.status === 'PENDING') {
      document.getElementById('uploadProgress').style.display = 'block';
      document.getElementById('progressText').textContent = 'Job ' + jobId + ' still processing...';
      startPolling(jobId);
    } else {
      await pollResult(jobId);
    }
  } catch(e) { alert('Error: ' + e.message); }
}

async function pollResult(jobId) {
  try {
    const res = await fetch(API + '/result/' + jobId);
    if (res.status === 202) return alert('Job still processing. Try again in a moment.');
    const data = await res.json();
    currentData = data;
    currentItems = data.LineItems || [];
    renderAll();
  } catch(e) { alert('Error: ' + e.message); }
}

function loadLocalJSON() {
  const file = document.getElementById('jsonFileInput').files[0];
  if (!file) return alert('Select a JSON file');
  const reader = new FileReader();
  reader.onload = function(e) {
    try {
      const data = JSON.parse(e.target.result);
      currentData = data;
      currentItems = data.LineItems || [];
      renderAll();
    } catch(err) { alert('Invalid JSON: ' + err.message); }
  };
  reader.readAsText(file);
}

function renderAll() {
  document.getElementById('emptyState').style.display = 'none';
  document.getElementById('statusBar').style.display = 'flex';
  renderStats();
  renderTable(currentItems);
  renderJSON();
  renderLog();
}

function renderStats() {
  const d = currentData;
  document.getElementById('statItems').textContent = d.total_items || currentItems.length || '-';

  const score = d.quality_score;
  const el = document.getElementById('statScore');
  if (score != null && score > 0) {
    const pct = (score * 100).toFixed(0);
    const cls = score >= 0.8 ? 'badge-green' : score >= 0.7 ? 'badge-yellow' : 'badge-red';
    const label = score >= 0.8 ? 'GOOD' : score >= 0.7 ? 'OK' : 'LOW';
    el.innerHTML = pct + '% <span class="badge ' + cls + '">' + label + '</span>';
  } else { el.textContent = '-'; }

  const statusEl = document.getElementById('statStatus');
  const success = d.success;
  statusEl.innerHTML = '<span class="badge ' + (success ? 'badge-green' : 'badge-red') + '">' + (success ? 'SUCCESS' : 'FAILED') + '</span>';

  document.getElementById('statTime').textContent = d.elapsed_seconds ? d.elapsed_seconds + 's' : '-';
  document.getElementById('statChunks').textContent = d.chunks_processed && d.total_chunks ? d.chunks_processed + '/' + d.total_chunks : '-';

  const dd = d.dedup_summary;
  if (dd && dd.before) {
    document.getElementById('statDedup').textContent = dd.before + ' > ' + dd.after;
  } else { document.getElementById('statDedup').textContent = '-'; }

  // Invoice header
  const hdr = document.getElementById('invoiceHeader');
  if (d.InvoiceNo || d.Date || d.Exporter || d.Classification) {
    hdr.style.display = 'flex';
    document.getElementById('hdrInvNo').textContent = d.InvoiceNo || '-';
    document.getElementById('hdrDate').textContent = d.Date || '-';
    document.getElementById('hdrCurrency').textContent = d.InvoiceCurrency || '-';
    document.getElementById('hdrInco').textContent = d.IncoTerms || '-';
    document.getElementById('hdrExporter').textContent = d.Exporter?.Name || d.Exporter || '-';
    document.getElementById('hdrImporter').textContent = d.Importer?.Name || d.Importer || '-';
    document.getElementById('hdrClassification').textContent = d.Classification || '-';
  }
}

function renderTable(items) {
  const tbody = document.getElementById('tableBody');
  if (!items || !items.length) {
    tbody.innerHTML = '<tr><td colspan="9" style="text-align:center;padding:40px;color:#a0aec0">No line items</td></tr>';
    return;
  }
  tbody.innerHTML = items.map((item, i) =>
    '<tr><td>' + (item.ItemNo || i+1) + '</td>'
    + '<td><strong>' + esc(item.PartNo || '') + '</strong></td>'
    + '<td>' + esc(item.ItemCode || '') + '</td>'
    + '<td>' + esc(item.ItemDescription || '') + '</td>'
    + '<td class="num">' + esc(item.Quantity || '') + '</td>'
    + '<td>' + esc(item.UnitOfQty || '') + '</td>'
    + '<td class="num">' + esc(item.UnitPrice || '') + '</td>'
    + '<td>' + esc(item.RITC || '') + '</td>'
    + '<td>' + esc(item.CountryOfOrigin || '') + '</td></tr>'
  ).join('');
}

function renderJSON() {
  document.getElementById('jsonContent').textContent = JSON.stringify(currentData, null, 2);
}

async function renderLog() {
  const el = document.getElementById('logContent');

  // Try to load rich trace from API first
  const jobId = document.getElementById('jobIdInput').value.trim();
  if (jobId) {
    try {
      const res = await fetch(API + '/jobs/' + jobId + '/trace');
      if (res.ok) {
        const trace = await res.json();
        if (trace.steps && trace.steps.length) {
          const cost = trace.estimated_cost != null ? '$' + trace.estimated_cost.toFixed(3) : '';
          const elapsed = trace.elapsed != null ? trace.elapsed + 's' : '';
          const tok = trace.total_tokens ? (trace.total_tokens.total/1000).toFixed(0) + 'K tok' : '';
          let html = '<div style="margin-bottom:8px;color:#ff9900;font-weight:600">JOB ' + (trace.job_id||jobId) + '  ' + (trace.filename||'') + '</div>';
          html += '<div style="margin-bottom:10px;color:#a0aec0;font-size:11px">' + [elapsed, tok, cost].filter(Boolean).join('  |  ') + '</div>';
          html += trace.steps.map(s => {
            const label = s.tool.toUpperCase().replace('_',' ');
            const chunk = s.chunk != null ? '[' + s.chunk + ']' : '';
            const result = s.result || s.error || '';
            const dur = s.duration ? s.duration + 's' : '';
            const stok = (s.tokens_in||0) + (s.tokens_out||0);
            const tokStr = stok > 0 ? (stok/1000).toFixed(0) + 'K' : '';
            const isErr = !!s.error;
            const isPass = result.toUpperCase().includes('PASS');
            const isFail = result.toUpperCase().includes('FAIL');
            const cls = isErr ? 'log-fail' : isPass ? 'log-pass' : isFail ? 'log-fail' : 'log-tool';
            return '<div class="log-entry" style="display:flex;gap:8px;align-items:baseline">'
              + '<span class="' + cls + '" style="min-width:120px;display:inline-block">' + label + chunk + '</span>'
              + '<span style="flex:1">' + esc(result) + '</span>'
              + (tokStr ? '<span style="color:#718096;min-width:50px;text-align:right">' + tokStr + '</span>' : '')
              + (dur ? '<span style="color:#718096;min-width:45px;text-align:right">' + dur + '</span>' : '')
              + '</div>';
          }).join('');
          el.innerHTML = html;
          return;
        }
      }
    } catch(e) { /* fall through to old format */ }
  }

  // Fallback: use tool_log from result
  const log = currentData?.tool_log;
  if (!log || !log.length) { el.textContent = 'No trace available'; return; }
  el.innerHTML = log.map(e => {
    let cls = 'log-tool';
    let extra = '';
    if (e.tool === 'review_chunk') {
      cls = e.passed ? 'log-pass' : 'log-fail';
      extra = e.passed ? ' PASSED' : ' FAILED (' + (e.critical||0) + ' critical)';
    }
    if (e.tool === 'extract_chunk') extra = ' -> ' + (e.items_found||0) + ' items' + (e.truncated ? ' (TRUNCATED)' : '');
    if (e.tool === 'deduplicate') extra = ' -> ' + e.before + ' > ' + e.after;
    if (e.tool === 'verify_final') { cls = e.passed ? 'log-pass' : 'log-fail'; extra = e.passed ? ' PASS' : ' FAIL'; }
    if (e.tool === 're_extract') extra = ' -> +' + (e.additional||0) + ' new, ' + (e.corrected||0) + ' fixed';
    if (e.tool === 'split_pdf') extra = ' -> ' + e.chunks + ' chunks (' + e.total_pages + ' pages)';
    const ci = e.chunk_index != null ? '[' + e.chunk_index + ']' : '';
    return '<div class="log-entry"><span class="' + cls + '">' + e.tool + ci + '</span>' + extra + '</div>';
  }).join('');
}

function filterTable() {
  const q = document.getElementById('searchInput').value.toLowerCase();
  if (!q) { renderTable(currentItems); return; }
  renderTable(currentItems.filter(item =>
    (item.PartNo||'').toLowerCase().includes(q) ||
    (item.ItemDescription||'').toLowerCase().includes(q) ||
    (item.CountryOfOrigin||'').toLowerCase().includes(q)
  ));
}

function switchTab(tab) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('tableView').style.display = 'none';
  document.getElementById('jsonView').style.display = 'none';
  document.getElementById('logView').style.display = 'none';
  if (tab === 'table') { document.getElementById('tableView').style.display = ''; document.querySelectorAll('.tab')[0].classList.add('active'); }
  else if (tab === 'json') { document.getElementById('jsonView').style.display = ''; document.querySelectorAll('.tab')[1].classList.add('active'); }
  else { document.getElementById('logView').style.display = ''; document.querySelectorAll('.tab')[2].classList.add('active'); }
}

function exportCSV() {
  if (!currentItems.length) return alert('No data');
  const h = ['ItemNo','PartNo','ItemCode','ItemDescription','Quantity','UnitOfQty','UnitPrice','RITC','CountryOfOrigin'];
  const rows = currentItems.map(item => h.map(k => '"' + String(item[k]||'').replace(/"/g,'""') + '"').join(','));
  dl([h.join(','), ...rows].join('\\n'), 'invoice_items.csv', 'text/csv');
}
function exportJSON() {
  if (!currentData) return alert('No data');
  dl(JSON.stringify(currentData,null,2), 'invoice_result.json', 'application/json');
}
function dl(c,f,t) { const b=new Blob([c],{type:t}); const u=URL.createObjectURL(b); const a=document.createElement('a'); a.href=u; a.download=f; a.click(); URL.revokeObjectURL(u); }

function esc(s) { const d=document.createElement('div'); d.textContent=s; return d.innerHTML; }
function fmt(v) { return v!=null ? Number(v).toLocaleString() : ''; }
function fmtP(v) { return v!=null ? Number(v).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:4}) : ''; }
</script>
</body>
</html>"""


# ─── Background job runner ────────────────────────────────

def _run_extraction(job_id: str, pdf_path: str, output_path: str):
    jobs[job_id].status = JobState.PROCESSING
    try:
        import tools
        tools.reset_accumulator()
        tools.init_tracer(job_id, jobs[job_id].filename)
        result = agent.run(pdf_path, output_path)
        if result and result.get("success"):
            jobs[job_id].result = ExtractionResult(**result)
            jobs[job_id].status = JobState.AWAITING_VERIFICATION
            verification_mgr.create_verification(
                job_id, result, format_key=result.get('format_key'),
            )
            logger.info(f"Job {job_id}: extraction done, awaiting worker verification")
        else:
            jobs[job_id].status = JobState.FAILED
            jobs[job_id].result = ExtractionResult(
                success=False,
                filename=jobs[job_id].filename,
                error=result.get("error", "Unknown error") if result else "No result returned",
            )
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        jobs[job_id].status = JobState.FAILED
        jobs[job_id].result = ExtractionResult(
            success=False,
            filename=jobs[job_id].filename,
            error=str(e),
        )
    jobs[job_id].completed_at = time.time()


def _save_upload(file: UploadFile) -> tuple[str, str, str]:
    job_id = uuid.uuid4().hex[:8]
    ext = Path(file.filename or "upload.pdf").suffix or ".pdf"
    pdf_path = str(UPLOAD_DIR / f"{job_id}{ext}")
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    output_path = str(RESULT_DIR / f"{job_id}.json")
    # Persist PDF to S3 so worker can view source after container restart
    try:
        import storage
        with open(pdf_path, "rb") as f:
            storage.put_blob(f"uploads/{job_id}{ext}", f.read())
        logger.info(f"PDF saved to storage: uploads/{job_id}{ext}")
    except Exception as e:
        logger.warning(f"PDF storage save failed (non-blocking): {e}")
    return job_id, pdf_path, output_path


# ─── Endpoints ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the Invoice Parser UI."""
    return VIEWER_HTML


@app.post("/parse", response_model=ExtractionResult)
async def parse_sync(file: UploadFile = File(...)):
    """Upload a PDF and get extraction results synchronously.
    Blocks until extraction is complete — use /parse/async for large files.
    """
    job_id, pdf_path, output_path = _save_upload(file)
    result = agent.run(pdf_path, output_path)
    if not result:
        raise HTTPException(status_code=500, detail="Extraction returned no result")
    return ExtractionResult(**result)


@app.post("/parse/async", response_model=ParseResponse)
async def parse_async(file: UploadFile = File(...)):
    """Upload a PDF and start extraction in the background.
    Returns a job_id — poll GET /jobs/{job_id} for status.
    """
    job_id, pdf_path, output_path = _save_upload(file)

    jobs[job_id] = JobStatus(
        job_id=job_id,
        status=JobState.PENDING,
        filename=file.filename or "upload.pdf",
        created_at=time.time(),
    )

    executor.submit(_run_extraction, job_id, pdf_path, output_path)
    return ParseResponse(
        job_id=job_id,
        status=JobState.PENDING,
        filename=file.filename or "upload.pdf",
    )


@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str):
    """Get the status of an extraction job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    job = jobs[job_id]
    if job.status == JobState.PROCESSING:
        import tools
        acc = tools.accumulator
        chunks_done = len(acc.processed_indices)
        total = acc.total_chunks
        items = len(acc.items)
        if total > 0 and chunks_done >= total:
            phase = "Finalizing"
        elif chunks_done > 0:
            phase = "Extracting"
        elif total > 0:
            phase = "Splitting"
        else:
            phase = "Starting"
        job.progress = JobProgress(
            items_found=items,
            chunks_done=chunks_done,
            total_chunks=total,
            current_phase=phase,
        )
    return job


@app.get("/result/{job_id}")
async def get_result(job_id: str):
    """Get the final result for a job.
    Returns worker-verified result if verification is done,
    otherwise returns agent result with verification_status.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    job = jobs[job_id]
    if job.status == JobState.PROCESSING:
        raise HTTPException(status_code=202, detail="Job still processing")
    if job.status == JobState.PENDING:
        raise HTTPException(status_code=202, detail="Job pending")
    if job.result is None:
        raise HTTPException(status_code=500, detail="No result available")

    v = verification_mgr.get_verification(job_id)
    result_dict = job.result.model_dump()

    if v and v.get('status') == 'verified' and v.get('worker_result'):
        worker = v['worker_result']
        result_dict.update({
            k: worker.get(k) for k in [
                'InvoiceNo', 'Date', 'InvoiceCurrency', 'FreightTerms',
                'IncoTerms', 'TermsOfPayment', 'Exporter', 'Importer',
            ] if worker.get(k) is not None
        })
        if worker.get('LineItems'):
            result_dict['LineItems'] = worker['LineItems']
            result_dict['total_items'] = len(worker['LineItems'])
        result_dict['verification_status'] = 'verified'
        result_dict['verification_accuracy'] = v.get('accuracy', {}).get('field_accuracy', 0)
    elif job.status == JobState.AWAITING_VERIFICATION:
        result_dict['verification_status'] = 'awaiting_verification'
    else:
        result_dict['verification_status'] = 'none'

    return result_dict


@app.get("/result/{job_id}/download")
async def download_result(job_id: str):
    """Download the raw JSON result file."""
    result_path = RESULT_DIR / f"{job_id}.json"
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Result file not found")
    return FileResponse(str(result_path), media_type="application/json", filename=f"{job_id}_extracted.json")


@app.get("/jobs/{job_id}/trace")
async def get_trace(job_id: str):
    """Get the step-by-step execution trace for a job."""
    trace = storage.get_doc('traces', job_id)
    if trace:
        return trace
    if job_id in jobs and jobs[job_id].result:
        return {"job_id": job_id, "steps": jobs[job_id].result.tool_log}
    raise HTTPException(status_code=404, detail="Trace not found")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check."""
    active = sum(1 for j in jobs.values() if j.status in (JobState.PENDING, JobState.PROCESSING, JobState.AWAITING_VERIFICATION))
    return HealthResponse(status="ok", active_jobs=active)


# ─── Verification UI ─────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Verification Dashboard</title>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#f5f7fa;color:#1a1a2e}
  .header{background:linear-gradient(135deg,#232f3e 0%,#1a1a2e 100%);color:#fff;padding:24px 32px;display:flex;justify-content:space-between;align-items:center}
  .header h1{font-size:22px;font-weight:600}
  .header p{font-size:13px;color:#a0aec0;margin-top:4px}
  .header a{color:#ff9900;text-decoration:none;font-size:13px}
  .header a:hover{text-decoration:underline}
  .summary-bar{display:flex;gap:20px;padding:20px 32px;background:#fff;border-bottom:1px solid #e2e8f0;flex-wrap:wrap}
  .stat{text-align:center}
  .stat-value{font-size:28px;font-weight:700;color:#232f3e}
  .stat-label{font-size:11px;color:#718096;text-transform:uppercase;letter-spacing:.5px}
  .content{padding:20px 32px}
  .table-wrap{background:#fff;border-radius:12px;border:1px solid #e2e8f0;overflow-x:auto}
  table{width:100%;border-collapse:collapse;font-size:13px}
  thead{position:sticky;top:0;z-index:2}
  th{background:#f7fafc;padding:12px 14px;text-align:left;font-weight:600;color:#4a5568;border-bottom:2px solid #e2e8f0;white-space:nowrap}
  td{padding:10px 14px;border-bottom:1px solid #f0f0f0}
  tr.clickable{cursor:pointer}
  tr.clickable:hover td{background:#f7fafc}
  .badge{display:inline-block;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600}
  .badge-green{background:#c6f6d5;color:#276749}
  .badge-yellow{background:#fefcbf;color:#975a16}
  .badge-red{background:#fed7d7;color:#9b2c2c}
  .empty-state{text-align:center;padding:80px 32px;color:#a0aec0}
  .empty-state h2{font-size:18px;color:#4a5568;margin-bottom:8px}
  .loader{display:inline-block;width:20px;height:20px;border:3px solid #e2e8f0;border-top-color:#ff9900;border-radius:50%;animation:spin .6s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<div class="header">
  <div>
    <h1>Verification Dashboard</h1>
    <p>Review pending invoice extractions</p>
  </div>
  <a href="/">Back to Parser</a>
</div>

<div class="summary-bar" id="summaryBar">
  <div class="stat"><div class="stat-value" id="totalPending">-</div><div class="stat-label">Pending</div></div>
  <div class="stat"><div class="stat-value" id="countAuto">-</div><div class="stat-label">Auto Approve</div></div>
  <div class="stat"><div class="stat-value" id="countQuick">-</div><div class="stat-label">Quick Review</div></div>
  <div class="stat"><div class="stat-value" id="countFull">-</div><div class="stat-label">Full Review</div></div>
</div>

<div class="content">
  <div class="table-wrap">
    <table>
      <thead><tr>
        <th>Job ID</th><th>Filename</th><th>Confidence</th><th>Review Level</th><th>Created</th>
      </tr></thead>
      <tbody id="tableBody">
        <tr><td colspan="5" style="text-align:center;padding:40px"><div class="loader"></div></td></tr>
      </tbody>
    </table>
  </div>
  <div class="empty-state" id="emptyState" style="display:none">
    <h2>No pending verifications</h2>
    <p>All caught up. Upload a new invoice to create one.</p>
  </div>
</div>

<script>
const API = '';

async function load() {
  try {
    const res = await fetch(API + '/verify/pending/list');
    const pending = await res.json();
    render(pending);
  } catch(e) {
    document.getElementById('tableBody').innerHTML =
      '<tr><td colspan="5" style="text-align:center;padding:40px;color:#e53e3e">Failed to load: ' + e.message + '</td></tr>';
  }
}

function render(items) {
  const tbody = document.getElementById('tableBody');
  if (!items || !items.length) {
    tbody.innerHTML = '';
    document.getElementById('emptyState').style.display = '';
    document.getElementById('totalPending').textContent = '0';
    document.getElementById('countAuto').textContent = '0';
    document.getElementById('countQuick').textContent = '0';
    document.getElementById('countFull').textContent = '0';
    return;
  }

  document.getElementById('totalPending').textContent = items.length;
  document.getElementById('countAuto').textContent = items.filter(i => i.review_level === 'auto_approve').length;
  document.getElementById('countQuick').textContent = items.filter(i => i.review_level === 'quick_review').length;
  document.getElementById('countFull').textContent = items.filter(i => i.review_level === 'full_review').length;

  tbody.innerHTML = items.map(item => {
    const lvl = item.review_level || 'full_review';
    const cls = lvl === 'auto_approve' ? 'badge-green' : lvl === 'quick_review' ? 'badge-yellow' : 'badge-red';
    const label = lvl === 'auto_approve' ? 'Auto Approve' : lvl === 'quick_review' ? 'Quick Review' : 'Full Review';
    const score = item.confidence != null ? (item.confidence * 100).toFixed(0) + '%' : '-';
    const created = item.created_at ? new Date(item.created_at * 1000).toLocaleString() : '-';
    return '<tr class="clickable" onclick="window.location.href=\\'/verify/ui/' + item.job_id + '\\'">'
      + '<td><strong>' + esc(item.job_id) + '</strong></td>'
      + '<td>' + esc(item.filename || '-') + '</td>'
      + '<td>' + score + '</td>'
      + '<td><span class="badge ' + cls + '">' + label + '</span></td>'
      + '<td>' + created + '</td>'
      + '</tr>';
  }).join('');
}

function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

load();
</script>
</body>
</html>"""


VERIFY_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Verify Invoice</title>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#f5f7fa;color:#1a1a2e}
  .header{background:linear-gradient(135deg,#232f3e 0%,#1a1a2e 100%);color:#fff;padding:16px 32px;display:flex;justify-content:space-between;align-items:center}
  .header h1{font-size:20px;font-weight:600}
  .header-meta{font-size:12px;color:#a0aec0;display:flex;gap:16px;align-items:center}
  .header a{color:#ff9900;text-decoration:none;font-size:13px}
  .header a:hover{text-decoration:underline}
  .badge{display:inline-block;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600}
  .badge-green{background:#c6f6d5;color:#276749}
  .badge-yellow{background:#fefcbf;color:#975a16}
  .badge-red{background:#fed7d7;color:#9b2c2c}
  .badge-blue{background:#bee3f8;color:#2a4365}

  .accuracy-bar{display:flex;gap:20px;padding:12px 32px;background:#fff;border-bottom:1px solid #e2e8f0;align-items:center;flex-wrap:wrap}
  .accuracy-bar .stat-value{font-size:22px;font-weight:700;color:#232f3e}
  .accuracy-bar .stat-label{font-size:11px;color:#718096;text-transform:uppercase}
  .accuracy-ring{width:52px;height:52px;position:relative}
  .accuracy-ring svg{transform:rotate(-90deg)}
  .accuracy-ring .pct{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-size:13px;font-weight:700}

  .main-layout{display:flex;height:calc(100vh - 130px)}
  .edit-panel{flex:1;overflow-y:auto;padding:20px 24px}
  .source-panel{width:45%;border-left:1px solid #e2e8f0;display:flex;flex-direction:column;background:#fff}
  .source-panel iframe,.source-panel embed,.source-panel object{flex:1;width:100%;border:none}
  .source-panel .source-header{padding:10px 16px;font-size:13px;font-weight:600;color:#4a5568;border-bottom:1px solid #e2e8f0;background:#f7fafc}
  .source-panel .no-pdf{flex:1;display:flex;align-items:center;justify-content:center;color:#a0aec0;font-size:14px}

  .section{margin-bottom:20px}
  .section-title{font-size:14px;font-weight:700;color:#4a5568;margin-bottom:10px;text-transform:uppercase;letter-spacing:.5px}

  .field-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:10px}
  .field-item{position:relative}
  .field-item label{display:block;font-size:11px;color:#718096;margin-bottom:3px;font-weight:600;text-transform:uppercase}
  .field-item input,.field-item textarea{width:100%;padding:8px 10px;border:2px solid #c6f6d5;border-radius:6px;font-size:13px;background:#fff;transition:border-color .2s}
  .field-item input:focus,.field-item textarea:focus{outline:none;border-color:#4299e1}
  .field-item.corrected input,.field-item.corrected textarea{border-color:#fed7d7;background:#fff5f5}
  .field-item .toggle-btn{position:absolute;top:0;right:0;font-size:10px;cursor:pointer;padding:2px 6px;border-radius:4px;border:1px solid #e2e8f0;background:#fff;color:#718096}
  .field-item .toggle-btn:hover{background:#f7fafc}

  .table-section{background:#fff;border-radius:12px;border:1px solid #e2e8f0;overflow:hidden}
  .table-toolbar{display:flex;justify-content:space-between;align-items:center;padding:10px 14px;border-bottom:1px solid #e2e8f0;background:#f7fafc;flex-wrap:wrap;gap:8px}
  .table-toolbar input{padding:6px 10px;border:1px solid #cbd5e0;border-radius:6px;font-size:13px;min-width:200px}
  .table-scroll{overflow:auto;max-height:55vh}
  table{width:100%;border-collapse:collapse;font-size:12px}
  thead{position:sticky;top:0;z-index:2}
  th{background:#f7fafc;padding:8px 10px;text-align:left;font-weight:600;color:#4a5568;border-bottom:2px solid #e2e8f0;white-space:nowrap}
  td{padding:4px 6px;border-bottom:1px solid #f0f0f0;vertical-align:middle}
  td input{width:100%;padding:5px 6px;border:2px solid #c6f6d5;border-radius:4px;font-size:12px;background:#fff;transition:border-color .2s}
  td input:focus{outline:none;border-color:#4299e1}
  td input.corrected{border-color:#fed7d7;background:#fff5f5}
  td.row-actions{white-space:nowrap;text-align:center}
  .row-btn{border:none;background:none;cursor:pointer;font-size:14px;padding:2px 4px;border-radius:4px}
  .row-btn:hover{background:#edf2f7}
  .row-btn.del{color:#e53e3e}
  .row-deleted td{opacity:.35;text-decoration:line-through}
  .row-added{background:#f0fff4}

  .actions-bar{position:sticky;bottom:0;background:#fff;border-top:1px solid #e2e8f0;padding:14px 24px;display:flex;gap:10px;align-items:center;flex-wrap:wrap;z-index:10}
  .btn{padding:10px 20px;border:none;border-radius:8px;font-size:14px;font-weight:600;cursor:pointer;transition:all .2s}
  .btn-primary{background:#ff9900;color:#fff}
  .btn-primary:hover{background:#ec8f00}
  .btn-success{background:#38a169;color:#fff}
  .btn-success:hover{background:#2f855a}
  .btn-secondary{background:#edf2f7;color:#4a5568}
  .btn-secondary:hover{background:#e2e8f0}
  .btn-danger{background:#e53e3e;color:#fff}
  .btn-danger:hover{background:#c53030}
  .comment-input{flex:1;min-width:200px;padding:8px 12px;border:1px solid #cbd5e0;border-radius:8px;font-size:13px}

  .toast{position:fixed;top:20px;right:20px;padding:12px 20px;border-radius:8px;color:#fff;font-weight:600;font-size:14px;z-index:100;opacity:0;transition:opacity .3s}
  .toast.show{opacity:1}
  .toast-success{background:#38a169}
  .toast-error{background:#e53e3e}
  .loader{display:inline-block;width:20px;height:20px;border:3px solid #e2e8f0;border-top-color:#ff9900;border-radius:50%;animation:spin .6s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>Verify: <span id="jobIdLabel">-</span></h1>
    <div class="header-meta">
      <span id="fileLabel">-</span>
      <span id="reviewBadge"></span>
    </div>
  </div>
  <div style="display:flex;gap:12px;align-items:center">
    <a href="/verify/ui/dashboard">Dashboard</a>
    <a href="/">Parser</a>
  </div>
</div>

<div class="accuracy-bar" id="accuracyBar">
  <div class="accuracy-ring" id="accuracyRing">
    <svg width="52" height="52"><circle cx="26" cy="26" r="22" fill="none" stroke="#e2e8f0" stroke-width="4"/><circle id="ringCircle" cx="26" cy="26" r="22" fill="none" stroke="#38a169" stroke-width="4" stroke-dasharray="138.2" stroke-dashoffset="0" stroke-linecap="round"/></svg>
    <div class="pct" id="ringPct">-</div>
  </div>
  <div><div class="accuracy-bar stat-value" id="accValue">-</div><div class="accuracy-bar stat-label" id="accLabel">AI Quality</div></div>
  <div><div class="accuracy-bar stat-value" id="totalFieldsVal">-</div><div class="accuracy-bar stat-label">Total Fields</div></div>
  <div><div class="accuracy-bar stat-value" id="correctFieldsVal">-</div><div class="accuracy-bar stat-label">Unchanged</div></div>
  <div><div class="accuracy-bar stat-value" id="correctedFieldsVal">-</div><div class="accuracy-bar stat-label">Corrected</div></div>
</div>

<div id="agentIssuesPanel" style="display:none;margin:0 32px;padding:14px 20px;background:#fff8f0;border:1px solid #f59e0b;border-radius:8px">
  <div style="font-weight:700;color:#92400e;font-size:13px;margin-bottom:8px">AGENT FLAGGED ISSUES — Check These First</div>
  <div id="agentIssuesList" style="font-size:13px;color:#78350f"></div>
</div>

<div class="main-layout">
  <div class="edit-panel" id="editPanel">
    <!-- Header Fields -->
    <div class="section">
      <div class="section-title">Invoice Header</div>
      <div class="field-grid" id="headerFields"></div>
    </div>

    <!-- Exporter / Importer -->
    <div class="section">
      <div class="section-title">Parties</div>
      <div class="field-grid" id="partyFields"></div>
    </div>

    <!-- Line Items -->
    <div class="section">
      <div class="section-title">Line Items (<span id="itemCount">0</span>)</div>
      <div class="table-section">
        <div class="table-toolbar">
          <input type="text" id="itemSearch" placeholder="Search items..." oninput="filterItems()">
          <div style="display:flex;gap:6px">
            <button class="btn btn-secondary" style="padding:6px 12px;font-size:12px" onclick="addRow()">+ Add Row</button>
          </div>
        </div>
        <div class="table-scroll">
          <table>
            <thead><tr>
              <th style="width:40px">#</th>
              <th>PartNo</th><th>ItemCode</th><th style="min-width:200px">Description</th>
              <th style="width:80px">Qty</th><th style="width:70px">UOM</th>
              <th style="width:90px">UnitPrice</th><th>RITC</th><th>Country</th>
              <th style="width:70px">Actions</th>
            </tr></thead>
            <tbody id="itemsBody"></tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- Actions -->
    <div class="actions-bar">
      <button class="btn btn-success" onclick="approveAll()">Approve All</button>
      <button class="btn btn-primary" onclick="submitCorrections()">Submit Corrections</button>
      <input type="text" class="comment-input" id="commentInput" placeholder="Add comment or feedback...">
      <button class="btn btn-secondary" onclick="sendComment()">Send Comment</button>
    </div>
  </div>

  <div class="source-panel" id="sourcePanel">
    <div class="source-header">Source Document</div>
    <div class="no-pdf" id="noPdf">Loading...</div>
  </div>
</div>

<div class="toast" id="toast"></div>

<script>
const API = '';
const JOB_ID = location.pathname.split('/').pop();

let agentResult = null;
let headerFieldStates = {};   // field -> { value, corrected }
let partyFieldStates = {};    // 'Exporter.Name' -> { value, corrected }
let itemRows = [];            // [ { fields: {PartNo:..., ...}, deleted: false, added: false, corrected: {} } ]
let originalItemCount = 0;

// ── Load data ──────────────────────────────────────────
async function loadData() {
  try {
    const res = await fetch(API + '/verify/data/' + JOB_ID);
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    agentResult = data.agent_result;
    initFields();
    loadPdf(data.pdf_url);
    document.getElementById('jobIdLabel').textContent = JOB_ID;
    document.getElementById('fileLabel').textContent = agentResult.filename || '-';
    const lvl = data.review_level || 'full_review';
    const cls = lvl === 'auto_approve' ? 'badge-green' : lvl === 'quick_review' ? 'badge-yellow' : 'badge-red';
    const label = lvl === 'auto_approve' ? 'Auto Approve' : lvl === 'quick_review' ? 'Quick Review' : 'Full Review';
    document.getElementById('reviewBadge').innerHTML = '<span class="badge ' + cls + '">' + label + '</span>';

    // Show AI quality score (not accuracy — accuracy is computed after worker submits)
    const qs = agentResult.quality_score || 0;
    const qPct = Math.round(qs * 100);
    document.getElementById('ringPct').textContent = qPct + '%';
    document.getElementById('accValue').textContent = qPct + '%';
    document.getElementById('accLabel').textContent = 'AI Quality';
    const circumference = 138.2;
    document.getElementById('ringCircle').setAttribute('stroke-dashoffset', circumference * (1 - qs));
    document.getElementById('ringCircle').setAttribute('stroke', qPct >= 80 ? '#38a169' : qPct >= 50 ? '#d69e2e' : '#e53e3e');
    document.getElementById('totalFieldsVal').textContent = '-';
    document.getElementById('correctFieldsVal').textContent = '-';
    document.getElementById('correctedFieldsVal').textContent = '0';

    // Show agent issues for worker
    const alerts = data.worker_alerts || [];
    if (alerts.length > 0) {
      const panel = document.getElementById('agentIssuesPanel');
      panel.style.display = 'block';
      document.getElementById('agentIssuesList').innerHTML = alerts.map(function(a) {
        var icon = '⚠️';
        if (a.indexOf('WRONG_VALUE') >= 0) icon = '🔧';
        else if (a.indexOf('MISSING') >= 0) icon = '❌';
        else if (a.indexOf('CLASSIFICATION') >= 0) icon = '📄';
        else if (a.indexOf('quality') >= 0 || a.indexOf('Quality') >= 0) icon = '📊';
        return '<div style="padding:6px 0;border-bottom:1px solid #fde68a;line-height:1.5">' + icon + ' ' + a + '</div>';
      }).join('');
    }
  } catch(e) {
    document.getElementById('editPanel').innerHTML =
      '<div style="text-align:center;padding:80px;color:#e53e3e"><h2>Failed to load</h2><p>' + e.message + '</p><p style="margin-top:12px"><a href="/verify/ui/dashboard">Back to Dashboard</a></p></div>';
  }
}

function initFields() {
  const r = agentResult;
  // Header scalars
  const hdrFields = ['InvoiceNo','Date','InvoiceCurrency','FreightTerms','IncoTerms','TermsOfPayment','Classification'];
  hdrFields.forEach(f => {
    headerFieldStates[f] = { value: r[f] || '', corrected: false };
  });
  renderHeaderFields();

  // Parties
  ['Exporter','Importer'].forEach(party => {
    const obj = r[party] || {};
    const dict = typeof obj === 'object' ? obj : { Name: String(obj), Address: '' };
    ['Name','Address'].forEach(sub => {
      const key = party + '.' + sub;
      partyFieldStates[key] = { value: dict[sub] || '', corrected: false };
    });
  });
  renderPartyFields();

  // Line items
  const items = r.LineItems || [];
  originalItemCount = items.length;
  itemRows = items.map((item, i) => ({
    fields: {
      PartNo: item.PartNo || '',
      ItemCode: item.ItemCode || '',
      ItemDescription: item.ItemDescription || '',
      Quantity: item.Quantity || '',
      UnitOfQty: item.UnitOfQty || '',
      UnitPrice: item.UnitPrice || '',
      RITC: item.RITC || '',
      CountryOfOrigin: item.CountryOfOrigin || '',
    },
    original: { ...item },
    deleted: false,
    added: false,
    corrected: {},
  }));
  renderItems();
  updateAccuracy();
}

// ── Header fields ──────────────────────────────────────
function renderHeaderFields() {
  const grid = document.getElementById('headerFields');
  grid.innerHTML = '';
  Object.entries(headerFieldStates).forEach(([field, state]) => {
    const div = document.createElement('div');
    div.className = 'field-item' + (state.corrected ? ' corrected' : '');
    div.innerHTML =
      '<label>' + esc(field) + '</label>'
      + '<input type="text" value="' + escAttr(state.value) + '" data-field="' + field + '" oninput="onHeaderInput(this)" />'
      + '<span class="toggle-btn" onclick="toggleHeader(\\'' + field + '\\')">' + (state.corrected ? 'Mark OK' : 'Mark Wrong') + '</span>';
    grid.appendChild(div);
  });
}

function onHeaderInput(el) {
  const f = el.dataset.field;
  headerFieldStates[f].value = el.value;
  const orig = agentResult[f] || '';
  if (el.value !== String(orig)) {
    headerFieldStates[f].corrected = true;
    el.closest('.field-item').classList.add('corrected');
    el.closest('.field-item').querySelector('.toggle-btn').textContent = 'Mark OK';
  }
  updateAccuracy();
}

function toggleHeader(field) {
  const s = headerFieldStates[field];
  s.corrected = !s.corrected;
  if (!s.corrected) s.value = agentResult[field] || '';
  renderHeaderFields();
  updateAccuracy();
}

// ── Party fields ───────────────────────────────────────
function renderPartyFields() {
  const grid = document.getElementById('partyFields');
  grid.innerHTML = '';
  Object.entries(partyFieldStates).forEach(([key, state]) => {
    const div = document.createElement('div');
    div.className = 'field-item' + (state.corrected ? ' corrected' : '');
    const isAddr = key.endsWith('.Address');
    const inputTag = isAddr
      ? '<textarea rows="2" data-field="' + key + '" oninput="onPartyInput(this)">' + esc(state.value) + '</textarea>'
      : '<input type="text" value="' + escAttr(state.value) + '" data-field="' + key + '" oninput="onPartyInput(this)" />';
    div.innerHTML = '<label>' + esc(key) + '</label>' + inputTag
      + '<span class="toggle-btn" onclick="toggleParty(\\'' + key + '\\')">' + (state.corrected ? 'Mark OK' : 'Mark Wrong') + '</span>';
    grid.appendChild(div);
  });
}

function onPartyInput(el) {
  const key = el.dataset.field;
  partyFieldStates[key].value = el.value;
  const parts = key.split('.');
  const orig = (agentResult[parts[0]] || {})[parts[1]] || '';
  if (el.value !== String(orig)) {
    partyFieldStates[key].corrected = true;
    el.closest('.field-item').classList.add('corrected');
    el.closest('.field-item').querySelector('.toggle-btn').textContent = 'Mark OK';
  }
  updateAccuracy();
}

function toggleParty(key) {
  const s = partyFieldStates[key];
  s.corrected = !s.corrected;
  if (!s.corrected) {
    const parts = key.split('.');
    s.value = (agentResult[parts[0]] || {})[parts[1]] || '';
  }
  renderPartyFields();
  updateAccuracy();
}

// ── Line items ─────────────────────────────────────────
const ITEM_COLS = ['PartNo','ItemCode','ItemDescription','Quantity','UnitOfQty','UnitPrice','RITC','CountryOfOrigin'];

function renderItems() {
  const tbody = document.getElementById('itemsBody');
  const activeItems = itemRows.filter(r => !r.deleted);
  document.getElementById('itemCount').textContent = activeItems.length;

  tbody.innerHTML = itemRows.map((row, idx) => {
    if (row.deleted) {
      return '<tr class="row-deleted" data-idx="' + idx + '"><td>' + (idx+1) + '</td>'
        + ITEM_COLS.map(c => '<td><input disabled value="' + escAttr(row.fields[c]) + '"></td>').join('')
        + '<td class="row-actions"><button class="row-btn" onclick="undeleteRow(' + idx + ')" title="Restore">&#x21A9;</button></td></tr>';
    }
    const rowCls = row.added ? ' class="row-added"' : '';
    return '<tr' + rowCls + ' data-idx="' + idx + '"><td>' + (idx+1) + '</td>'
      + ITEM_COLS.map(c => {
        const corrected = row.corrected[c] ? ' corrected' : '';
        return '<td><input class="' + corrected + '" value="' + escAttr(row.fields[c]) + '" data-idx="' + idx + '" data-col="' + c + '" oninput="onItemInput(this)" /></td>';
      }).join('')
      + '<td class="row-actions">'
      + '<button class="row-btn del" onclick="deleteRow(' + idx + ')" title="Delete">&#x2715;</button>'
      + '</td></tr>';
  }).join('');
}

function onItemInput(el) {
  const idx = parseInt(el.dataset.idx);
  const col = el.dataset.col;
  const row = itemRows[idx];
  row.fields[col] = el.value;
  const orig = row.original ? String(row.original[col] || '') : '';
  if (el.value !== orig) {
    row.corrected[col] = true;
    el.classList.add('corrected');
  } else {
    row.corrected[col] = false;
    el.classList.remove('corrected');
  }
  updateAccuracy();
}

function deleteRow(idx) {
  itemRows[idx].deleted = true;
  renderItems();
  updateAccuracy();
}

function undeleteRow(idx) {
  itemRows[idx].deleted = false;
  renderItems();
  updateAccuracy();
}

function addRow() {
  const empty = {};
  ITEM_COLS.forEach(c => empty[c] = '');
  itemRows.push({ fields: { ...empty }, original: null, deleted: false, added: true, corrected: {} });
  renderItems();
  updateAccuracy();
  // Scroll to bottom
  const scroll = document.querySelector('.table-scroll');
  scroll.scrollTop = scroll.scrollHeight;
}

function filterItems() {
  const q = document.getElementById('itemSearch').value.toLowerCase();
  const rows = document.querySelectorAll('#itemsBody tr');
  rows.forEach(tr => {
    if (!q) { tr.style.display = ''; return; }
    const text = tr.textContent.toLowerCase();
    tr.style.display = text.includes(q) ? '' : 'none';
  });
}

// ── Accuracy ───────────────────────────────────────────
function updateAccuracy() {
  let total = 0, correct = 0;

  // Header
  Object.values(headerFieldStates).forEach(s => {
    total++;
    if (!s.corrected) correct++;
  });

  // Party
  Object.values(partyFieldStates).forEach(s => {
    total++;
    if (!s.corrected) correct++;
  });

  // Items
  itemRows.forEach(row => {
    if (row.deleted && !row.added) {
      // Agent had this row, worker deleted = all fields wrong
      total += ITEM_COLS.length;
    } else if (row.added) {
      // Worker added = agent missed, all fields wrong
      total += ITEM_COLS.length;
    } else {
      ITEM_COLS.forEach(c => {
        total++;
        if (!row.corrected[c]) correct++;
      });
    }
  });

  const pct = total > 0 ? Math.round(correct / total * 100) : 100;
  const circumference = 2 * Math.PI * 22;
  const offset = circumference * (1 - pct / 100);

  const corrected = total - correct;
  // Switch from "AI Quality" to "Worker Accuracy" once worker starts editing
  if (corrected > 0) {
    document.getElementById('accLabel').textContent = 'Worker Accuracy';
    document.getElementById('ringPct').textContent = pct + '%';
    document.getElementById('ringCircle').setAttribute('stroke-dashoffset', offset);
    document.getElementById('ringCircle').setAttribute('stroke', pct >= 90 ? '#38a169' : pct >= 70 ? '#d69e2e' : '#e53e3e');
    document.getElementById('accValue').textContent = pct + '%';
  }
  document.getElementById('totalFieldsVal').textContent = total;
  document.getElementById('correctFieldsVal').textContent = correct;
  document.getElementById('correctedFieldsVal').textContent = corrected;
}

// ── PDF viewer ─────────────────────────────────────────
function loadPdf(url) {
  const panel = document.getElementById('sourcePanel');
  const noPdf = document.getElementById('noPdf');
  if (!url) {
    noPdf.textContent = 'No PDF available';
    return;
  }
  noPdf.style.display = 'none';
  const embed = document.createElement('embed');
  embed.src = url;
  embed.type = 'application/pdf';
  panel.appendChild(embed);
}

// ── Actions ────────────────────────────────────────────
function buildWorkerResult() {
  const result = {};
  // Header scalars
  Object.entries(headerFieldStates).forEach(([f, s]) => {
    result[f] = s.value;
  });
  // Parties
  result.Exporter = {};
  result.Importer = {};
  Object.entries(partyFieldStates).forEach(([key, s]) => {
    const parts = key.split('.');
    result[parts[0]][parts[1]] = s.value;
  });
  // Line items (exclude deleted, include added)
  result.LineItems = itemRows.filter(r => !r.deleted).map((r, i) => ({
    ItemNo: i + 1,
    ...r.fields,
  }));
  return result;
}

async function approveAll() {
  // Reset all to uncorrected (approve as-is)
  Object.keys(headerFieldStates).forEach(f => {
    headerFieldStates[f].corrected = false;
    headerFieldStates[f].value = agentResult[f] || '';
  });
  Object.keys(partyFieldStates).forEach(key => {
    partyFieldStates[key].corrected = false;
    const parts = key.split('.');
    partyFieldStates[key].value = (agentResult[parts[0]] || {})[parts[1]] || '';
  });
  // Reset items to original, remove added, undelete deleted
  itemRows = (agentResult.LineItems || []).map((item, i) => ({
    fields: {
      PartNo: item.PartNo || '',
      ItemCode: item.ItemCode || '',
      ItemDescription: item.ItemDescription || '',
      Quantity: item.Quantity || '',
      UnitOfQty: item.UnitOfQty || '',
      UnitPrice: item.UnitPrice || '',
      RITC: item.RITC || '',
      CountryOfOrigin: item.CountryOfOrigin || '',
    },
    original: { ...item },
    deleted: false,
    added: false,
    corrected: {},
  }));
  renderHeaderFields();
  renderPartyFields();
  renderItems();
  updateAccuracy();
  await submitCorrections();
}

async function submitCorrections() {
  const workerResult = buildWorkerResult();
  const notes = document.getElementById('commentInput').value.trim() || null;
  try {
    const url = API + '/verify/' + JOB_ID + (notes ? '?worker_notes=' + encodeURIComponent(notes) : '');
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(workerResult),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'HTTP ' + res.status);
    }
    const result = await res.json();
    showToast('Submitted! Accuracy: ' + ((result.accuracy?.field_accuracy || 0) * 100).toFixed(1) + '%', 'success');
  } catch(e) {
    showToast('Error: ' + e.message, 'error');
  }
}

async function sendComment() {
  const text = document.getElementById('commentInput').value.trim();
  if (!text) return showToast('Enter a comment first', 'error');
  try {
    const res = await fetch(API + '/feedback/' + JOB_ID + '?text=' + encodeURIComponent(text), { method: 'POST' });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'HTTP ' + res.status);
    }
    showToast('Comment sent', 'success');
    document.getElementById('commentInput').value = '';
  } catch(e) {
    showToast('Error: ' + e.message, 'error');
  }
}

function showToast(msg, type) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = 'toast toast-' + type + ' show';
  setTimeout(() => el.classList.remove('show'), 3000);
}

function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }
function escAttr(s) { return String(s||'').replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

loadData();
</script>
</body>
</html>"""


# ─── Verification UI Endpoints ──────────────────────────
# NOTE: These must come before /verify/{job_id} to avoid route conflicts.

@app.get("/verify/ui/dashboard", response_class=HTMLResponse)
async def verify_dashboard():
    """Serve the worker verification dashboard."""
    return DASHBOARD_HTML


@app.get("/verify/ui/{job_id}", response_class=HTMLResponse)
async def verify_page(job_id: str):
    """Serve the verification page for a specific job."""
    return VERIFY_PAGE_HTML


@app.get("/verify/data/{job_id}")
async def verify_data(job_id: str):
    """Return JSON with extraction result + PDF info for the verification page."""
    v = verification_mgr.get_verification(job_id)
    if not v:
        # Auto-create verification from result if it exists
        if job_id in jobs and jobs[job_id].result:
            result_dict = jobs[job_id].result.model_dump()
            v = verification_mgr.create_verification(job_id, result_dict)
        else:
            raise HTTPException(status_code=404, detail="Job not found")

    agent_result = v.get('agent_result', {})

    # Find PDF file — local first, fall back to S3
    pdf_url = None
    for ext in ('.pdf', '.PDF'):
        pdf_path = UPLOAD_DIR / f"{job_id}{ext}"
        if pdf_path.exists():
            pdf_url = f"/uploads/{job_id}{ext}"
            break
    if not pdf_url:
        # Try to restore from S3
        import storage
        for ext in ('.pdf', '.PDF'):
            blob_key = f"uploads/{job_id}{ext}"
            data = storage.get_blob(blob_key)
            if data:
                local_path = UPLOAD_DIR / f"{job_id}{ext}"
                local_path.write_bytes(data)
                pdf_url = f"/uploads/{job_id}{ext}"
                logger.info(f"PDF restored from storage: {blob_key}")
                break

    # Build human-readable issues for worker from tool_log
    worker_alerts = []
    tool_log = agent_result.get('tool_log', [])
    seen_issues = set()
    for step in tool_log:
        if step.get('tool') == 'review_chunk' and step.get('issues'):
            for issue_text in step['issues']:
                if issue_text not in seen_issues:
                    seen_issues.add(issue_text)
                    worker_alerts.append(issue_text)

    # Fallback: if no detailed issues in trace, build from review_summary
    if not worker_alerts:
        review_summary = agent_result.get('review_summary', {})
        for chunk_key, chunk_info in review_summary.items():
            if not chunk_info.get('passed'):
                score = chunk_info.get('score', '?')
                expected = chunk_info.get('expected', '?')
                extracted = chunk_info.get('extracted', '?')
                crits = chunk_info.get('critical', 0)
                page_hint = chunk_key.replace('chunk_', 'Chunk ')
                if crits > 0:
                    worker_alerts.append(
                        f"{page_hint}: {crits} issues found (score {score}). "
                        f"Expected {expected} items, got {extracted}. Review all items from this section."
                    )

    # Add overall quality warning
    quality = agent_result.get('quality_score', 0)
    if quality and quality < 0.7:
        worker_alerts.insert(0,
            f"Overall quality is low ({quality:.0%}). This extraction needs careful review of all fields."
        )

    return {
        'job_id': job_id,
        'agent_result': agent_result,
        'review_level': v.get('review_level', 'full_review'),
        'status': v.get('status', 'pending'),
        'pdf_url': pdf_url,
        'created_at': v.get('created_at'),
        'worker_alerts': worker_alerts[:15],
    }


@app.get("/uploads/{filename}")
async def serve_upload(filename: str):
    """Serve uploaded PDF files for the verification viewer."""
    if not re.match(r'^[a-zA-Z0-9._-]+$', filename):
        raise HTTPException(status_code=400, detail="Invalid filename")
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(file_path), media_type="application/pdf")


# ─── Verification Endpoints ──────────────────────────────

@app.get("/verify/pending/list")
async def list_pending_verifications():
    """List all pending verification tasks."""
    return verification_mgr.list_pending()


@app.get("/verify/{job_id}")
async def get_verification(job_id: str):
    """Get verification status for a job."""
    v = verification_mgr.get_verification(job_id)
    if not v:
        # Auto-create verification from result if it exists
        if job_id in jobs and jobs[job_id].result:
            result_dict = jobs[job_id].result.model_dump()
            v = verification_mgr.create_verification(job_id, result_dict)
        else:
            raise HTTPException(status_code=404, detail="Job not found")
    return v


@app.post("/verify/{job_id}")
async def submit_verification(job_id: str, worker_result: Dict, worker_id: str = None, worker_notes: str = None):
    """Submit worker-verified result. Marks job as COMPLETED."""
    v = verification_mgr.submit_verification(job_id, worker_result, worker_id, worker_notes)
    if not v:
        raise HTTPException(status_code=404, detail="Verification not found")

    # Mark job as COMPLETED now that worker has verified
    if job_id in jobs:
        jobs[job_id].status = JobState.COMPLETED
        jobs[job_id].completed_at = time.time()
        logger.info(f"Job {job_id}: worker verified, status → COMPLETED")

    # Process feedback — create patterns from corrections
    corrections = v.get('corrections', [])
    if corrections:
        agent_result = v.get('agent_result', {})
        source_text = ""
        if job_id in jobs and jobs[job_id].result:
            source_text = str(jobs[job_id].result.model_dump())

        invoice_header = {
            k: agent_result.get(k) for k in [
                'InvoiceNo', 'Date', 'InvoiceCurrency', 'FreightTerms',
                'IncoTerms', 'TermsOfPayment', 'Exporter', 'Importer',
            ] if agent_result.get(k) is not None
        }

        feedback_result = feedback_proc.process_feedback(
            job_id=job_id,
            corrections=corrections,
            source_text=source_text,
            invoice_header=invoice_header,
            format_key=v.get('format_key'),
            worker_notes=worker_notes,
        )
        v['feedback'] = feedback_result

    # Update pattern confidence from worker accuracy
    accuracy = v.get('accuracy', {})
    patterns_used = v.get('agent_result', {}).get('patterns_used', [])
    if patterns_used and accuracy:
        field_acc = accuracy.get('field_accuracy', 0)
        for pid in patterns_used:
            if pid:
                pattern_lib.update_confidence(pid, worked=(field_acc >= 0.85))

    # Auto-promote patterns that have earned it
    promotable = pattern_lib.get_promotable_patterns()
    for p in promotable:
        pattern_lib.promote_to_kb(p['pattern_id'], 'learned_patterns.md')

    # Record metrics
    if accuracy:
        agent_result = v.get('agent_result', {})
        metrics_tracker.record_invoice(
            job_id=job_id,
            filename=agent_result.get('filename', ''),
            format_key=v.get('format_key', 'unknown'),
            company=str(agent_result.get('Exporter', {}).get('Name', '') if isinstance(agent_result.get('Exporter'), dict) else agent_result.get('Exporter', '')),
            accuracy=accuracy,
            quality_score=agent_result.get('quality_score', 0),
            review_level=v.get('review_level', 'full_review'),
            elapsed_seconds=agent_result.get('elapsed_seconds', 0),
        )

    return v


# ─── Feedback Endpoints ──────────────────────────────────

@app.post("/feedback/{job_id}")
async def submit_feedback(job_id: str, text: str):
    """Submit plain text feedback from worker."""
    agent_result = None
    if job_id in jobs and jobs[job_id].result:
        agent_result = jobs[job_id].result.model_dump()
    else:
        v = verification_mgr.get_verification(job_id)
        if v:
            agent_result = v.get('agent_result')
    if not agent_result:
        raise HTTPException(status_code=404, detail="Job not found")

    result = feedback_proc.process_plain_text_feedback(
        job_id=job_id,
        text=text,
        agent_result=agent_result,
        source_text=str(agent_result),
    )
    return result


# ─── Metrics Endpoints ───────────────────────────────────

@app.get("/metrics")
async def get_metrics(days: int = 30):
    """Get overall accuracy metrics."""
    return {
        "overall": metrics_tracker.get_overall_accuracy(days),
        "by_field": metrics_tracker.get_field_accuracy(days),
        "trend": metrics_tracker.get_improvement_trend(days * 2),
    }


@app.get("/metrics/format/{format_key}")
async def get_format_metrics(format_key: str):
    """Get accuracy metrics for a specific format."""
    return metrics_tracker.get_format_accuracy(format_key)


@app.get("/metrics/company/{company}")
async def get_company_metrics(company: str):
    """Get accuracy metrics for a specific company."""
    return metrics_tracker.get_company_accuracy(company)


@app.get("/patterns")
async def list_patterns():
    """List all patterns in the library."""
    return {"patterns": pattern_lib.patterns, "total": len(pattern_lib.patterns)}


@app.get("/patterns/promotable")
async def list_promotable_patterns():
    """List patterns ready for KB promotion."""
    return {"promotable": pattern_lib.get_promotable_patterns()}


# ─── Run ───────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting Invoice Parser API on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
