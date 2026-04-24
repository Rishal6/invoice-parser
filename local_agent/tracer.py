"""
Job Tracer — structured step-by-step logging for invoice extraction.

Each tool call becomes a step. Steps track: name, duration, tokens, result, errors.
Outputs: clean console log + JSON trace file per job.
"""
import json
import time
import logging
from pathlib import Path
from typing import Optional

import storage

logger = logging.getLogger(__name__)

INPUT_COST_PER_M = 3.0
OUTPUT_COST_PER_M = 15.0

STEP_LABELS = {
    'split_pdf': 'SPLIT',
    'extract_chunk': 'EXTRACT',
    'review_chunk': 'REVIEW',
    're_extract': 'RETRY',
    'process_all_chunks': 'PARALLEL',
    'deduplicate': 'DEDUP',
    'verify_final': 'VERIFY',
}

_BOLD = '\033[1m'
_DIM = '\033[2m'
_GREEN = '\033[32m'
_RED = '\033[31m'
_YELLOW = '\033[33m'
_CYAN = '\033[36m'
_ORANGE = '\033[38;5;208m'
_RESET = '\033[0m'
_LINE = '━'


class JobTracer:
    """Tracks every step of one extraction job."""

    def __init__(self, job_id: str, filename: str = ""):
        self.job_id = job_id
        self.filename = filename
        self.steps = []
        self.started_at = time.time()
        self.total_in = 0
        self.total_out = 0

        self._print_header()

    def _print_header(self):
        bar = _LINE * 50
        print(f"\n{_BOLD}{_CYAN}{bar}{_RESET}")
        print(f"{_BOLD}  JOB {self.job_id}  {_DIM}{self.filename}{_RESET}")
        print(f"{_BOLD}{_CYAN}{bar}{_RESET}")

    def step(self, tool: str, chunk: Optional[int] = None,
             result: str = "", tokens_in: int = 0, tokens_out: int = 0,
             duration: float = 0, error: str = "", extra: dict = None):
        """Log one completed step."""
        self.total_in += tokens_in
        self.total_out += tokens_out

        entry = {
            'tool': tool,
            'chunk': chunk,
            'result': result,
            'tokens_in': tokens_in,
            'tokens_out': tokens_out,
            'duration': round(duration, 1),
            'error': error,
            'timestamp': time.time(),
        }
        if extra:
            entry.update(extra)
        self.steps.append(entry)

        self._print_step(entry)
        self._save()

    def finish(self, total_items: int, quality_score: float, passed: bool):
        """Print final summary and save."""
        elapsed = time.time() - self.started_at
        cost = self._cost()

        tok_str = self._fmt_tokens(self.total_in + self.total_out)
        status_color = _GREEN if passed else _RED
        status_label = "PASS" if passed else "FAIL"

        bar = _LINE * 50
        print(f"{_BOLD}{_CYAN}{bar}{_RESET}")
        print(
            f"  {status_color}{_BOLD}{status_label}{_RESET}"
            f"  {total_items} items"
            f"  {_DIM}|{_RESET}  score {quality_score:.2f}"
            f"  {_DIM}|{_RESET}  {elapsed:.1f}s"
            f"  {_DIM}|{_RESET}  {tok_str}"
            f"  {_DIM}|{_RESET}  ~${cost:.3f}"
        )
        print(f"{_BOLD}{_CYAN}{bar}{_RESET}\n")

        self._save()

    def get_trace(self) -> dict:
        """Return the full trace as a dict."""
        elapsed = time.time() - self.started_at
        return {
            'job_id': self.job_id,
            'filename': self.filename,
            'started_at': self.started_at,
            'elapsed': round(elapsed, 1),
            'total_tokens': {
                'in': self.total_in,
                'out': self.total_out,
                'total': self.total_in + self.total_out,
            },
            'estimated_cost': round(self._cost(), 4),
            'steps': self.steps,
        }

    def get_tool_log(self) -> list:
        """Return steps in the old tool_log format for backward compat."""
        log = []
        for s in self.steps:
            entry = {
                'tool': s['tool'],
                'chunk_index': s.get('chunk'),
                'input_tokens': s.get('tokens_in', 0),
                'output_tokens': s.get('tokens_out', 0),
            }
            if s.get('error'):
                entry['error'] = s['error']
            for k in ('items_found', 'passed', 'score', 'critical',
                       'additional', 'corrected', 'before', 'after',
                       'exact_dupes', 'fuzzy_dupes', 'total_pages',
                       'chunks', 'truncated', 'vision', 'model',
                       'retry', 'total_chunks', 'total_items',
                       'parallel_workers', 'elapsed', 'overlap'):
                if k in s:
                    entry[k] = s[k]
            log.append(entry)
        return log

    def _cost(self) -> float:
        return (self.total_in / 1_000_000 * INPUT_COST_PER_M
                + self.total_out / 1_000_000 * OUTPUT_COST_PER_M)

    def _print_step(self, entry: dict):
        tool = entry['tool']
        chunk = entry.get('chunk')
        label = STEP_LABELS.get(tool, tool.upper())
        if chunk is not None:
            label = f"{label}[{chunk}]"

        label = f"{label:<14}"
        result = entry.get('result', '')
        dur = entry.get('duration', 0)
        tok = entry.get('tokens_in', 0) + entry.get('tokens_out', 0)
        err = entry.get('error', '')

        tok_str = self._fmt_tokens(tok) if tok > 0 else ""
        dur_str = f"{dur:.1f}s"

        if err:
            color = _RED
            result = f"ERROR: {err[:60]}"
        elif 'PASS' in result.upper() or 'DONE' in result.upper():
            color = _GREEN
        elif 'FAIL' in result.upper():
            color = _RED
        else:
            color = _RESET

        line = f"  {_ORANGE}{label}{_RESET} {color}{result:<45}{_RESET}"
        if tok_str:
            line += f" {_DIM}{tok_str:>8}{_RESET}"
        line += f" {_DIM}{dur_str:>6}{_RESET}"

        print(line)

    def _fmt_tokens(self, tok: int) -> str:
        if tok >= 1_000_000:
            return f"{tok/1_000_000:.1f}M tok"
        if tok >= 1000:
            return f"{tok/1000:.0f}K tok"
        return f"{tok} tok"

    def _save(self):
        try:
            storage.put_doc('traces', self.job_id, self.get_trace())
        except Exception as e:
            logger.warning(f"Failed to save trace: {e}")
