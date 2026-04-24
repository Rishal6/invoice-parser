"""
Sandboxed Runner — executes saved Python extraction code safely.

Restricts globals to a minimal set (re, json, builtins).
No file I/O, no imports beyond re/json, no network.
Timeout: 10 seconds max.
"""
import json
import logging
import re
import signal
import sys

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 10

ALLOWED_MODULES = {'re': re, 'json': json}


def _safe_import(name, *args, **kwargs):
    if name in ALLOWED_MODULES:
        return ALLOWED_MODULES[name]
    raise ImportError(f"Import not allowed: {name}")


# Restricted globals — only safe builtins + re + json
SAFE_GLOBALS = {
    '__builtins__': {
        '__import__': _safe_import,
        'str': str,
        'int': int,
        'float': float,
        'len': len,
        'range': range,
        'enumerate': enumerate,
        'list': list,
        'dict': dict,
        'tuple': tuple,
        'set': set,
        'None': None,
        'True': True,
        'False': False,
        'isinstance': isinstance,
        'ValueError': ValueError,
        'TypeError': TypeError,
        'KeyError': KeyError,
        'IndexError': IndexError,
        'Exception': Exception,
        'min': min,
        'max': max,
        'abs': abs,
        'round': round,
        'sorted': sorted,
        'zip': zip,
        'map': map,
        'filter': filter,
        'any': any,
        'all': all,
        'strip': str.strip,
        'split': str.split,
    },
    're': re,
    'json': json,
}


class RunnerTimeout(Exception):
    pass


class RunnerError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise RunnerTimeout(f"Extractor timed out after {TIMEOUT_SECONDS}s")


def run_extractor(code: str, raw_text: str) -> dict:
    """
    Execute saved Python extraction code in a sandboxed environment.

    Args:
        code: Python source code containing a `def extract(raw_text: str) -> dict` function
        raw_text: The OCR text to extract from

    Returns:
        Extraction result dict, or error dict on failure.
    """
    # Compile the code
    try:
        compiled = compile(code, '<extractor>', 'exec')
    except SyntaxError as e:
        logger.error(f"Extractor syntax error: {e}")
        return {"error": f"SyntaxError: {e}", "success": False}

    # Prepare sandbox
    sandbox = dict(SAFE_GLOBALS)
    sandbox['__name__'] = '__extractor__'

    # Set timeout (Unix only — skip on Windows)
    use_timeout = hasattr(signal, 'SIGALRM')
    if use_timeout:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(TIMEOUT_SECONDS)

    try:
        # Execute the code to define the extract function
        exec(compiled, sandbox)

        # Check that extract function was defined
        extract_fn = sandbox.get('extract')
        if not callable(extract_fn):
            return {"error": "Code did not define an 'extract' function", "success": False}

        # Call the function
        result = extract_fn(raw_text)

        if not isinstance(result, dict):
            return {"error": f"extract() returned {type(result).__name__}, expected dict", "success": False}

        result['success'] = True
        return result

    except RunnerTimeout:
        logger.error("Extractor timed out")
        return {"error": f"Timeout after {TIMEOUT_SECONDS}s", "success": False}
    except Exception as e:
        logger.error(f"Extractor runtime error: {type(e).__name__}: {e}")
        return {"error": f"{type(e).__name__}: {e}", "success": False}
    finally:
        if use_timeout:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
