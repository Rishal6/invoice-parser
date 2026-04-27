"""
Template Registry — stores generated Python extractors keyed by template_id.

Each template is a regex/string-parsing Python function that can extract
invoice data without calling the LLM. Templates are saved as .py files
with metadata tracked in a JSON index.
"""
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

REGISTRY_DIR = Path(__file__).parent / 'data' / 'templates'
REGISTRY_FILE = REGISTRY_DIR / 'registry.json'


class TemplateRegistry:
    def __init__(self, registry_dir: Path = REGISTRY_DIR):
        self.registry_dir = registry_dir
        self.registry_file = registry_dir / 'registry.json'
        self.entries: Dict[str, Dict] = {}
        self._load()

    def _load(self):
        """Load registry index from JSON."""
        try:
            if self.registry_file.exists():
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                self.entries = {e['template_id']: e for e in data}
                logger.info(f"Template registry loaded: {len(self.entries)} templates")
            else:
                self.entries = {}
                logger.info("No template registry found, starting empty")
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load template registry: {e}")
            self.entries = {}

    def _save(self):
        """Save registry index to JSON."""
        try:
            self.registry_dir.mkdir(parents=True, exist_ok=True)
            with open(self.registry_file, 'w') as f:
                json.dump(list(self.entries.values()), f, indent=2, default=str)
            logger.info(f"Saved template registry: {len(self.entries)} templates")
        except IOError as e:
            logger.error(f"Failed to save template registry: {e}")

    def lookup(self, template_id: str) -> Optional[str]:
        """Return saved Python code for a template_id, or None if not found."""
        entry = self.entries.get(template_id)
        if not entry:
            return None

        code_path = self.registry_dir / f"{template_id}.py"
        if not code_path.exists():
            logger.warning(f"Template entry exists but code file missing: {code_path}")
            return None

        try:
            code = code_path.read_text(encoding='utf-8')
            # Update usage stats
            entry['times_used'] = entry.get('times_used', 0) + 1
            entry['last_used'] = int(time.time())
            self._save()
            return code
        except IOError as e:
            logger.error(f"Failed to read template code {code_path}: {e}")
            return None

    def save(self, template_id: str, code: str, metadata: dict):
        """Save a new extractor: write .py file + update registry index."""
        try:
            self.registry_dir.mkdir(parents=True, exist_ok=True)

            # Write code file
            code_path = self.registry_dir / f"{template_id}.py"
            code_path.write_text(code, encoding='utf-8')

            # Build registry entry
            entry = {
                'template_id': template_id,
                'column_headers': metadata.get('column_headers', []),
                'format_key': metadata.get('format_key', 'unknown'),
                'company': metadata.get('company'),
                'created_at': int(time.time()),
                'times_used': 0,
                'success_count': 0,
                'last_used': None,
            }
            self.entries[template_id] = entry
            self._save()
            logger.info(f"Saved template: {template_id} -> {code_path}")
        except IOError as e:
            logger.error(f"Failed to save template {template_id}: {e}")

    def record_success(self, template_id: str):
        """Increment success_count for a template."""
        entry = self.entries.get(template_id)
        if entry:
            entry['success_count'] = entry.get('success_count', 0) + 1
            self._save()

    def list_templates(self) -> list:
        """List all saved templates with metadata."""
        return list(self.entries.values())

    def sync_to_s3(self, bucket: str, prefix: str = "templates/"):
        """Upload registry + code files to S3."""
        import boto3
        try:
            s3 = boto3.client("s3")
            s3.upload_file(str(self.registry_file), bucket, f"{prefix}registry.json")
            for tid in self.entries:
                code_path = self.registry_dir / f"{tid}.py"
                if code_path.exists():
                    s3.upload_file(str(code_path), bucket, f"{prefix}{tid}.py")
            logger.info(f"[TEMPLATE] Synced {len(self.entries)} templates to s3://{bucket}/{prefix}")
        except Exception as e:
            logger.error(f"[TEMPLATE] S3 sync failed: {e}")

    def sync_from_s3(self, bucket: str, prefix: str = "templates/"):
        """Download registry + code files from S3 on startup."""
        import boto3
        try:
            s3 = boto3.client("s3")
            self.registry_dir.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, f"{prefix}registry.json", str(self.registry_file))
            self._load()
            for tid in self.entries:
                code_path = self.registry_dir / f"{tid}.py"
                try:
                    s3.download_file(bucket, f"{prefix}{tid}.py", str(code_path))
                except Exception:
                    logger.warning(f"[TEMPLATE] Code file not found in S3: {tid}.py")
            logger.info(f"[TEMPLATE] Loaded {len(self.entries)} templates from s3://{bucket}/{prefix}")
        except Exception as e:
            logger.warning(f"[TEMPLATE] S3 load failed (starting fresh): {e}")
