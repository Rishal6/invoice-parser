"""
Storage abstraction layer.

Switches between local filesystem (dev) and AWS S3+DynamoDB (prod)
based on the STORAGE_BACKEND env var ("local" or "aws").

Blob functions  -> S3 or local files
Document functions -> DynamoDB or local JSON files
"""

import json
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "local")
LOCAL_BLOB_ROOT = os.getenv("LOCAL_BLOB_ROOT", "/tmp/invoice_data/")
LOCAL_DOC_ROOT = os.getenv("LOCAL_DOC_ROOT", os.path.join(os.path.dirname(__file__), "data"))

# ---------------------------------------------------------------------------
# Local backend
# ---------------------------------------------------------------------------

def _local_blob_path(key: str) -> Path:
    return Path(LOCAL_BLOB_ROOT) / key


def _local_put_blob(key: str, data: bytes) -> None:
    p = _local_blob_path(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def _local_get_blob(key: str) -> bytes | None:
    p = _local_blob_path(key)
    try:
        return p.read_bytes()
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.error("get_blob(%s) failed: %s", key, e)
        return None


def _local_blob_exists(key: str) -> bool:
    return _local_blob_path(key).exists()


def _local_get_blob_url(key: str, expires: int = 3600) -> str | None:
    p = _local_blob_path(key)
    return f"file://{p}" if p.exists() else None


def _local_doc_path(collection: str, doc_id: str) -> Path:
    return Path(LOCAL_DOC_ROOT) / collection / f"{doc_id}.json"


def _local_put_doc(collection: str, doc_id: str, data: dict) -> None:
    p = _local_doc_path(collection, doc_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, default=str))


def _local_get_doc(collection: str, doc_id: str) -> dict | None:
    p = _local_doc_path(collection, doc_id)
    try:
        return json.loads(p.read_text())
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.error("get_doc(%s/%s) failed: %s", collection, doc_id, e)
        return None


def _local_query_docs(collection: str, filter_key: str = None, filter_value: str = None) -> list[dict]:
    folder = Path(LOCAL_DOC_ROOT) / collection
    if not folder.is_dir():
        return []
    results = []
    for f in folder.glob("*.json"):
        try:
            doc = json.loads(f.read_text())
        except Exception:
            continue
        if filter_key and filter_value:
            if str(doc.get(filter_key)) != str(filter_value):
                continue
        results.append(doc)
    return results


def _local_delete_doc(collection: str, doc_id: str) -> None:
    p = _local_doc_path(collection, doc_id)
    try:
        p.unlink(missing_ok=True)
    except Exception as e:
        logger.error("delete_doc(%s/%s) failed: %s", collection, doc_id, e)

# ---------------------------------------------------------------------------
# AWS backend
# ---------------------------------------------------------------------------

_s3 = None
_dynamo = None


def _get_s3():
    global _s3
    if _s3 is None:
        import boto3
        _s3 = boto3.client("s3")
    return _s3


def _get_dynamo():
    global _dynamo
    if _dynamo is None:
        import boto3
        _dynamo = boto3.resource("dynamodb")
    return _dynamo


def _s3_bucket() -> str:
    return os.environ["S3_BUCKET"]


def _dynamo_table():
    return _get_dynamo().Table(os.environ["DYNAMO_TABLE"])


def _aws_put_blob(key: str, data: bytes) -> None:
    try:
        _get_s3().put_object(Bucket=_s3_bucket(), Key=key, Body=data)
    except Exception as e:
        logger.error("aws put_blob(%s) failed: %s", key, e)


def _aws_get_blob(key: str) -> bytes | None:
    try:
        resp = _get_s3().get_object(Bucket=_s3_bucket(), Key=key)
        return resp["Body"].read()
    except _get_s3().exceptions.NoSuchKey:
        return None
    except Exception as e:
        logger.error("aws get_blob(%s) failed: %s", key, e)
        return None


def _aws_blob_exists(key: str) -> bool:
    try:
        _get_s3().head_object(Bucket=_s3_bucket(), Key=key)
        return True
    except Exception:
        return False


def _aws_get_blob_url(key: str, expires: int = 3600) -> str | None:
    try:
        return _get_s3().generate_presigned_url(
            "get_object",
            Params={"Bucket": _s3_bucket(), "Key": key},
            ExpiresIn=expires,
        )
    except Exception as e:
        logger.error("aws get_blob_url(%s) failed: %s", key, e)
        return None


def _aws_put_doc(collection: str, doc_id: str, data: dict) -> None:
    try:
        _dynamo_table().put_item(Item={
            "pk": collection,
            "sk": doc_id,
            "data": json.dumps(data, default=str),
        })
    except Exception as e:
        logger.error("aws put_doc(%s/%s) failed: %s", collection, doc_id, e)


def _aws_get_doc(collection: str, doc_id: str) -> dict | None:
    try:
        resp = _dynamo_table().get_item(Key={"pk": collection, "sk": doc_id})
        item = resp.get("Item")
        return json.loads(item["data"]) if item else None
    except Exception as e:
        logger.error("aws get_doc(%s/%s) failed: %s", collection, doc_id, e)
        return None


def _aws_query_docs(collection: str, filter_key: str = None, filter_value: str = None) -> list[dict]:
    from boto3.dynamodb.conditions import Key, Attr
    try:
        kwargs = {"KeyConditionExpression": Key("pk").eq(collection)}
        if filter_key and filter_value:
            kwargs["FilterExpression"] = Attr("data").contains(f'"{filter_key}": "{filter_value}"')
        resp = _dynamo_table().query(**kwargs)
        return [json.loads(item["data"]) for item in resp.get("Items", [])]
    except Exception as e:
        logger.error("aws query_docs(%s) failed: %s", collection, e)
        return []


def _aws_delete_doc(collection: str, doc_id: str) -> None:
    try:
        _dynamo_table().delete_item(Key={"pk": collection, "sk": doc_id})
    except Exception as e:
        logger.error("aws delete_doc(%s/%s) failed: %s", collection, doc_id, e)

# ---------------------------------------------------------------------------
# Public API — dispatches to the active backend
# ---------------------------------------------------------------------------

def put_blob(key: str, data: bytes) -> None:
    (_aws_put_blob if STORAGE_BACKEND == "aws" else _local_put_blob)(key, data)


def get_blob(key: str) -> bytes | None:
    return (_aws_get_blob if STORAGE_BACKEND == "aws" else _local_get_blob)(key)


def blob_exists(key: str) -> bool:
    return (_aws_blob_exists if STORAGE_BACKEND == "aws" else _local_blob_exists)(key)


def get_blob_url(key: str, expires: int = 3600) -> str | None:
    return (_aws_get_blob_url if STORAGE_BACKEND == "aws" else _local_get_blob_url)(key, expires)


def put_doc(collection: str, doc_id: str, data: dict) -> None:
    (_aws_put_doc if STORAGE_BACKEND == "aws" else _local_put_doc)(collection, doc_id, data)


def get_doc(collection: str, doc_id: str) -> dict | None:
    return (_aws_get_doc if STORAGE_BACKEND == "aws" else _local_get_doc)(collection, doc_id)


def query_docs(collection: str, filter_key: str = None, filter_value: str = None) -> list[dict]:
    return (_aws_query_docs if STORAGE_BACKEND == "aws" else _local_query_docs)(collection, filter_key, filter_value)


def delete_doc(collection: str, doc_id: str) -> None:
    (_aws_delete_doc if STORAGE_BACKEND == "aws" else _local_delete_doc)(collection, doc_id)


def download_blob_to_tmp(key: str) -> str:
    """Download a blob to /tmp and return the local file path.
    Needed for libraries like pypdf/pymupdf that require local files.
    """
    data = get_blob(key)
    if data is None:
        raise FileNotFoundError(f"Blob not found: {key}")
    suffix = Path(key).suffix or ""
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="invoice_")
    os.close(fd)
    Path(tmp_path).write_bytes(data)
    return tmp_path
