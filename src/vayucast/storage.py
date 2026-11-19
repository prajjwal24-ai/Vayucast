from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from vayucast.config import get_env

logger = logging.getLogger(__name__)


def is_s3_enabled() -> bool:
    return bool(get_env().aws_s3_bucket)


def _get_client():
    return boto3.client("s3", region_name=get_env().aws_region)


def upload_file(local_path: Path, s3_key: str) -> bool:
    bucket = get_env().aws_s3_bucket
    if not bucket:
        return False
    _get_client().upload_file(str(local_path), bucket, s3_key)
    logger.info("Uploaded %s -> s3://%s/%s", local_path, bucket, s3_key)
    return True


def download_file(s3_key: str, local_path: Path) -> bool:
    bucket = get_env().aws_s3_bucket
    if not bucket:
        return False
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _get_client().download_file(bucket, s3_key, str(local_path))
        logger.info("Downloaded s3://%s/%s -> %s", bucket, s3_key, local_path)
        return True
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in {"404", "NoSuchKey"}:
            logger.warning("S3 key not found: s3://%s/%s", bucket, s3_key)
            return False
        raise


def upload_json(payload: dict[str, Any], s3_key: str) -> bool:
    bucket = get_env().aws_s3_bucket
    if not bucket:
        return False
    body = json.dumps(payload, indent=2).encode("utf-8")
    _get_client().put_object(
        Bucket=bucket,
        Key=s3_key,
        Body=body,
        ContentType="application/json",
    )
    logger.info("Uploaded JSON -> s3://%s/%s", bucket, s3_key)
    return True


def download_json(s3_key: str) -> dict[str, Any] | None:
    bucket = get_env().aws_s3_bucket
    if not bucket:
        return None
    try:
        resp = _get_client().get_object(Bucket=bucket, Key=s3_key)
        return json.loads(resp["Body"].read().decode("utf-8"))
    except (ClientError, BotoCoreError) as exc:
        logger.warning("Could not fetch s3://%s/%s: %s", bucket, s3_key, exc)
        return None