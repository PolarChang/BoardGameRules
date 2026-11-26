"""Rulebook file storage that works locally and on Google Cloud Storage."""
from __future__ import annotations

import shutil
import uuid
from datetime import timedelta
from pathlib import Path

from src.config import DATA_DIR, get_cloud_settings


def store_rulebook(source_path: str, game_id: str, document_id: str, filename: str) -> str:
    """Persist a rulebook and return its durable local path or ``gs://`` URI."""
    settings = get_cloud_settings()
    if settings.uses_cloud_storage:
        from google.cloud import storage

        object_name = f"rulebooks/{game_id}/{document_id}/{Path(filename).name}"
        client = storage.Client()
        blob = client.bucket(settings.storage_bucket).blob(object_name)
        blob.upload_from_filename(source_path, content_type="application/pdf")
        return f"gs://{settings.storage_bucket}/{object_name}"

    destination_dir = DATA_DIR / game_id
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / Path(filename).name
    shutil.copy2(source_path, destination)
    return str(destination)


def create_direct_upload(game_id: str, filename: str) -> dict:
    settings = get_cloud_settings()
    if not settings.uses_cloud_storage:
        raise RuntimeError("Direct Cloud Storage uploads are not configured")
    import google.auth
    from google.auth.transport.requests import Request
    from google.cloud import storage

    safe_name = Path(filename).name
    object_name = f"rulebooks/{game_id}/uploads/{uuid.uuid4().hex}/{safe_name}"
    credentials, _ = google.auth.default()
    credentials.refresh(Request())
    service_account_email = getattr(credentials, "service_account_email", None)
    if not service_account_email:
        raise RuntimeError("Cloud credentials do not identify a service account")

    blob = storage.Client(credentials=credentials).bucket(settings.storage_bucket).blob(object_name)
    upload_url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=15),
        method="PUT",
        content_type="application/pdf",
        service_account_email=service_account_email,
        access_token=credentials.token,
    )
    return {
        "object_name": object_name,
        "upload_url": upload_url,
        "expires_in_seconds": 900,
    }


def download_uploaded_rulebook(game_id: str, object_name: str, destination: str) -> str:
    settings = get_cloud_settings()
    if not settings.uses_cloud_storage or not object_name.startswith(f"rulebooks/{game_id}/uploads/"):
        raise ValueError("Invalid direct-upload object")
    from google.cloud import storage
    storage.Client().bucket(settings.storage_bucket).blob(object_name).download_to_filename(destination)
    return f"gs://{settings.storage_bucket}/{object_name}"
