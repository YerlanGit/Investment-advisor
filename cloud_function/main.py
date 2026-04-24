"""
Cloud Function: автоматическая индексация PDF при загрузке в GCS bucket.

Развертывание (одна команда):
    gcloud functions deploy ingest-pdf-trigger \
      --gen2 \
      --runtime=python312 \
      --region=us-central1 \
      --source=./cloud_function \
      --entry-point=on_pdf_uploaded \
      --trigger-event-filters="type=google.cloud.storage.object.v1.finalized" \
      --trigger-event-filters="bucket=ramp-bot-reports" \
      --set-env-vars FINTECH_MASTER_KEY=$FINTECH_MASTER_KEY \
      --memory=1Gi \
      --timeout=300s

Как это работает:
  1. Пользователь загружает PDF в GCS bucket 'ramp-bot-reports'
  2. Eventarc автоматически вызывает эту функцию
  3. Функция скачивает PDF, парсит, добавляет в ChromaDB на Cloud SQL/GCS
  4. ChromaDB синхронизируется обратно в bucket 'ramp-bot-chroma-db'
  5. При следующем деплое бота — свежая база уже встроена в образ
"""
import functions_framework
import os
import tempfile
import sys

from google.cloud import storage


@functions_framework.cloud_event
def on_pdf_uploaded(cloud_event):
    """
    Triggered by a finalized object in a GCS bucket.
    Only processes .pdf files.
    """
    data      = cloud_event.data
    bucket_name = data["bucket"]
    blob_name   = data["name"]

    if not blob_name.lower().endswith(".pdf"):
        print(f"[Skip] Not a PDF: {blob_name}")
        return

    print(f"[Trigger] New PDF: gs://{bucket_name}/{blob_name}")

    # Download PDF to /tmp
    gcs = storage.Client()
    bucket = gcs.bucket(bucket_name)
    blob   = bucket.blob(blob_name)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name
        blob.download_to_filename(tmp_path)
        print(f"[Download] Saved to {tmp_path}")

    # Download existing ChromaDB from GCS (if exists) to /tmp/chroma_db
    chroma_bucket_name = os.getenv("CHROMA_BUCKET", "ramp-bot-chroma-db")
    chroma_local = "/tmp/chroma_db"
    os.makedirs(chroma_local, exist_ok=True)
    _download_chroma_db(gcs, chroma_bucket_name, chroma_local)

    # Ingest the PDF
    sys.path.insert(0, "/workspace/src")   # Cloud Function working dir
    from agent.rag_engine import FinancialRAG

    rag = FinancialRAG(db_path=chroma_local)
    n   = rag.ingest_pdf(tmp_path, doc_metadata={"filename": blob_name})
    print(f"[Ingest] Added {n} chunks. Total: {rag.collection.count()}")

    # Upload updated ChromaDB back to GCS
    _upload_chroma_db(gcs, chroma_bucket_name, chroma_local)

    # Cleanup
    os.unlink(tmp_path)
    print("[Done] ChromaDB updated in GCS.")


def _download_chroma_db(gcs_client, bucket_name: str, local_path: str):
    """Download all files from GCS bucket prefix 'chroma_db/' to local_path."""
    try:
        bucket = gcs_client.bucket(bucket_name)
        blobs  = list(bucket.list_blobs(prefix="chroma_db/"))
        for blob in blobs:
            rel      = blob.name[len("chroma_db/"):]
            dest     = os.path.join(local_path, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            blob.download_to_filename(dest)
        print(f"[GCS] Downloaded {len(blobs)} ChromaDB files.")
    except Exception as e:
        print(f"[GCS] ChromaDB not found in bucket (first run?): {e}")


def _upload_chroma_db(gcs_client, bucket_name: str, local_path: str):
    """Upload all local ChromaDB files to GCS bucket under prefix 'chroma_db/'."""
    bucket = gcs_client.bucket(bucket_name)
    uploaded = 0
    for root, _, files in os.walk(local_path):
        for fname in files:
            local_file = os.path.join(root, fname)
            rel        = os.path.relpath(local_file, local_path)
            blob_name  = f"chroma_db/{rel}".replace("\\", "/")
            blob       = bucket.blob(blob_name)
            blob.upload_from_filename(local_file)
            uploaded += 1
    print(f"[GCS] Uploaded {uploaded} ChromaDB files.")
