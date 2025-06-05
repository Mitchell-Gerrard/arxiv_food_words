from google.oauth2 import service_account
from google.cloud import storage
import re
from collections import defaultdict
import time
import os
from concurrent.futures import ProcessPoolExecutor
def download_and_process(item):
    paper_id, (version, blob_name) = item

    # Recreate the storage client in the subprocess (important!)
    client = storage.Client()
    bucket = client.bucket("arxiv-dataset")  # Adjust if needed
    blob = bucket.blob(blob_name)

    filename = f"{paper_id}v{version}.pdf"
    filepath = os.path.join(output_dir, filename)

    print(f"⬇️ Downloading {blob_name} -> {filename}")
    blob.download_to_filename(filepath)

    # === Add CPU-intensive logic here ===
    # For example: run PDF → text → sentence classification

    return filename
if __name__ == "__main__":
# --- Credentials ---
    credentials_path = r'C:\Users\mg6u19\Downloads\future-env-326822-d1f4c594ed5b.json'
    credentials = service_account.Credentials.from_service_account_file(credentials_path)

    # --- GCS Setup ---
    BUCKET_NAME = 'arxiv-dataset'
    PREFIX = 'arxiv/arxiv/pdf/'

    client = storage.Client(credentials=credentials)
    bucket = client.bucket(BUCKET_NAME)
    blobs = bucket.list_blobs(prefix=PREFIX)

    # --- Track latest version per paper ---
    latest_versions = {}

    # Regex to extract: 0801.0007v2 → (0801.0007, 2)
    pattern = re.compile(r'(\d{4}\.\d{4,5})v(\d+)')
    start=time.time()
    for blob in blobs:

        if not blob.name.endswith('.pdf'):
            continue

        match = pattern.search(blob.name)
        if not match:
            continue

        paper_id, version_str = match.groups()
        version = int(version_str)

        if paper_id not in latest_versions or version > latest_versions[paper_id][0]:
            latest_versions[paper_id] = (version, blob)

    # --- Download latest versions ---
    print(len(latest_versions),time.time()-start)
    output_dir = 'downloaded_papers'
    os.makedirs(output_dir, exist_ok=True)

    for paper_id, (version, blob) in latest_versions.items():
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(download_and_process, latest_versions.items()))

        print(f"\n✅ Downloaded and processed {len(results)} PDFs.")