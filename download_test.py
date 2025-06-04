from google.oauth2 import service_account
from google.cloud import storage
import re
from collections import defaultdict
import time
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
    filename = f"{paper_id}v{version}.pdf"
    filepath = os.path.join(output_dir, filename)

    print(f"Downloading {blob.name} -> {filename}")
    #blob.download_to_filename(filepath)

print(f"\n✅ Downloaded {len(latest_versions)} latest-version PDFs.")