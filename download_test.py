from google.cloud import storage
import io
import zipfile
from pdfminer.high_level import extract_text
from nltk.tokenize import sent_tokenize
import nltk

nltk.download('punkt')

# --- Settings ---
BUCKET_NAME = 'your-gcs-bucket-name'
PREFIX = 'your/prefix/'  # Optional GCS folder prefix

# --- GCS Client Setup ---
client = storage.Client()
bucket = client.bucket(BUCKET_NAME)