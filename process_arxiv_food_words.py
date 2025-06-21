import os
import re
import json
import glob
import logging
import subprocess
from tqdm import tqdm
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from multiprocessing import Pool
import pdfplumber
import pymupdf  # PyMuPDF
import pandas as pd
import matplotlib.pyplot as plt
import time
from google.cloud import storage
from google.oauth2 import service_account
import gc
from concurrent.futures import as_completed
import warnings
import sys
import os
import contextlib
import msvcrt

os.makedirs("logs", exist_ok=True)
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/process.log", mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Now add the filter immediately after logging setup
class PymupdfWarningFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        if "Cannot set gray non-stroke color" in msg:
            print(f"Filtered pymupdf warning: {msg}", file=sys.stderr)
            return False
        print(f"Logging pymupdf warning: {msg}", file=sys.stderr)
        return True

logging.getLogger().addFilter(PymupdfWarningFilter())
logging.getLogger("pymupdf").addFilter(PymupdfWarningFilter())

warnings.filterwarnings("ignore", message=r".*Cannot set gray non-stroke color.*", module="pymupdf")


# === Globals ===
food_words_set = set()
arxiv_metadata = {}

# === Utility Functions ===
def load_food_words(csv_path):
    df = pd.read_csv(csv_path)
    words = [
        word.strip().lower()
        for word in df['description']
        if isinstance(word, str) and len(word.strip()) > 1
    ]
    return Counter(words)

def init_worker(csv_path):
    global food_words_set
    food_words_set = load_food_words(csv_path)
@contextlib.contextmanager
def suppress_stderr():
    # Get stderr handle
    devnull = os.open(os.devnull, os.O_WRONLY)
    stderr_fileno = sys.stderr.fileno()
    saved = os.dup(stderr_fileno)
    try:
        os.dup2(devnull, stderr_fileno)
        yield
    finally:
        os.dup2(saved, stderr_fileno)
        os.close(saved)
        os.close(devnull)
def get_cleaned_pdf_path(original_path, suffix="_cleaned"):
    base_dir = os.path.dirname(original_path)
    filename = os.path.basename(original_path)
    name, ext = os.path.splitext(filename)
    cleaned_filename = f"{name}{suffix}{ext}"
    return os.path.join(base_dir, cleaned_filename)

def sanitize_pdf(input_path):

    output_path = get_cleaned_pdf_path(input_path)

    gs_path = r"C:\Program Files\gs\gs10.05.1\bin\gswin64c.exe"  # Adjust version/path
    command = [
        gs_path, "-dNOPAUSE", "-dBATCH", "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.4",
        f"-sOutputFile={output_path}", input_path
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        return output_path
    except subprocess.CalledProcessError:

        return None
def pymupdf_extract_text(filepath):
    try:
        with suppress_stderr():
            with pymupdf.open(filepath) as doc:
                return "".join(page.get_text("text") for page in doc)
    except Exception:
        return None
def extract_text_with_subprocess(filepath, timeout=30):
    with Pool(processes=1) as pool:
        result = pool.apply_async(pymupdf_extract_text, (filepath,))
        try:
            
            text = result.get(timeout=timeout)
            return text
        except TimeoutError:
            # The subprocess hung or crashed
            return None
def extract_text(filepath):
    cleaned_path = sanitize_pdf(filepath)
    target_path = cleaned_path if cleaned_path else filepath
    try:
        

        text = extract_text_with_subprocess(target_path)
        if text:
            return text
    except Exception as e:
        logging.warning(f"pymupdf failed on {target_path}: {e}, trying Plumber fallback")

    try:
        from pdfminer.high_level import extract_text as pdfminer_extract_text
        return pdfminer_extract_text(target_path)
    except Exception as e:
        logging.error(f"All PDF text extraction methods failed on {filepath}: {e}")
        return None

def get_result_path(paper_id, version):
    match = re.match(r"(\d{2})(\d{2})\.\d+", paper_id)
    if match:
        year, month = match.groups()
        result_dir = os.path.join("results", year, f"{year}{month}")
    else:
        result_dir = os.path.join("results", "misc")
    os.makedirs(result_dir, exist_ok=True)
    return os.path.join(result_dir, f"{paper_id}v{version}.json")

def download_and_process(paper_id, version, blob_name):
    result_path = get_result_path(paper_id, version)
    if os.path.exists(result_path):
        return None

    filepath = None
    try:
        credentials_path = r'D:\download store\future-env-326822-6ae492a4c60a.json'
        #credentials_path = r'C:\Users\mg6u19\Downloads\future-env-326822-d1f4c594ed5b.json'
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        client = storage.Client(credentials=credentials)
        bucket = client.bucket("arxiv-dataset")
        blob = bucket.blob(blob_name)

        filename = f"{paper_id}v{version}.pdf"
        filepath = os.path.join("downloaded_papers", filename)
        blob.download_to_filename(filepath)

        text = extract_text(filepath)
        if not text:
            return None

        text_words = set(re.findall(r'\b\w+\b', text.lower()))
        matched_words = {word: food_words_set[word] for word in text_words if word in food_words_set}
        subjects = arxiv_metadata.get(paper_id, {}).get("categories", "").split()

        result_data = {
            'filename': filename,
            'matched_words': matched_words,
            'subjects': subjects
        }

        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False)
        del text
        del text_words
    
        gc.collect()
  

        return filename, matched_words, subjects

    except Exception as e:
        logger.error(f"Failed to process {paper_id}: {e}")
        return None
    finally:
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
                if os.path.exists(get_cleaned_pdf_path(filepath)):
                    os.remove(get_cleaned_pdf_path(filepath))
            except:
                pass

def load_metadata_chunk(metadata_path, chunk_prefix):
    filtered = {}
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            paper_id = record['id']
            if paper_id.startswith(chunk_prefix):
                filtered[paper_id] = record
    return filtered

# === Main Workflow ===
def main(chunk_prefixes=None, agro=True):
    if chunk_prefixes is None:
        chunk_prefixes = [None]

    logger.info("Starting PDF processing")
    credentials_path = r'D:\download store\future-env-326822-6ae492a4c60a.json'
    #credentials_path = r'C:\Users\mg6u19\Downloads\future-env-326822-d1f4c594ed5b.json'
    csv_path = "FoodData_Central_csv_2025-04-24/food.csv"
    metadata_path = 'arxiv-metadata-oai-snapshot.json'

    os.makedirs("downloaded_papers", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    client = storage.Client(credentials=credentials)
    bucket = client.bucket("arxiv-dataset")

    logger.info("Loading metadata...")

    for chunk_prefix in chunk_prefixes:
        chunk_meta = load_metadata_chunk(metadata_path, chunk_prefix)
        arxiv_metadata.update(chunk_meta)
        logger.info(f"Processing chunk '{chunk_prefix}'")
        blob_prefix = f'arxiv/arxiv/pdf/{chunk_prefix}' if chunk_prefix else 'arxiv/arxiv/pdf/'
        blobs = bucket.list_blobs(prefix=blob_prefix)

        latest_versions = {}
        pattern = re.compile(r'(\d{4}\.\d{4,5})v(\d+)')
        logger.info("Parsing blob list...")
        for blob in blobs:
            if not blob.name.endswith('.pdf'):
                continue
            match = pattern.search(blob.name)
            if not match:
                continue
            paper_id, version_str = match.groups()
            if chunk_prefix and not paper_id.startswith(chunk_prefix):
                continue
            version = int(version_str)
            if paper_id not in latest_versions or version > latest_versions[paper_id][0]:
                latest_versions[paper_id] = (version, blob.name)

        logger.info(f" {len(latest_versions)} PDFs to consider in chunk '{chunk_prefix}'")

        input_args = []
        for paper_id, (version, blob_name) in latest_versions.items():
            result_path = get_result_path(paper_id, version)
            if not os.path.exists(result_path):
                input_args.append((paper_id, version, blob_name))

        logger.info(f" {len(input_args)} PDFs to process after skipping completed.")

        with ThreadPoolExecutor(initializer=init_worker, initargs=(csv_path,), max_workers=16) as executor:
            futures = [executor.submit(download_and_process, *args) for args in input_args]
            for f in tqdm(as_completed(futures), desc=f"Processing PDFs chunk {chunk_prefix}", unit='pdf', unit_scale=True):
                try:
                    f.result()
                except Exception as e:
                    logger.error(f"Error in thread: {e}")
        arxiv_metadata.clear()
        futures.clear()
    logger.info("PDF processing complete.")

    if not agro:
        logger.info("Skipping aggregation.")
        return

    # === Aggregation ===
    logger.info("Aggregating results...")
    results = []
    all_subjects = set()
    for filepath in glob.glob("results/**/*.json", recursive=True):
        with open(filepath, encoding='utf-8') as f:
            data = json.load(f)
            results.append((data['filename'], data['matched_words'], data['subjects']))
            all_subjects.update(data['subjects'])

    counters = {subject: Counter() for subject in all_subjects}
    total_counter = Counter()

    for filename, matches, subjects in results:
        total_counter.update(matches)
        for subject in subjects:
            counters[subject].update(matches)

    for subject, counter in counters.items():
        df = pd.DataFrame(counter.items(), columns=['word', 'count'])
        df.to_csv(f"data/{subject}_food_words.csv", index=False)

    df_total = pd.DataFrame(total_counter.items(), columns=['word', 'count'])
    df_total.to_csv("data/total_food_words.csv", index=False)

    top_words = total_counter.most_common(10)
    if top_words:
        words, counts = zip(*top_words)
        plt.figure()
        plt.bar(words, counts, color='skyblue')
        plt.xlabel("Food Words")
        plt.ylabel("Count")
        plt.title("Top Food Word Frequencies")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("data/top_food_words.png")

    logger.info("Aggregation complete.")

if __name__ == "__main__":
    logging.getLogger("pymupdf").setLevel(logging.ERROR)
    args = [f"{year:02d}{month:02d}" for year in range(21, 26) for month in range(1, 13)]
    args = [arg for arg in args if  2407 <=int(arg) <= 2506]
    main(chunk_prefixes=args, agro=True)