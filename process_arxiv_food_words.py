import os
import re
import json
import glob
import logging
from tqdm import tqdm
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import fitz
import pandas as pd
import matplotlib.pyplot as plt

from google.cloud import storage
from google.oauth2 import service_account

# === Setup Logging ===
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/process.log", mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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

def download_and_process(paper_id, version, blob_name):
    result_path = f"results/{paper_id}v{version}.json"
    if os.path.exists(result_path):
        return None

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

        text = ""
        with fitz.open(filepath) as doc:
            for page in doc:
                text += page.get_text()

        text_words = set(re.findall(r'\b\w+\b', text.lower()))
        matched_words = {word: food_words_set[word] for word in text_words if word in food_words_set}
        subjects = arxiv_metadata.get(paper_id, {}).get("categories", "").split()

        result_data = {
            'filename': filename,
            'matched_words': matched_words,
            'subjects': subjects
        }

        os.makedirs("results", exist_ok=True)
        with open(result_path, 'w') as f:
            json.dump(result_data, f)

        #logger.info(f"Processed {filename} with {len(matched_words)} matches")
        return filename, matched_words, subjects

    except Exception as e:
        logger.error(f"Failed to process {paper_id}: {e}")
        return None
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

# === Main Workflow ===
def main(chunk_prefix=None,agro=True):
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
    blob_prefix = f'arxiv/arxiv/pdf/{chunk_prefix}' if chunk_prefix else 'arxiv/arxiv/pdf/'
    blobs = bucket.list_blobs(prefix=blob_prefix)

    logger.info("Loading metadata...")
    with open(metadata_path) as f:
        records = [json.loads(line) for line in f]
    global arxiv_metadata
    arxiv_metadata = {r["id"]: r for r in records}
    logger.info(f"Loaded {len(arxiv_metadata)} metadata records")

    # Track latest version of each paper
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
            continue  # Skip if not in this chunk

        version = int(version_str)
        if paper_id not in latest_versions or version > latest_versions[paper_id][0]:
            latest_versions[paper_id] = (version, blob.name)

    logger.info(f" {len(latest_versions)} PDFs to consider in chunk '{chunk_prefix}'")

    # Build work list
    input_args = []
    for paper_id, (version, blob_name) in latest_versions.items():
        result_path = f"results/{paper_id}v{version}.json"
        if not os.path.exists(result_path):
            input_args.append((paper_id, version, blob_name))

    logger.info(f" {len(input_args)} PDFs to process after skipping completed.")

    # Process in parallel
    with ThreadPoolExecutor(initializer=init_worker, initargs=(csv_path,), max_workers=16) as executor:
        futures = [executor.submit(download_and_process, *args) for args in input_args]
        for f in tqdm(futures, desc="Processing PDFs", unit='pdf', unit_scale=True):
            try:
                f.result()
            except Exception as e:
                logger.error(f"Error in thread: {e}")

    logger.info("PDF processing complete. Aggregating results...")
    if agro==True:
        # === Aggregation ===
        results = []
        all_subjects = set()
        for filepath in glob.glob("results/*.json"):
            with open(filepath) as f:
                data = json.load(f)
                results.append((data['filename'], data['matched_words'], data['subjects']))
                all_subjects.update(data['subjects'])

        counters = {subject: Counter() for subject in all_subjects}
        total_counter = Counter()

        for filename, matches, subjects in results:
            total_counter.update(matches)
            for subject in subjects:
                counters[subject].update(matches)

        # Save counters
        for subject, counter in counters.items():
            df = pd.DataFrame(counter.items(), columns=['word', 'count'])
            df.to_csv(f"data/{subject}_food_words.csv", index=False)

        df_total = pd.DataFrame(total_counter.items(), columns=['word', 'count'])
        df_total.to_csv("data/total_food_words.csv", index=False)

        # Plot
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
    else:
        logger.info("Skipping aggregation as agro is set to False.")


if __name__ == "__main__":
    import argparse
    #parser = argparse.ArgumentParser(description="Process ArXiv PDFs for food-related words.")
    #parser.add_argument("--chunk", type=str, default=None, help="Optional chunk prefix (e.g., '23' or '2401')")
    args = '1106'#parser.parse_args()
    main(chunk_prefix=args,agro=False)
