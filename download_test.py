from google.oauth2 import service_account
from google.cloud import storage
import pandas as pd
import re
import os
from concurrent.futures import ThreadPoolExecutor
import fitz
from collections import Counter
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import glob

# Global variables
food_words_set = set()
arxiv_metadata = {}

def load_food_words(csv_path):
    """
    Load a CSV with food-related words into a set.
    Assumes a column called 'description'.
    """
    df = pd.read_csv(csv_path)
    words = [
        word.strip().lower()
        for word in df['description']
        if isinstance(word, str) and len(word.strip()) > 1
    ]
    return Counter(words)

def init_worker(csv_path):
    """
    Runs once in each subprocess to load the food words into a global variable.
    """
    global food_words_set
    food_words_set = load_food_words(csv_path)

def download_and_process(paper_id, version, blob_name):
    """
    Download a PDF from GCS, extract text, match food words, and save the result.
    Skips if already processed.
    """
    result_path = f"results/{paper_id}v{version}.json"
    if os.path.exists(result_path):
        return None  # Skip already processed
    credentials_path = r'D:\download store\future-env-326822-6ae492a4c60a.json'
    #credentials_path = r'C:\Users\mg6u19\Downloads\future-env-326822-d1f4c594ed5b.json'
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    client = storage.Client(credentials=credentials)
    bucket = client.bucket("arxiv-dataset")
    blob = bucket.blob(blob_name)

    filename = f"{paper_id}v{version}.pdf"
    filepath = os.path.join("downloaded_papers", filename)

    try:
        blob.download_to_filename(filepath)
    except Exception as e:
        print(f"⚠️ Failed to download {blob_name}: {e}")
        return None

    try:
        text = ""
        with fitz.open(filepath) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print(f"⚠️ Failed to extract text from {filename}: {e}")
        return None
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

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

    return filename, matched_words, subjects

def main():
    # === Setup ===
    credentials_path = r'D:\download store\future-env-326822-6ae492a4c60a.json'
    #credentials_path = r'C:\Users\mg6u19\Downloads\future-env-326822-d1f4c594ed5b.json'
    csv_path = "FoodData_Central_csv_2025-04-24/food.csv"
    metadata_path = 'arxiv-metadata-oai-snapshot.json'
    os.makedirs("downloaded_papers", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    client = storage.Client(credentials=credentials)
    bucket = client.bucket("arxiv-dataset")
    blobs = bucket.list_blobs(prefix='arxiv/arxiv/pdf/')

    print("🔄 Loading arxiv metadata...")
    with open(metadata_path) as f:
        records = [json.loads(line) for line in f]
    print(f"✅ Loaded {len(records)} metadata records.")
    global arxiv_metadata
    arxiv_metadata = {record["id"]: record for record in records}

    # === Track latest version of each paper ===
    latest_versions = {}
    pattern = re.compile(r'(\d{4}\.\d{4,5})v(\d+)')
    print("🔍 Parsing blob list...")
    for blob in blobs:
        if not blob.name.endswith('.pdf'):
            continue
        match = pattern.search(blob.name)
        if not match:
            continue
        paper_id, version_str = match.groups()
        version = int(version_str)
        if paper_id not in latest_versions or version > latest_versions[paper_id][0]:
            latest_versions[paper_id] = (version, blob.name)

    print(f"✅ Found {len(latest_versions)} latest-version PDFs.")
    
    # === Build input args and skip already-processed files ===
    input_args = []
    for paper_id, (version, blob_name) in latest_versions.items():
        result_path = f"results/{paper_id}v{version}.json"
        if not os.path.exists(result_path):
            input_args.append((paper_id, version, blob_name))
    
    print(f"📦 {len(input_args)} PDFs to process (skipping already completed).")

    # === Run processing ===
    with ThreadPoolExecutor(initializer=init_worker, initargs=(csv_path,), max_workers=28) as executor:
        futures = [executor.submit(download_and_process, *args) for args in input_args]
        for f in tqdm(futures, desc="Processing PDFs", unit='pdf', unit_scale=True):
            try:
                f.result()
            except Exception as e:
                print(f"⚠️ Error: {e}")

    print("\n✅ Processing complete. Aggregating results...\n")

    # === Aggregate Results ===
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

    # Save CSVs
    for subject, counter in counters.items():
        df = pd.DataFrame(counter.items(), columns=['word', 'count'])
        df.to_csv(f"data/{subject}_food_words.csv", index=False)

    df_total = pd.DataFrame(total_counter.items(), columns=['word', 'count'])
    df_total.to_csv("data/total_food_words.csv", index=False)

    # Plot top 10
    top_words = total_counter.most_common(10)
    if top_words:
        words, counts = zip(*top_words)
        plt.figure(figsize=(10, 5))
        plt.bar(words, counts, color='skyblue')
        plt.xlabel("Food Words")
        plt.ylabel("Count")
        plt.title("Top Food Word Frequencies")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print("No food words found to visualize.")

if __name__ == "__main__":
    main()