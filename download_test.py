from google.oauth2 import service_account
from google.cloud import storage
import pandas as pd
import re
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import fitz
from collections import Counter
import matplotlib.pyplot as plt
import json
# Global variable to hold food words in each subprocess
food_words_set = set()

def load_food_words(csv_path):
    """
    Load a CSV with food-related words into a set.
    Assumes a column called 'food_word'.
    """
    df = pd.read_csv(csv_path)
    words=[
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
    global arxiv_metadata
    #print("🔄 Initializing subprocess...")
    food_words_set = load_food_words(csv_path)
    file_path = 'arxiv-metadata-oai-snapshot.json'
    #with open(file_path) as f:
    #    records = [json.loads(line) for line in f]
    #arxiv_metadata = {record["id"]: record for record in records}
    #print(f"✅ Subprocess initialized with {len(food_words_set)} food words")

def download_and_process(paper_id, version, blob_name):
    # Recreate GCS client inside subprocess
    #credentials_path = r'C:\Users\mg6u19\Downloads\future-env-326822-d1f4c594ed5b.json'
    credentials_path = r'D:\download store\future-env-326822-6ae492a4c60a.json'
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    client = storage.Client(credentials=credentials)
    bucket = client.bucket("arxiv-dataset")
    blob = bucket.blob(blob_name)

    filename = f"{paper_id}v{version}.pdf"
    filepath = os.path.join("downloaded_papers", filename)
    subjects = arxiv_metadata[paper_id]["categories"].split()
    #print(f"⬇️ Downloading {blob_name} -> {filename}")
    blob.download_to_filename(filepath)

    text = ""
    with fitz.open(filepath) as doc:
        for page in doc:
            text += page.get_text()

    # === Basic food word matching ===
    text_words = set(re.findall(r'\b\w+\b', text.lower()))
    matched_words = {word: food_words_set[word] for word in text_words if word in food_words_set}

    txt_output_path = filepath.replace(".pdf", ".txt")

    #print(f"✅ {filename}: found {len(matched_words)} unique food words with counts.")
    #print(matched_words)
    os.remove(filepath)
    return filename, matched_words,subjects


def main():
    # === Setup ===
    #credentials_path = r'C:\Users\mg6u19\Downloads\future-env-326822-d1f4c594ed5b.json'
    credentials_path = r'D:\download store\future-env-326822-6ae492a4c60a.json'
    csv_path = "FoodData_Central_csv_2025-04-24/food.csv"

    credentials = service_account.Credentials.from_service_account_file(credentials_path)

    BUCKET_NAME = 'arxiv-dataset'
    PREFIX = 'arxiv/arxiv/pdf/'

    client = storage.Client(credentials=credentials)
    bucket = client.bucket(BUCKET_NAME)
    blobs = bucket.list_blobs(prefix=PREFIX)

    # === Track latest version per paper ===
    latest_versions = {}
    pattern = re.compile(r'(\d{4}\.\d{4,5})v(\d+)')

    start = time.time()
    i = 1
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

        if i == 10000:
            break
        i += 1

    #print(f"\n📦 Found {len(latest_versions)} latest-version PDFs in {time.time() - start:.2f}s.")

    # === Download & process PDFs in parallel ===
    os.makedirs("downloaded_papers", exist_ok=True)
    file_path = 'arxiv-metadata-oai-snapshot.json'
    with open(file_path) as f:
        #print("🔄 Loading arxiv metadata...")
        records = [json.loads(line) for line in f]
    #print(f"✅ Loaded {len(records)} records from arxiv metadata.")
    global arxiv_metadata
    all_subjects = set()

    for record in records:
        subjects = record.get("categories", "").split()
        all_subjects.update(subjects)
    arxiv_metadata = {record["id"]: record for record in records}
    #print("Unique subjects:", all_subjects)
    all_subjects = list(all_subjects)
    input_args = [(paper_id, version, blob_name) for paper_id, (version, blob_name) in latest_versions.items()]

    with ThreadPoolExecutor(initializer=init_worker, initargs=(csv_path,),max_workers=50) as executor:
        futures = [executor.submit(download_and_process, *args) for args in input_args]
        results = [f.result() for f in futures]

    print(f"\n✅ Downloaded and processed {len(results)} PDFs.\n")

    # Optional: #print summary of match es
    counters = {subject:Counter() for subject in all_subjects}
    total_counter=Counter()
    for filename, matches,subjects in results:
        #print(f"{filename}: {matches}")
        total_counter.update(matches)
        for subject in subjects:
            counters[subject].update(matches)
    print(total_counter)
    for subject, counter in counters.items():
        pdf_counter = pd.DataFrame(counter.items(), columns=['word', 'count'])
        pdf_counter.to_csv(f"data/{subject}_food_words.csv", index=False)
    pdf_total_counter = pd.DataFrame(total_counter.items(), columns=['word', 'count'])
    pdf_total_counter.to_csv("data/total_food_words.csv", index=False)
    top_words = total_counter.most_common(10)
    words, counts = zip(*top_words)
    plt.figure(figsize=(10, 5))
    plt.bar(words, counts, color='skyblue')
    plt.xlabel("Food Words")
    plt.ylabel("Count")
    plt.title("Top Food Word Frequencies")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
