from google.oauth2 import service_account
from google.cloud import storage
import pandas as pd
import re
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import fitz
from collections import Counter

# Global variable to hold food words in each subprocess/thread
food_words_set = Counter()

def load_food_words(csv_path):
    """
    Load a CSV with food-related words into a Counter.
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
    Runs once in each subprocess/thread to load the food words into a global variable.
    """
    global food_words_set
    food_words_set = load_food_words(csv_path)
    print(f"✅ Worker initialized with {len(food_words_set)} food words")

def create_gcs_client():
    credentials_path = r'C:\Users\mg6u19\Downloads\future-env-326822-d1f4c594ed5b.json'
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    return storage.Client(credentials=credentials)

def download_and_process(paper_id, version, blob_name):
    # Recreate GCS client inside worker
    client = create_gcs_client()
    bucket = client.bucket("arxiv-dataset")
    blob = bucket.blob(blob_name)

    filename = f"{paper_id}v{version}.pdf"
    filepath = os.path.join("downloaded_papers", filename)

    print(f"⬇️ Downloading {blob_name} -> {filename}")
    blob.download_to_filename(filepath)

    text = ""
    with fitz.open(filepath) as doc:
        for page in doc:
            text += page.get_text()

    # === Food word matching ===
    text_words = re.findall(r'\b\w+\b', text.lower())
    matched_words = Counter()
    for word in text_words:
        if word in food_words_set:
            matched_words[word] += 1

    os.remove(filepath)  # Clean up PDF file

    print(f"✅ {filename}: found {sum(matched_words.values())} total food word occurrences, {len(matched_words)} unique.")
    return filename, matched_words

def get_latest_papers(limit=10):
    client = create_gcs_client()
    bucket = client.bucket('arxiv-dataset')
    blobs = bucket.list_blobs(prefix='arxiv/arxiv/pdf/')

    latest_versions = {}
    pattern = re.compile(r'(\d{4}\.\d{4,5})v(\d+)')

    i = 0
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

        i += 1
        if i >= limit:
            break

    input_args = [(pid, ver, blob_name) for pid, (ver, blob_name) in latest_versions.items()]
    return input_args

def run_executor(executor_class, worker_func, args_list, csv_path, max_workers=4):
    start = time.time()
    with executor_class(max_workers=max_workers, initializer=init_worker, initargs=(csv_path,)) as executor:
        results = list(executor.map(worker_func, args_list))
    elapsed = time.time() - start
    return elapsed, results

def main():
    csv_path = "FoodData_Central_csv_2025-04-24/food.csv"
    input_args = get_latest_papers(limit=10)
    os.makedirs("downloaded_papers", exist_ok=True)

    print("\n=== Running with ProcessPoolExecutor ===")
    t_process, results_process = run_executor(ProcessPoolExecutor, download_and_process, input_args, csv_path)
    print(f"ProcessPoolExecutor took {t_process:.2f} seconds")

    print("\n=== Running with ThreadPoolExecutor ===")
    t_thread, results_thread = run_executor(ThreadPoolExecutor, download_and_process, input_args, csv_path)
    print(f"ThreadPoolExecutor took {t_thread:.2f} seconds\n")

    print("Summary of matches from ProcessPoolExecutor:")
    total_counter_proc = Counter()
    for filename, matches in results_process:
        print(f"{filename}: {sum(matches.values())} occurrences, {len(matches)} unique words")
        total_counter_proc.update(matches)
    print(f"Total matched food words: {sum(total_counter_proc.values())}\n")

    print("Summary of matches from ThreadPoolExecutor:")
    total_counter_thread = Counter()
    for filename, matches in results_thread:
        print(f"{filename}: {sum(matches.values())} occurrences, {len(matches)} unique words")
        total_counter_thread.update(matches)
    print(f"Total matched food words: {sum(total_counter_thread.values())}\n")

if __name__ == "__main__":
    main()
