import os
import requests
import feedparser
import fitz  # PyMuPDF
import csv

# SETTINGS
SEARCH_QUERY = "machine learning"
KEYWORDS = ["transformer", "bayesian", "reinforcement"]
MAX_RESULTS = 100
SAVE_DIR = "downloaded_pdfs"
RESULTS_CSV = "arxiv_results.csv"

# Ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

# Query arXiv API
query_url = f"http://export.arxiv.org/api/query?search_query=all:{SEARCH_QUERY.replace(' ', '+')}&start=0&max_results={MAX_RESULTS}"
feed = feedparser.parse(query_url)

# Function to download a PDF
def download_pdf(pdf_url, filename):
    response = requests.get(pdf_url)
    with open(filename, "wb") as f:
        f.write(response.content)

# Function to extract text and search for keywords
def extract_and_search(pdf_path, keywords):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    found = [kw for kw in keywords if kw.lower() in text.lower()]
    return found

# Process each paper
results = []
for entry in feed.entries:
    title = entry.title
    pdf_url = entry.link.replace("abs", "pdf") + ".pdf"
    safe_title = "".join(c if c.isalnum() else "_" for c in title)[:100]
    pdf_path = os.path.join(SAVE_DIR, f"{safe_title}.pdf")
    
    print(f"\n📥 Downloading: {title}")
    download_pdf(pdf_url, pdf_path)
    
    print(f"🔍 Searching for keywords in: {pdf_path}")
    found_keywords = extract_and_search(pdf_path, KEYWORDS)
    
    if found_keywords:
        print(f"✅ Found keywords: {found_keywords}")
        results.append((title, pdf_url, ", ".join(found_keywords)))
    else:
        print("❌ No keywords found.")

# Write summary to CSV
with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Title", "PDF URL", "Found Keywords"])
    for row in results:
        writer.writerow(row)

print(f"\n===== SUMMARY =====")
print(f"Saved to {RESULTS_CSV}")
for title, url, keywords in results:
    print(f"- {title}\n  Keywords: {keywords}\n  URL: {url}\n")
