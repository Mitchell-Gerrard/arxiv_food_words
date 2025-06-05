import json
import pandas as pd

file_path = 'arxiv-metadata-oai-snapshot.json'
with open(file_path) as f:
    records = [json.loads(line) for line in f]

all_subjects = set()

for record in records:
    subjects = record.get("categories", "").split()
    all_subjects.update(subjects)

print("Unique subjects:", all_subjects)
id_lookup = {record["id"]: record for record in records}
query_id = "0704.0218"
if query_id in id_lookup:
    categories = id_lookup[query_id]["categories"].split()
    print(f"Subjects for {query_id}:", type(categories))
else:
    print(f"No record found for ID {query_id}")