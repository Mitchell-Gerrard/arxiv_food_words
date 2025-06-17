import os
import shutil
import re
from tqdm import tqdm

def split_results_directory(results_dir="results"):
    file_pattern = re.compile(r'(\d{2})(\d{2})\.\d+v\d+\.json')

    files = [f for f in os.listdir(results_dir) if f.endswith('.json')]

    for filename in tqdm(files, desc="Reorganizing results"):
        match = file_pattern.match(filename)
        if not match:
            continue  # Skip files not matching the expected pattern

        year, month = match.groups()
        subdir = os.path.join(results_dir, year, f"{year}{month}")
        os.makedirs(subdir, exist_ok=True)

        src = os.path.join(results_dir, filename)
        dst = os.path.join(subdir, filename)
        shutil.move(src, dst)

    print("Result directory successfully split.")

if __name__ == "__main__":
    split_results_directory()
