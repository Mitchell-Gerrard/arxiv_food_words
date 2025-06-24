import os
import glob
import json
import logging
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/analysis.log", mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)




def main():

    results = []
    all_subjects = set()
    Dates=set()
    for filepath in glob.glob("results/**/*.json", recursive=True):

        with open(filepath, encoding='utf-8') as f:
            try:
                data = json.load(f)
                date= filepath.split('/')[2]

                results.append((data['filename'], data['matched_words'], data['subjects'],date))
                all_subjects.update(data['subjects'])
                Dates.update([date])
    
            except Exception as e:
                logger.error(f"Failed to load {filepath}: {e}")
                if e == json.JSONDecodeError:
                    os.remove(filepath)  # Remove corrupted file

    Dates_counters = {date: Counter() for date in Dates}
    counters = {subject: Counter() for subject in all_subjects}
    total_counter = Counter()

    for filename, matches, subjects,date in results:
        total_counter.update(matches)
        for subject in subjects:
            counters[subject].update(matches)
        Dates_counters[date].update(matches)

    for subject, counter in counters.items():
        df = pd.DataFrame(counter.items(), columns=['word', 'count'])
        df.to_csv(f"data/{subject}_food_words.csv", index=False)

    df_total = pd.DataFrame(total_counter.items(), columns=['word', 'count'])
    df_total.to_csv("data/total_food_words.csv", index=False)
    for date, counter in Dates_counters.items():
        df = pd.DataFrame(counter.items(), columns=['word', 'count'])
        df.to_csv(f"data/date/{date}_food_words.csv", index=False)
    for date, counter in Dates_counters.items():
        most_common = counter.most_common(1)
        if most_common:
            word, count = most_common[0]
            logger.info(f"Most common word in {date}: {word} ({count})")
        else:
            logger.info(f"No words found for date {date}")
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
if __name__ == "__main__":

    main()
    logger.info("Data analysis completed successfully.")