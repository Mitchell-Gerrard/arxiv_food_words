import os
import glob
import json
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from wordcloud import WordCloud
from matplotlib.animation import FuncAnimation
import scipy.stats
import scipy.signal as signal
# Set up logging
os.makedirs("logs", exist_ok=True)
os.makedirs("data/date", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/analysis.log", mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def gaussian_process_regression(X, y):
    kernel = C(1.0, constant_value_bounds=(1e-6, 1e6)) * RBF(length_scale=100, length_scale_bounds=(1e-6, 1e6))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)
    gpr.fit(X, y)
    return gpr

def load_json(filepath):
    try:
        with open(filepath, encoding='utf-8') as f:
            data = json.load(f)
            date = filepath.split('\\')[2]
            return (data['filename'], data['matched_words'], data['subjects'], date)
    except Exception as e:
        logger.error(f"Failed to load {filepath}: {e}")
        if isinstance(e, json.JSONDecodeError):
            os.remove(filepath)
        return None

def mmyy_key(mmyy):
    month = int(mmyy[:2])
    year = int(mmyy[2:]) # Adjust for 2000s
    year += 2000 
    return (year, month)

def filter_months_by_gap(dates, entropies, min_gap_months=3):
    filtered_indices = []
    # Sort indices by entropy descending
    sorted_indices = sorted(range(len(entropies)), key=lambda i: entropies[i], reverse=True)
    for idx in sorted_indices:
        if all(abs(idx - selected_idx) >= min_gap_months for selected_idx in filtered_indices):
            filtered_indices.append(idx)
        if len(filtered_indices) == 10:
            break
    # Sort by date ascending for nicer presentation
    sorted_filtered = sorted(filtered_indices)
    return [(dates[i], entropies[i]) for i in sorted_filtered]

def main():
    results = []
    all_subjects = set()
    Dates = set()
    logger.info("Starting to load JSON files...")
    filepaths = glob.glob("results\\**\\*.json", recursive=True)
    logger.info(f"Found {len(filepaths)} JSON files to process.")
    with ThreadPoolExecutor(max_workers=128) as executor:
        futures = {executor.submit(load_json, fp): fp for fp in filepaths}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result:
                filename, matched_words, subjects, date = result
                results.append(result)
                all_subjects.update(subjects)
                Dates.add(date)
            if i % 1000 == 0:
                logger.info(f"Processed {i} / {len(filepaths)} files")

    Dates = sorted(Dates, key=mmyy_key)
    Dates_counters = {date: Counter() for date in Dates}
    counters = {subject: Counter() for subject in all_subjects}
    total_counter = Counter()

    for filename, matches, subjects, date in results:
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

    bacon_counts, all_counts, ramen_counts = [], [], []
    word_clouds_over_time = []

    for date in Dates:
        print(date)
        counter = Dates_counters[date]
 
        bacon_counts.append(counter['bacon'])
        all_counts.append(sum(counter.values()))
        ramen_counts.append(counter['ramen'])

        wordcloud = WordCloud(width=800, height=400, background_color='white', random_state=42)
        wordcloud.generate_from_frequencies(counter)
        word_clouds_over_time.append((date, wordcloud))

    bacon_counts = np.array(bacon_counts)
    all_counts = np.array(all_counts)
    ramen_counts = np.array(ramen_counts)

    bacon_gpr = gaussian_process_regression(np.arange(len(bacon_counts)).reshape(-1, 1), bacon_counts / bacon_counts.max())
    all_gpr = gaussian_process_regression(np.arange(len(all_counts)).reshape(-1, 1), all_counts / all_counts.max())
    ramen_gpr = gaussian_process_regression(np.arange(len(ramen_counts)).reshape(-1, 1), ramen_counts / ramen_counts.max())

    dates = [pd.Timestamp("2007-04-01") + pd.DateOffset(months=i) for i in range(len(all_counts))]
    plt.plot(dates, bacon_gpr.predict(np.arange(len(bacon_counts)).reshape(-1, 1)), label='bacon (GPR)')
    plt.plot(dates, all_gpr.predict(np.arange(len(all_counts)).reshape(-1, 1)), label='all words (GPR)')
    plt.plot(dates, ramen_gpr.predict(np.arange(len(ramen_counts)).reshape(-1, 1)), label='ramen (GPR)')
    plt.title('Gaussian Process Regression of Food Word Counts Over Time')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Normalized Counts')
    plt.gcf().autofmt_xdate()
    plt.savefig('data/date/counts_over_time.png')
    plt.close()

    fig = plt.figure(figsize=(10, 5))
    def update(frame):
        plt.clf()
        date, wordcloud = word_clouds_over_time[frame]
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f"Word Cloud for {date}")
        plt.axis('off')
    ani = FuncAnimation(fig, update, frames=len(word_clouds_over_time), repeat=False)
    ani.save('data/date/word_clouds_over_time.gif', writer='pillow', fps=1)
    plt.close()

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
        plt.close()

    entropy_over_time = []
    for date in Dates:
        counter = Dates_counters[date]
        freqs = np.array(list(counter.values()))
        if freqs.sum() == 0:
            entropy = 0
        else:
            p = freqs / freqs.sum()
            entropy = scipy.stats.entropy(p)
        entropy_over_time.append(entropy)

    plt.figure()
    peaks, _=signal.find_peaks(entropy_over_time, height=3.7, distance=3)
    plt.scatter([dates[i] for i in peaks], [entropy_over_time[i] for i in peaks], color='red', label='Peaks')

    plt.plot(dates, entropy_over_time, label='Lexical Diversity (Entropy)', color='purple')
    for peak in peaks:
        plt.annotate(
            dates[peak].strftime('%Y-%m'),  # Format the date string as needed
            (dates[peak], entropy_over_time[peak]),
            textcoords="offset points",        # Position text with offset
            xytext=(0,5),                     # Offset label 10 points above the peak
            ha='center',                      # Center align text horizontally
            fontsize=8,
            color='red'
        )
    plt.xlabel('Date')
    plt.ylabel('Entropy')
    plt.title('Vocabulary Diversity Over Time')
    plt.savefig("data/date/entropy_over_time.png")
    plt.close()

    max_entropy_value = max(entropy_over_time)
    max_entropy_index = entropy_over_time.index(max_entropy_value)
    max_entropy_date = Dates[max_entropy_index]
    logger.info(f"Month with highest entropy: {max_entropy_date} (Entropy: {max_entropy_value:.4f})")

    subject_entropy = {}
    for subject, counter in counters.items():
        if subject == 'physics.optics':
            top_words = counter.most_common(10)
            words, counts = zip(*top_words)
            plt.figure()
            plt.bar(words, counts, color='skyblue')
            plt.xlabel("Food Words")
            plt.ylabel("Count")
            plt.title(f"Top Food Words for {subject}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"data/{subject}_top_food_words.png")
            plt.close()
        freqs = np.array(list(counter.values()))
        if freqs.sum() == 0:
            entropy = 0
        else:
            p = freqs / freqs.sum()
            entropy = scipy.stats.entropy(p)
        subject_entropy[subject] = entropy

    sorted_entropy = sorted(subject_entropy.items(), key=lambda x: x[1], reverse=True)[:10]
    subjects, values = zip(*sorted_entropy)
    plt.figure()
    plt.bar(subjects, values, color='orange')
    plt.xticks(rotation=45)
    plt.ylabel('Entropy')
    plt.title('Top 10 Subjects by Food Vocabulary Diversity')
    plt.tight_layout()
    plt.savefig("data/top_subject_entropy.png")
    plt.close()

    # Save the entropy ranking by month
    df_entropy = pd.DataFrame({'date': Dates, 'entropy': entropy_over_time})
    df_entropy.sort_values(by='entropy', ascending=False).to_csv("data/date/month_entropy_ranking.csv", index=False)

    # === New Code: Generate top 10 months table image with spacing ===
    import matplotlib.table as tbl
    top_entropy_months = filter_months_by_gap(Dates, entropy_over_time, min_gap_months=5)
    table_data = [(date, f"{entropy:.4f}") for date, entropy in top_entropy_months]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis('off')
    table = ax.table(
        cellText=table_data,
        colLabels=["Month", "Entropy"],
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    plt.title("Top 10 Months by Entropy (min 3 months apart)")
    plt.savefig("data/date/top_entropy_months_table.png")
    plt.close()
    logger.info("Saved top entropy months table image.")

if __name__ == "__main__":
    logger.info("Starting data analysis...")
    main()
    logger.info("Data analysis completed successfully.")
