import os
import glob
import json
import logging
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import numpy as np
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from matplotlib.animation import FuncAnimation
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


def gaussian_process_regression(X, y):
    """
    Perform Gaussian Process Regression on the given data.
    
    Parameters:
    X (array-like): Input features.
    y (array-like): Target values.
    
    Returns:
    GaussianProcessRegressor: Fitted Gaussian Process model.
    """
    kernel = C(1.0, constant_value_bounds=(1e-6, 1e6)) * RBF(length_scale=100, length_scale_bounds=(1e1, 1e4))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)
    gpr.fit(X, y)
    return gpr

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
                if isinstance(e, json.JSONDecodeError):
                    os.remove(filepath)  # Remove corrupted file
    def mmyy_key(mmyy):
        month = int(mmyy[:2])
        year = int(mmyy[2:]) 
        return ( month, year)

    Dates = sorted(Dates, key=mmyy_key)
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

    bacon_counts=[]
    all_counts=[]
    ramen_counts=[]
    word_clouds_over_time=[]
    for date, counter in Dates_counters.items():
        most_common = counter.most_common(1)
        bacon_count=counter['bacon']
        all_count = sum(counter.values())
        ramen_counts.append(counter['ramen'])

        # Generate word cloud for the current date
        wordcloud = WordCloud(width=800, height=400, background_color='white',random_state=42)
        wordcloud.generate_from_frequencies(counter)
        word_clouds_over_time.append((date, wordcloud))
        all_counts.append(all_count)
        if bacon_count!=0:
            
            bacon_counts.append(bacon_count)
            
        else:
            bacon_counts.append(0)
            logger.info(f"No 'bacon' found for date {date}")
            logger.info(f"No words found for date {date}")
    bacon_x= np.arange(0, len(bacon_counts))
    all_x = np.arange(0, len(all_counts))
    ramen_x = np.arange(0, len(ramen_counts))
    bacon_gpr = gaussian_process_regression(bacon_x.reshape(-1, 1), np.array(bacon_counts/np.max(bacon_counts)))
    all_gpr = gaussian_process_regression(all_x.reshape(-1, 1), np.array(all_counts/np.max(all_counts)))
    ramen_gpr = gaussian_process_regression(ramen_x.reshape(-1, 1), np.array(ramen_counts/np.max(ramen_counts)))
    bacon_x_pred = np.arange(0, len(bacon_counts)).reshape(-1, 1)
    all_x_pred = np.arange(0, len(all_counts)).reshape(-1, 1)
    ramen_x_pred = np.arange(0, len(ramen_counts)).reshape(-1, 1)
    bacon_y_pred = bacon_gpr.predict(bacon_x_pred)
    all_y_pred = all_gpr.predict(all_x_pred)
    ramen_y_pred = ramen_gpr.predict(ramen_x_pred)
    start_date = pd.Timestamp("2007-04-01")

    # Generate dates for each month
    dates = [start_date + pd.DateOffset(months=i) for i in range(len(all_counts))]

    plt.plot(dates, bacon_y_pred, color='blue', label='bacon (GPR)')
    plt.plot(dates, all_y_pred, color='red', label='all words (GPR)')
    plt.plot(dates, ramen_y_pred, color='green', label='ramen (GPR)')
    plt.title('Gaussian Process Regression of Food Word Counts Over Time')
    #plt.scatter(bacon_x,bacon_counts/np.max(bacon_counts), color='blue', alpha=0.5,label='bacon')
    #plt.scatter(all_x,all_counts/np.max(all_counts), color='red', alpha=0.5,label='all words')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('counts')
    plt.gcf().autofmt_xdate() 
    plt.savefig('data/date/counts_over_time.png')
    def update(frame):


        plt.clf()
        date, wordcloud = word_clouds_over_time[frame]
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f"Word Cloud for {date}")
        plt.axis('off')
    
    fig = plt.figure(figsize=(10, 5))
    ani = FuncAnimation(fig, update, frames=len(word_clouds_over_time), repeat=False)
    ani.save('data/date/word_clouds_over_time.gif', writer='pillow', fps=1)
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