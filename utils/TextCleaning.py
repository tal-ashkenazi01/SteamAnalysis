import string
import re
import nltk
from nltk.corpus import stopwords
import numpy as np
from scipy.stats import mannwhitneyu

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stop_words.add(["like", "yes", "just", 'actually', 'basically', 'seriously', 'literally'])

def clean_text(text):
    # ASSUMES THAT THE TEXT COMES IN LOWER-CASE
    # REMOVE PUNCTUATION
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    # REMOVE NUMBERS
    text = re.sub(f"[{re.escape(string.punctuation)}0-9]", "", text)
    # REMOVE STOPWORDS
    text = ' '.join([word for word in text.split() if word.lower() not in stop_words])
    return text


# USING COHEN'S TEST FOR MAGNITUDE OF DIFFERENCE BECAUSE WE CAN ASSUME NORMALITY FOR TIME PLAYED AT REVIEW
def cohen_d(group1, group2):
    # CALCULATE MEAN AND SD OF REVIEW HOURS
    mean1, mean2 = np.mean(group1), np.mean(group2)
    sd1, sd2 = np.std(group1, ddof=1), np.std(group2, ddof=1)

    # POOLED SD
    n1, n2 = len(group1), len(group2)
    pooled_sd = np.sqrt(((n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2) / (n1 + n2 - 2))
    d = (mean1 - mean2) / pooled_sd

    return d


def u_test(group1, group2):
    u_statistic, p_value = mannwhitneyu(group1, group2)

    # Calculate the effect size 'r'
    n1, n2 = len(group1), len(group2)
    r = u_statistic / (n1 * n2)
    r = np.abs(r - 0.5) / 0.5  # adjusting the effect size
    return r
