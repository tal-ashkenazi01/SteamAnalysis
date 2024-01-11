import string
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def clean_text(text):
    # ASSUMES THAT THE TEXT COMES IN LOWER-CASE
    # REMOVE PUNCTUATION
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    # REMOVE NUMBERS
    text = re.sub(f"[{re.escape(string.punctuation)}0-9]", "", text)
    # REMOVE STOPWORDS
    text = ' '.join([word for word in text.split() if word.lower() not in stop_words])
    return text
