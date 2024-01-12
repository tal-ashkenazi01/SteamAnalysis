import string
import re
import nltk
from nltk.corpus import stopwords
import numpy as np
from scipy.stats import mannwhitneyu

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stop_words.update(
    {'like', 'yes', 'just', 'actually', 'basically', 'seriously', 'literally', 've', 'im', "didn", "youre", "got"})


def clean_text(text):
    # ASSUMES THAT THE TEXT COMES IN LOWER-CASE
    # REMOVE PUNCTUATION
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    # REMOVE NUMBERS
    text = re.sub(f"[{re.escape(string.digits)}0-9]", "", text)
    # REMOVE STOPWORDS
    text = ' '.join([word for word in nltk.word_tokenize(text) if word.lower() not in stop_words])

    return text


def generate_topics(top_words, key):
    # DETAILED PARAMETER LIST
    # https://huggingface.co/docs/api-inference/detailed_parameters#text-generation-task
    # THE ZEPHYR MODEL
    # https://huggingface.co/HuggingFaceH4/zephyr-7b-beta

    # GENERATIVE AI ENDPOINT
    ENDPOINT = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"

    # SYSTEM PROMPT
    system_input = """<|system|>
    Generate a response that is either a one to two word category describing gameplay experience or a genre. The response must be enclosed in braces. Input consists of five keywords derived from video game reviews through NMF, representing a specific topic or aspect of the game. Ensure that the resulting word or phrase closely aligns with the theme indicated by the keywords.</s>"""

    # THE FULL TEXT OF THE REQUEST:
    word_input = f"<|user|>Based on the following five keywords: {', '.join(top_words)}, identify and generate a one to two word category or genre that encapsulates the gameplay experience or aspect these words represent. The response should be concise and enclosed in braces, reflecting a clear theme or aspect of gaming experience as suggested by the keywords.</s>"

    text_input = system_input + word_input

    # LOAD EVERYTHING IN
    import requests
    headers = {"Authorization": f"Bearer {key}"}
    query = {"inputs": text_input, 'parameters': {'return_full_text': False}}

    try:
        response = requests.post(ENDPOINT, headers=headers, json=query)
        content = response.json()

        matches = re.findall(r'\{(.*?)\}', content[0]['generated_text'])
        extracted_text = matches[0].replace("'", "")
        extracted_text = extracted_text.replace("\"", "")
        array_extracted_text = extracted_text.split(',')
        if len(extracted_text) > 3:
            array_extracted_text = array_extracted_text[:3]

        return array_extracted_text
    except Exception as e:
        return ["Error generating this topic"]


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

# OLD PROMPTS
# Extract and return the most substantive terms or combinations of terms from the following list of user input words.
#     The input is a collection top words extracted from video game reviews.
#     Combine logically related words: "really" and "interesting" should be combined to "really interesting"
#     Do not reuse terms.
#     If the terms are only loosely related, do not combine them.
#     Leave out terms that appear irrelevant to the rest.
#     Leave out spelling mistakes if you encounter them.
#
#     Return only the list of combinations separated by commas and enclosed by braces.
#
#     Example return format of substantive combinations/words:
#     example input: 'dungeon', 'crawler', 'cave', 'exploration', 'traps'
#     example output: { dungeon crawler, cave exploration, traps }</s>
