# data/preprocess.py
import pandas as pd
import re
from nltk.corpus import stopwords

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text)  # Remove URLs/mentions
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Example usage (uncomment to test)
# df = pd.read_csv("data/troll_data.csv")
# df['cleaned_text'] = df['comment_text'].apply(clean_text)
# df.to_csv("data/cleaned_data.csv", index=False)