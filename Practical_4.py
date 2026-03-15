import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Sample News Data (as required by lab sheet)
news_data = {
    'text': [
        'The election results are in for the political party.',
        'The striker scored a goal in the football match.',
        'New AI models are changing the tech landscape.'
    ],
    'category': ['Politics', 'Sports', 'Tech']
}
df = pd.DataFrame(news_data)

# 1. Cleaning & Lemmatization
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    words = text.lower().split()
    cleaned = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(cleaned)

df['cleaned'] = df['text'].apply(clean_text)

# 2. Label Encoding (Converting text categories to numbers)
le = LabelEncoder()
df['label'] = le.fit_transform(df['category'])

# 3. TF-IDF Representation
tfidf = TfidfVectorizer()
vectors = tfidf.fit_transform(df['cleaned'])

print("--- Practical 4: News Preprocessing ---")
print(df[['text', 'cleaned', 'label']])
print("\nTF-IDF Matrix Shape:", vectors.shape)