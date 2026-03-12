from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd

# Using a sample sentence as required 
corpus = [
    'The quick brown foxes are jumping over the lazy dogs!',
    'Natural Language Processing is interesting.',
    'I love coding in Python.'
]

print("--- 1. Bag of Words (Count Occurrence) ---")
count_vec = CountVectorizer()
bow_matrix = count_vec.fit_transform(corpus)
print(pd.DataFrame(bow_matrix.toarray(), columns=count_vec.get_feature_names_out()))

print("\n--- 2. TF-IDF (Normalized Count) ---")
tfidf_vec = TfidfVectorizer()
tfidf_matrix = tfidf_vec.fit_transform(corpus)
print(pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vec.get_feature_names_out()))