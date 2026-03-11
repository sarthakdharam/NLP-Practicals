import nltk
from nltk.tokenize import (word_tokenize, TreebankWordTokenizer, 
                           TweetTokenizer, WhitespaceTokenizer)
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer

# Download required resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab') # Adding this just in case for newer NLTK versions

# Sample sentence as per lab requirements
sample_text = "The quick brown foxes are jumping over the lazy dogs! #NLP @MozeCollege"

print("--- TOKENIZATION ---")
# Using the class-based approach to avoid the ImportError
ws_tokenizer = WhitespaceTokenizer()
print("Whitespace:", ws_tokenizer.tokenize(sample_text))

print("Word Tokenize:", word_tokenize(sample_text))
print("Treebank:", TreebankWordTokenizer().tokenize(sample_text))
print("Tweet:", TweetTokenizer().tokenize(sample_text))

print("\n--- STEMMING ---")
p_stemmer = PorterStemmer()
s_stemmer = SnowballStemmer("english")
words = ["running", "flies", "happily", "foxes"]

print("Porter Stemmer:", [p_stemmer.stem(w) for w in words])
print("Snowball Stemmer:", [s_stemmer.stem(w) for w in words])

print("\n--- LEMMATIZATION ---")
lemmatizer = WordNetLemmatizer()
# Lemmatization usually requires the word to be treated as a verb (pos='v') to see change
print("Lemmatization (Verbs):", [lemmatizer.lemmatize(w, pos='v') for w in words])