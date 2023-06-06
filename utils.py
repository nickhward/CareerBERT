from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

nltk.download('stopwords')
nltk.download('punkt')
SENT_DETECTOR = nltk.data.load('tokenizers/punkt/english.pickle')

def calculate_cosine_similarity(sentences1, sentences2):
    vectorizer = TfidfVectorizer().fit_transform(sentences1 + sentences2)
    vectors = vectorizer.toarray()
    vec1 = vectors[:len(sentences1), :]
    vec2 = vectors[len(sentences1):, :]
    return cosine_similarity(vec1, vec2).mean()

def preprocess_sentence(sentence):
    # Convert the sentence to lowercase
    sentence = sentence.lower()
    
    # Remove punctuation
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize the sentence into words
    words = word_tokenize(sentence)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Join the words back into a single string
    sentence = ' '.join(words)
    
    return sentence

def split_into_sentences(text):
    # Replace the bullet points with a period
    text = text.replace(' ● ', '. ')

    # Split the text into sentences
    sentences = text.split('. ')

    return sentences