from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
    return ' '.join(filtered_tokens)

def train_model(X_train, y_train):
    # Fit the TF-IDF vectorizer with your training data
    model.fit(X_train, y_train)

def predict_sentiment(text):
    # Make sure the model is fitted before making predictions
    try:
        # Try to make predictions
        preprocessed_text = preprocess_text(text)
        return model.predict([preprocessed_text])[0]
    except NotFittedError as e:
        # Handle the case when the model is not fitted
        print("Error:", e)
        return None  # Return None or handle the error as needed


