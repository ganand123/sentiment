import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os


# Construct the file path using os.path.join()
#file_path = os.path.join('C:\', 'Users', 'ganan', 'Documents', 'sentiment_analysis_project', 'sentiment_analysis', 'amazon_reviews.csv')

# Read the CSV file
#df = pd.read_csv(file_path)


# Load the dataset from CSV
df = pd.read_csv(r'C:\Users\ganan\Documents\sentiment_analysis_project\sentiment_analysis\amazon_reviews.csv')
#print('File path:', 'C:\\Users\\ganan\\Documents\\sentiment_analysis_project\\sentiment_analysis\\amazon_reviews.csv')
df['reviewText'].fillna('', inplace=True)

# Define class labels
class_labels = ["Negative", "Neutral", "Positive"]

# Map ratings to sentiment labels
def map_rating_to_sentiment(rating):
    if rating <= 2:
        return 0  # Negative
    elif rating == 3:
        return 1  # Neutral
    else:
        return 2  # Positive

df['sentiment'] = df['rating'].apply(map_rating_to_sentiment)

# Split the dataset into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = CountVectorizer()
#vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_df['reviewText'])
X_val = vectorizer.transform(val_df['reviewText'])

# Train a logistic regression classifier
model = LogisticRegression(max_iter=1000)
model.fit(X_train, train_df['sentiment'])

# Evaluate the model on the validation set
y_pred = model.predict(X_val)
accuracy = accuracy_score(val_df['sentiment'], y_pred)
print("Validation Accuracy:", accuracy)

# Predict sentiment for new review text
def predict_sentiment(review_text):
    text_vector = vectorizer.transform([review_text])
    predicted_sentiment = model.predict(text_vector)[0]
    return class_labels[predicted_sentiment]
