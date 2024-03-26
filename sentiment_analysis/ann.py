import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load the dataset
df = pd.read_csv(r'C:\Users\ganan\Documents\sentiment_analysis_project\sentiment_analysis\amazon_reviews.csv')
df['reviewText'].fillna('', inplace=True)

# Define class labels
class_labels = ["Negative", "Neutral", "Positive"]

# Map ratings to sentiment labels
def map_rating_to_sentiment(rating):
    if rating <= 2:
        return 'Negative'
    elif rating == 3:
        return 'Neutral'
    else:
        return 'Positive'

df['sentiment'] = df['rating'].apply(map_rating_to_sentiment)

# Split the dataset into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X_train = vectorizer.fit_transform(train_df['reviewText'])
X_val = vectorizer.transform(val_df['reviewText'])

# Encode class labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df['sentiment'])
y_val = label_encoder.transform(val_df['sentiment'])

# Define the neural network model
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=15, batch_size=64, validation_data=(X_val, y_val))

# Evaluate the model on the validation set
loss, accuracy = model.evaluate(X_val, y_val)
print("Validation Accuracy:", accuracy)

# Predict sentiment for new review text
def predict_sentiment(review_text):
    text_vector = vectorizer.transform([review_text])
    predicted_prob = model.predict(text_vector)[0]
    predicted_sentiment = label_encoder.inverse_transform([predicted_prob.argmax()])[0]
    return predicted_sentiment
