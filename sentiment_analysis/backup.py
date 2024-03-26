# from transformers import pipeline

# Load pre-trained sentiment analysis model
#sentiment_classifier = pipeline("sentiment-analysis")

#def predict_sentiment(text):
    # Perform sentiment analysis on the input text
 #   result = sentiment_classifier(text)
    
    # Extract the sentiment label and score from the result
  #  label = result[0]['label']
  #  score = result[0]['score']
    
   # return label, score

#from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
#import numpy as np

# Load pre-trained DistilBERT model and tokenizer
#tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
#model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

# Define class labels
#class_labels = ["Negative", "Neutral", "Positive"]

#def predict_sentiment(text):
    # Tokenize input text
 #   input_ids = tokenizer.encode(text, return_tensors='tf')

    # Perform inference
  #  outputs = model(input_ids)[0]

    # Get predicted class probabilities
   # probs = np.array(outputs[0])

    # Determine the predicted class label
    #predicted_class_idx = np.argmax(probs)
   # predicted_class = class_labels[predicted_class_idx]

    #return predicted_class
'''
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Define class labels
class_labels = ["Negative", "Neutral", "Positive"]

def predict_sentiment(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)

    # Perform inference
    outputs = model(**inputs)

    # Get predicted class probabilities
    probs = torch.softmax(outputs.logits, dim=1).detach().numpy()

    # Determine the predicted class label
    predicted_class_idx = np.argmax(probs)
    predicted_class = class_labels[predicted_class_idx]

    return predicted_class
'''
'''
# Importing necessary libraries
from transformers import BertTokenizer, TFBertForSequenceClassification
import numpy as np

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

# Define class labels
class_labels = ["Negative", "Neutral", "Positive"]

def predict_sentiment(text):
    # Tokenize input text
    input_ids = tokenizer.encode(text, return_tensors='tf')

    # Perform inference
    outputs = model(input_ids)[0]

    # Get predicted class probabilities
    probs = np.array(outputs[0])

    # Determine the predicted class label
    predicted_class_idx = np.argmax(probs)
    predicted_class = class_labels[predicted_class_idx]

    return predicted_class
'''
'''

import pandas as pd
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf

# Load the dataset from CSV
df = pd.read_csv(r'C:\Users\ganan\Documents\sentiment_analysis_project\sentiment_analysis\amazon_reviews.csv')  # Replace 'your_dataset.csv' with the path to your CSV file

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

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

# Define dataset class
class SentimentDataset(tf.keras.utils.Sequence):
    def __init__(self, dataframe, tokenizer, max_length, batch_size):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size

    def __len__(self):
        return len(self.dataframe) // self.batch_size

    def __getitem__(self, idx):
        batch = self.dataframe.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
        input_ids = []
        attention_masks = []

        for i, row in batch.iterrows():
            encoded = self.tokenizer.encode_plus(
                row['reviewText'],
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_token_type_ids=False,
                return_tensors='tf'
            )
            input_ids.append(encoded['input_ids'])
            attention_masks.append(encoded['attention_mask'])

        return (
            tf.concat(input_ids, axis=0),
            tf.concat(attention_masks, axis=0)
        )

# Define training and validation datasets
train_dataset = SentimentDataset(train_df, tokenizer, max_length=128, batch_size=32)
val_dataset = SentimentDataset(val_df, tokenizer, max_length=128, batch_size=32)

# Define optimizer and loss function
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Compile the model
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=3
)

# Predict sentiment for each review
def predict_sentiment(review_text):
    input_ids = tokenizer.encode(review_text, return_tensors='tf', max_length=128, padding='max_length', truncation=True)
    outputs = model(input_ids)
    probs = tf.nn.softmax(outputs.logits)
    predicted_class_idx = tf.argmax(probs, axis=1).numpy()[0]
    return class_labels[predicted_class_idx]
'''
