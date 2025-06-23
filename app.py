# app.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# --- 1. MODEL TRAINING (This will run only once) ---
# Use a decorator to cache the model training
@st.cache_data
def train_model():
    # Load the dataset
    df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])

    # Encode the labels
    encoder = LabelEncoder()
    df['label_encoded'] = encoder.fit_transform(df['label'])

    # Define features and target
    X = df['message']
    y = df['label_encoded']
    
    # Split data (not really needed for this simple app, but good practice)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Create and fit the TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

    # Train the Naive Bayes classifier
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    
    # Return the trained vectorizer and model
    return tfidf_vectorizer, model

# Train the model and vectorizer
vectorizer, model = train_model()


# --- 2. WEB INTERFACE using Streamlit ---

# Set the title of the web app
st.title("Phishing/Spam Text Classifier")

# Add some description
st.write(
    "Enter a message below to check if it's legitimate (Ham) or potential Phishing/Spam."
)

# Create a text area for user input
user_input = st.text_area("Enter your message here:")

# Create a button to classify the message
if st.button("Classify"):
    if user_input:
        # 1. Transform the user input using the trained vectorizer
        input_tfidf = vectorizer.transform([user_input])
        
        # 2. Make a prediction using the trained model
        prediction = model.predict(input_tfidf)[0] # get the first element
        
        # 3. Get the prediction probability
        prediction_prob = model.predict_proba(input_tfidf)[0]

        # 4. Display the result
        st.subheader("Result:")
        if prediction == 1: # 1 corresponds to 'spam'
            st.error("This message looks like SPAM / PHISHING.")
            st.write(f"Confidence: {prediction_prob[1]:.2%}")
        else: # 0 corresponds to 'ham'
            st.success("This message looks like HAM (Legitimate).")
            st.write(f"Confidence: {prediction_prob[0]:.2%}")
    else:
        st.warning("Please enter a message to classify.")