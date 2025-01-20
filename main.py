import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# MAping of words index back to words(for understanding)
word_index=imdb.get_word_index()
reverse_word_index={value: key for key, value in word_index.items()}

## load the pre-trained model with ReLu activation
model=load_model('simple_rnn_imdb.h5')

## Function to preprocess user unit
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

## Prediction function

def predict_sentiment(review):
    preprocess_input=preprocess_text(review)
    
    prediction=model.predict(preprocess_input)
    
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    
    return sentiment,prediction[0][0]


### Design my streamlit app
import streamlit as st
st.title("IMDB Movie Review Sentiment analysis")
st.write('Enter a movie review tp classify it as positive or negative')

#user input
user_input= st.text_area('Movie Review')

if st.button('Classify'):
    preprocess_input=preprocess_text(user_input)
    
    ##Make prediction
    prediction=model.predict(preprocess_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    
    #display result
    st.write(f'Sentiment: {sentiment}')
    st.write(f"Prediction Score: {prediction[0][0]}")
    
else:
    st.write('Please enter a movie review.')