import streamlit as st
import numpy as np
import pickle
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Load the LSTM Model
model=load_model('next_word_lstm.h5')

#3 Laod the tokenizer
with open('tokenizer.pickle','rb') as handle:
    tokenizer=pickle.load(handle)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return ' '.join(text.split())

# Predict the top k next words, skipping OOV predictions
def predict_top_k_next_words(model, tokenizer, text, max_sequence_len, k=3):
    text = clean_text(text)
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) == 0:
        return []
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)[0]
    top_k = []

    # Get top predictions, skip padding index 0 and the OOV index 1
    for idx in np.argsort(predicted)[::-1]:
        idx = int(idx)
        if idx <= 1:
            continue
        word = tokenizer.index_word.get(idx)
        if word and word != '<OOV>':
            top_k.append(word)
            if len(top_k) == k:
                break
    return top_k

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    top_words = predict_top_k_next_words(model, tokenizer, text, max_sequence_len, k=3)
    return top_words[0] if top_words else None

# streamlit app
st.title("Next Word Prediction With LSTM And Early Stopping")
input_text=st.text_input("Enter the sequence of Words","To be or not to")
if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1  # Retrieve the max sequence length from the model input shape
    next_words = predict_top_k_next_words(model, tokenizer, input_text, max_sequence_len, k=3)
    if next_words:
        st.write(f'Next word: {next_words[0]}')
        st.write(f'Top candidates: {next_words}')
    else:
        st.write('No valid prediction available.')

