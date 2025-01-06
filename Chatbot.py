import os
import re
import csv
import ssl
import json
import nltk
import joblib
import random
import datetime
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Ignore SSL certificate verification
def ignore_ssl_warning():
    ssl._create_default_https_context = ssl._create_unverified_context
    nltk.data.path.append(os.path.abspath("nltk_data"))
    nltk.download('punkt')

# Clean the input text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = text.split()
    return ' '.join(tokens)

# Validate and correct dataset structure
def validate_and_correct_structure(data):
    if not isinstance(data, dict) or 'intents' not in data:
        corrected_data = {'intents': data if isinstance(data, list) else []}
    else:
        corrected_data = data

    for item in corrected_data['intents']:
        if 'tag' not in item:
            item['tag'] = 'unknown'
        if 'patterns' not in item:
            item['patterns'] = []
        if 'responses' not in item:
            item['responses'] = []

    return corrected_data

# Load and clean dataset
def load_and_clean_dataset(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    data = validate_and_correct_structure(data)

    cleaned_data = []
    greetings = []

    for item in data['intents']:
        tag = item['tag']
        if tag == "greeting":
            greetings.extend(item['patterns'])
        for pattern in item['patterns']:
            question = clean_text(pattern)
            responses = [clean_text(response) for response in item['responses']]
            cleaned_data.append({'question': question, 'responses': responses, 'tag': tag})
    
    # Save cleaned data to data/cleaned/intents.json
    cleaned_data_path = os.path.abspath("./source_data/cleaned/intents.json")
    with open(cleaned_data_path, 'w') as file:
        json.dump({'intents': cleaned_data}, file, indent=4)
    
    return cleaned_data, greetings

# Load and clean dataset
file_path = os.path.abspath("./source_data/raw/intents.json")
cleaned_data, greetings = load_and_clean_dataset(file_path)

# Prepare training data
questions = [item['question'] for item in cleaned_data]
answers = [item['tag'] for item in cleaned_data]

# Create the vectorizer and classifier pipeline
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', LogisticRegression(random_state=0, max_iter=10000))
])

# Train the model
pipeline.fit(questions, answers)

# Save the trained model with a timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = os.path.abspath(f"./models/chatbot_model_{timestamp}.pkl")
joblib.dump(pipeline, model_path)

# Find the best match for the user's input
def find_best_match(user_input, cleaned_data):
    cleaned_input = clean_text(user_input)
    input_vector = pipeline.named_steps['vectorizer'].transform([cleaned_input])
    similarities = []
    for item in cleaned_data:
        question_vector = pipeline.named_steps['vectorizer'].transform([item['question']])
        similarity = cosine_similarity(input_vector, question_vector)
        similarities.append((similarity, item))
    best_match = max(similarities, key=lambda x: x[0])
    return random.choice(best_match[1]['responses'])

# Generate a response from the chatbot
def chatbot_response(user_input, cleaned_data, greetings):
    if user_input.lower() in greetings:
        return random.choice(greetings)
    return find_best_match(user_input, cleaned_data)

# Display the sidebar menu and handle navigation
def display_menu():
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    return choice

# Handle the Home menu functionality
def handle_home(counter, cleaned_data, greetings):
    st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation.")

    if not os.path.exists('chat_log.csv'):
        with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

    counter += 1
    user_input = st.text_input("You:", key=f"user_input_{counter}")

    if user_input:
        response = chatbot_response(user_input, cleaned_data, greetings)
        st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")

        timestamp = datetime.datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")

        with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([user_input, response, timestamp])

        if response.lower() in ['goodbye', 'bye']:
            st.write("Thank you for chatting with me. Have a great day!")
            st.stop()

# Handle the Conversation History menu functionality
def handle_conversation_history():
    st.header("Conversation History")
    with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)
        for row in csv_reader:
            st.text(f"User: {row[0]}")
            st.text(f"Chatbot: {row[1]}")
            st.text(f"Timestamp: {row[2]}")
            st.markdown("---")

# Handle the About menu functionality
def handle_about():
    st.write("The goal of this project is to create a chatbot that can understand and respond to user input based on intents. The chatbot is built using Natural Language Processing (NLP) library and Logistic Regression, to extract the intents and entities from user input. The chatbot is built using Streamlit, a Python library for building interactive web applications.")

    st.subheader("Project Overview:")

    st.write("""
    The project is divided into two parts:
    1. NLP techniques and Logistic Regression algorithm is used to train the chatbot on labeled intents and entities.
    2. For building the Chatbot interface, Streamlit web framework is used to build a web-based chatbot interface. The interface allows users to input text and receive responses from the chatbot.
    """)

    st.subheader("Dataset:")

    st.write("""
    The dataset used in this project is a collection of labelled intents and entities. The data is stored in a list.
    - Intents: The intent of the user input (e.g. "greeting", "budget", "about")
    - Entities: The entities extracted from user input (e.g. "Hi", "How do I create a budget?", "What is your purpose?")
    - Text: The user input text.
    """)

    st.subheader("Streamlit Chatbot Interface:")

    st.write("The chatbot interface is built using Streamlit. The interface includes a text input box for users to input their text and a chat window to display the chatbot's responses. The interface uses the trained model to generate responses to user input.")

    st.subheader("Conclusion:")

    st.write("In this project, a chatbot is built that can understand and respond to user input based on intents. The chatbot was trained using NLP and Logistic Regression, and the interface was built using Streamlit. This project can be extended by adding more data, using more sophisticated NLP techniques, deep learning algorithms.")