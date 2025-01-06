import os
import streamlit as st
from Chatbot import display_menu, handle_about, handle_conversation_history, handle_home, ignore_ssl_warning, load_and_clean_dataset

def app_deployment():
    counter = 0
    st.title("Chatbot by Vikash using NLP")
    st.markdown(
        "This is a simple chatbot that can help you with basic questions. "
        "You can ask questions, and the chatbot will try to answer them."
    )

    # To ignore SSL certificate warnings
    ignore_ssl_warning()

    # Load and clean dataset
    file_path = os.path.abspath("./source_data/raw/intents.json")
    cleaned_data, tags = load_and_clean_dataset(file_path)

    # Display the sidebar menu and handle navigation
    choice = display_menu()

    if choice == "Home":
        handle_home(counter, cleaned_data, tags)
    elif choice == "Conversation History":
        handle_conversation_history()
    elif choice == "About":
        handle_about()

if __name__ == '__main__':
    app_deployment()