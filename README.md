# AICTE Chatbot Using NLP

This project is an basic Chatbot developed using Natural Language Processing (NLP) techniques.

## Table of Contents
- [Introduction](#introduction)
- [Directory Structure](#directory_structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)


## Introduction
The AICTE Chatbot is an intelligent system that leverages NLP to understand and respond to user queries.

## Directory Structure
Ensure your directory structure looks like this:


Project__
├── data
│   ├── raw
│   │   └── intents.json
│   └── cleaned
│       └── intents.json
├── models
│   └── chatbot_model_<timestamp>.pkl
├── src
│   ├── app.py
│   └── Chatbot.py
├── requirements.txt
└── README.md


## Features
- Natural Language Understanding
- Contextual Responses
- Easy Integration
- User-Friendly Interface

## Installation
To install and run the chatbot locally, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/MAHAKAAL-vk/AICTE_Chatbot-using-NLP.git
    ```

2. **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    .venv\Scripts\activate  # On Windows
    ```

3. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. **Place your raw `intents.json` file in the `data/raw/` directory.**

2. **Run the Streamlit app:**
    ```bash
    streamlit run src/app.py
    ```

3. **Interact with the chatbot through the provided interface.**

## Credits
This project was developed by Vikash Kushwaha under an internship program organized by AICTE through Edunet Foundation.