Cyberbullying Detection Web Application

This project is a machine learning-based web application designed to detect cyberbullying in text comments. It uses a pre-trained LSTM model to classify user-provided text as either "Cyberbullying" or "Non-cyberbullying." The application also provides a user authentication system.

Features

Machine Learning Model: An LSTM model trained on text data to detect cyberbullying.

Data Preprocessing: Includes tokenization, stopword removal, POS tagging, and lemmatization.

Web Application: Built using Flask for user interaction.

Authentication: Simple username-password-based login system.

Prediction: Users can input text to determine whether it constitutes cyberbullying.

Project Structure

├── app.py               # Flask web application
├── pr.py                # Model training and preprocessing script
├── templates/           # HTML templates for the web interface
│   ├── login.html       # Login page
│   ├── detect.html      # Cyberbullying detection page
├── model/
│   ├── cyberbullying_model.h5 # Trained LSTM model
│   ├── tokenizer.pkl         # Tokenizer for preprocessing
├── dataset/
│   ├── data1.csv            # Dataset used for training
└── README.md            # Project documentation

Prerequisites

Python 3.7 or later

Virtual environment (optional but recommended)

Setup Instructions

Clone this repository:

git clone https://github.com/sahith2613/cyberbullying-detection.git
cd cyberbullying-detection

Install dependencies:

pip install -r requirements.txt

Download necessary NLTK resources:

import nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

Place the trained model and tokenizer in the model/ directory.

Running the Application

Start the Flask application:

python app.py

Open a web browser and navigate to http://127.0.0.1:5000.

Usage

Log in with one of the predefined credentials:

Username: sahith, Password: 12345678

Enter a comment in the provided input box to check if it's considered cyberbullying.

View the classification result.

Model Training:

To train or retrain the model, use the script pr.py. The script:

Cleans and preprocesses the dataset (data1.csv).

Splits the data into training and testing sets.

Builds and trains an LSTM model.

Saves the model (cyberbullying_model.h5) and tokenizer (tokenizer.pkl).

Technologies Used:

Frontend: HTML

Backend: Flask

Machine Learning: TensorFlow/Keras, scikit-learn

Text Processing: NLTK

Deployment: Localhost or web server

Future Enhancements:

Improve authentication with a database.

Add support for multiple languages.

Deploy the application on a cloud platform.