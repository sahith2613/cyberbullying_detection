from flask import Flask, render_template, request, redirect, url_for, session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
import string
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import joblib

app = Flask(__name__)

# Load the trained model and tokenizer
model = load_model('model/cyberbullying_model.h5')
tokenizer = joblib.load('model/tokenizer.pkl')

max_sequence_length = 100

# Ensure the necessary nltk resources are downloaded
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

def tokenize_remove_punctuation(text):
    clean_text = []
    text = text.split(" ")
    for word in text:
        word = list(word)
        new_word = []
        for c in word:
            if c not in string.punctuation:
                new_word.append(c)
        word = "".join(new_word)
        clean_text.append(word)
    return clean_text

def remove_stopwords(text):
    stopwords_list = set(stopwords.words('english'))
    return [word for word in text if word not in stopwords_list]

def pos_tagging(text):
    try:
        return nltk.pos_tag(text)
    except Exception as e:
        print(e)
        return []

def get_wordnet(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def clean_text(text):
    text = str(text).lower()
    text = tokenize_remove_punctuation(text)
    text = [word for word in text if not any(c.isdigit() for c in word)]
    text = remove_stopwords(text)
    text = [t for t in text if len(t) > 0]
    pos_tags = pos_tagging(text)
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet(t[1])) for t in pos_tags]
    text = [t for t in text if len(t) > 1]
    return " ".join(text)

def predict_cyberbullying(text, tokenizer, model, max_sequence_length):
    clean_text_input = clean_text(text)
    sequence = tokenizer.texts_to_sequences([clean_text_input])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)
    prediction = model.predict(padded_sequence)
    return 'Cyberbullying' if prediction >= 0.5 else 'Non-cyberbullying'

app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# Mock user database (replace with your actual authentication mechanism)
users = {
    'sahith': '12345678',
    'harsha': 'harsha',
    'karthik':'karthik',
}

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']

    if username in users and users[username] == password:
        # Successful login, store username in session
        session['username'] = username
        return redirect(url_for('detect_cyberbullying'))  # Redirect to detection page
    else:
        # Invalid credentials, redirect back to login page with error message
        return render_template('login.html', error='Invalid credentials. Please try again.')

@app.route('/detect', methods=['GET', 'POST'])
def detect_cyberbullying():
    if 'username' not in session:
        return redirect(url_for('home'))

    if request.method == 'POST':
        comment = request.form['comment']
        result = predict_cyberbullying(comment, tokenizer, model, max_sequence_length)
        return render_template('detect.html', prediction=result)

    return render_template('detect.html')

@app.route('/logout')
def logout():
    # Clear the session and redirect to login page
    session.pop('username', None)
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)