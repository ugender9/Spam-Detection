from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load saved model and vectorizer
model = joblib.load('spam_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ""
    message = ""

    if request.method == 'POST':
        message = request.form['message']
        if message.strip() == "":
            result = "Please enter a message."
        else:
            transformed = tfidf.transform([message])
            prediction = model.predict(transformed)[0]
            result = "Spam" if prediction == 1 else "Not Spam"

    return render_template('index.html', result=result, message=message)

if __name__ == '__main__':
    app.run(debug=True)
