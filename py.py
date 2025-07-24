import pandas as pd

# Load dataset from URL
import kagglehub
path = kagglehub.dataset_download("uciml/sms-spam-collection-dataset")
print("Path to dataset files:", path)

import os
dataset_file = os.path.join(path, 'spam.csv')

df = pd.read_csv(dataset_file, sep=',', header=0, encoding='latin1')
# Adjust column names to match expected format
df = df.rename(columns={df.columns[0]: 'label', df.columns[1]: 'message'})
df = df[['label', 'message']]
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Convert labels to binary values
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Drop rows with missing messages
df = df.dropna(subset=['message'])

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Vectorize text using TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Function to train and evaluate model
def train_and_evaluate(model, name):
    model.fit(X_train_tfidf, y_train)
    preds = model.predict(X_test_tfidf)
    print(f"\n--- {name} ---")
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Initialize and evaluate models
mnb = MultinomialNB()
lr = LogisticRegression(max_iter=1000)
import joblib

svm = LinearSVC()

train_and_evaluate(mnb, "Naive Bayes")
train_and_evaluate(lr, "Logistic Regression")
train_and_evaluate(svm, "Support Vector Machine")

# Save the trained SVM model and TF-IDF vectorizer
joblib.dump(svm, 'spam_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
def predict_spam(message):
    vectorized = tfidf.transform([message])
    prediction = svm.predict(vectorized)[0]
    return "Spam" if prediction == 1 else "Not Spam"

# Test with sample messages
print(predict_spam("Congratulations! You've won a $1000 gift card. Click here to claim."))  # Likely spam
print(predict_spam("Hey, are we still meeting at 5 PM today?"))  # Likely ham
import joblib

# After training SVM
# Removed duplicate saving of model and vectorizer
