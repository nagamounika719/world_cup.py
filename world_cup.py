# Import necessary libraries
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
nltk.download('wordnet')

# Load the dataset (assuming you have a CSV file with 'text' and 'label' columns)
data = pd.read_csv("C:\\Users\\nagar\\OneDrive\\Desktop\\fifa_world_cup_2022_tweets.csv")

# Text cleaning and preprocessing
def clean_text(Tweet):
    # Remove HTML tags
    Tweet = re.sub(r'<.*?>', '', Tweet)
    # Remove special characters, numbers, and punctuations
    Tweet = re.sub(r'[^a-zA-Z]', ' ', Tweet)
    # Convert to lowercase
    Tweet = Tweet.lower()
    # Tokenization (split the text into words)
    words = Tweet.split()
    # Remove stopwords
    words = [word for word in words if word not in set(stopwords.words('english'))]
    # Lemmatization (or you can use stemming if needed)
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

data['cleaned_text'] = data['Tweet'].apply(clean_text)


# Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(data['cleaned_text'])
y = data['Sentiment']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the models
# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# SVM
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Model evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    confusion = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, precision, recall, confusion, report

# Evaluate models
nb_accuracy, nb_precision, nb_recall, nb_confusion, nb_report = evaluate_model(nb_model, X_test, y_test)
svm_accuracy, svm_precision, svm_recall, svm_confusion, svm_report = evaluate_model(svm_model, X_test, y_test)
rf_accuracy, rf_precision, rf_recall, rf_confusion, rf_report = evaluate_model(rf_model, X_test, y_test)

print("Naive Bayes Accuracy:", nb_accuracy)
print("Naive Bayes Precision:", nb_precision)
print("Naive Bayes Recall:", nb_recall)

print("SVM Accuracy:", svm_accuracy)
print("SVM Precision:", svm_precision)
print("SVM Recall:", svm_recall)

print("Random Forest Accuracy:", rf_accuracy)
print("Random Forest Precision:", rf_precision)
print("Random Forest Recall:", rf_recall)

# Sort words by TF-IDF scores and select the top 20
tfidf_scores = dict(zip(tfidf_vectorizer.get_feature_names_out(), np.array(X.sum(axis=0)).ravel()))
sorted_tfidf_scores = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:20]
top_words = [word for word, _ in sorted_tfidf_scores]

# Create a filtered vocabulary for the word cloud
filtered_vocab = {word: tfidf_vectorizer.vocabulary_[word] for word in top_words}

wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(filtered_vocab)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()