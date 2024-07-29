import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from scipy.sparse import csr_matrix, hstack
import numpy as np

# Function to extract punctuation counts
def count_punctuation(sentences, punctuation):
    return [sentence.count(punctuation) for sentence in sentences]

# Function to extract bigrams
def extract_bigrams(sentences):
    vectorizer = TfidfVectorizer(ngram_range=(2, 2))
    return vectorizer.fit_transform(sentences), vectorizer

train_sentences = []
train_labels = []
test_sentences = []
test_labels = []

with open("isear-train.csv", "r") as file:
    lines = file.readlines()
    for line in lines:
        parts = line.strip().split(',', 1)
        if len(parts) != 2:
            continue
        label = parts[0].strip().strip('"').lower()
        train_labels.append(label)
        text = parts[1]
        train_sentences.append(text)

with open("isear-val.csv", "r") as file:
    lines = file.readlines()
    for line in lines:
        parts = line.strip().split(',', 1)
        if len(parts) != 2:
            continue
        label = parts[0].strip().strip('"').lower()
        test_labels.append(label)
        text = parts[1]
        test_sentences.append(text)

# Convert lists to DataFrame for better visualization
df_train = pd.DataFrame({'emotion': train_labels, 'sentences': train_sentences})
print(df_train.head())

# Create a TfidfVectorizer instance with a maximum number of features
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 1))

# Fit the TF-IDF vectorizer on the training data and transform both train and test sets
X_train_tfidf = tfidf_vectorizer.fit_transform(train_sentences)
X_test_tfidf = tfidf_vectorizer.transform(test_sentences)

# Convert y_train and y_test to numpy arrays
y_train = np.array(train_labels)
y_test = np.array(test_labels)

# Extract bigrams for anger and guilt
target_labels = ['anger', 'guilt']
bigram_train_indices = [i for i, label in enumerate(train_labels) if label in target_labels]
train_sentences_bigrams = [train_sentences[i] for i in bigram_train_indices]

X_train_bigrams, bigram_vectorizer = extract_bigrams(train_sentences_bigrams)

# Extract bigrams for the entire test set
X_test_bigrams = bigram_vectorizer.transform(test_sentences)

# Filter indices for anger and guilt in test set
test_indices_target = [i for i, label in enumerate(test_labels) if label in target_labels]
X_test_target_bigrams = X_test_bigrams[test_indices_target]

# Create a combined feature set for anger and guilt
X_train_combined = hstack([X_train_tfidf[bigram_train_indices], X_train_bigrams]).tocsr()
X_test_combined = hstack([X_test_tfidf[test_indices_target], X_test_target_bigrams]).tocsr()

# Filter labels for anger and guilt in train and test sets
y_train_target_subset = y_train[bigram_train_indices]
y_test_target_subset = y_test[test_indices_target]

# Extract punctuation features for joy, sadness and shame
joy_sadness_shame_labels = ['joy', 'sadness', 'shame']
joy_sadness_shame_train_indices = [i for i, label in enumerate(train_labels) if label in joy_sadness_shame_labels]
joy_sadness_shame_test_indices = [i for i, label in enumerate(test_labels) if label in joy_sadness_shame_labels]

train_sentences_joy_sadness_shame = [train_sentences[i] for i in joy_sadness_shame_train_indices]
test_sentences_joy_sadness_shame = [test_sentences[i] for i in joy_sadness_shame_test_indices]

train_exclamation = count_punctuation(train_sentences_joy_sadness_shame, '!')
train_question = count_punctuation(train_sentences_joy_sadness_shame, '?')
train_period = count_punctuation(train_sentences_joy_sadness_shame, '.')
test_exclamation = count_punctuation(test_sentences_joy_sadness_shame, '!')
test_question = count_punctuation(test_sentences_joy_sadness_shame, '?')
test_period = count_punctuation(test_sentences_joy_sadness_shame, '.')

# Convert punctuation features to sparse matrix format
X_train_punct = csr_matrix([train_exclamation, train_question, train_period]).transpose()
X_test_punct = csr_matrix([test_exclamation, test_question, test_period]).transpose()

# Concatenate TF-IDF features with punctuation features for "joy", "sadness" and "shame" labels
X_train_joy_sadness_shame = hstack([X_train_tfidf[joy_sadness_shame_train_indices], X_train_punct]).tocsr()
X_test_joy_sadness_shame = hstack([X_test_tfidf[joy_sadness_shame_test_indices], X_test_punct]).tocsr()

# Train a general logistic regression model using only TF-IDF features
general_model = LogisticRegression(max_iter=1000)
general_model.fit(X_train_tfidf, y_train)

# Train a specialized logistic regression model using both TF-IDF and bigram features
# Only for "anger" and "guilt" labels
specialized_model = LogisticRegression(max_iter=1000)
specialized_model.fit(X_train_combined, y_train_target_subset)

# Train a specialized logistic regression model using both TF-IDF and punctuation features
# Only for "joy", "sadness" and "shame" labels
specialized_model_joy_sadness_shame = LogisticRegression(max_iter=1000)
specialized_model_joy_sadness_shame.fit(X_train_joy_sadness_shame, y_train[joy_sadness_shame_train_indices])

# Make predictions on the test set using the general model
general_pred = general_model.predict(X_test_tfidf)

# Make predictions on the test set using the specialized model for "anger" and "guilt"
target_pred = specialized_model.predict(X_test_combined)

# Make predictions on the test set using the specialized model for "joy", "sadness" and "shame"
joy_sadness_shame_pred = specialized_model_joy_sadness_shame.predict(X_test_joy_sadness_shame)

# Combine predictions: use specialized model's predictions for "anger", "guilt", "joy", "sadness" and "shame" labels
final_pred = general_pred.tolist()
for i, idx in enumerate(test_indices_target):
    final_pred[idx] = target_pred[i]
for i, idx in enumerate(joy_sadness_shame_test_indices):
    final_pred[idx] = joy_sadness_shame_pred[i]

# Evaluate the model
report = classification_report(y_test, final_pred, output_dict=True)

# Get the average f-score for all the classes
average_fscore = report['macro avg']['f1-score']

# Print the full classification report
print("\nClassification Report:")
print(classification_report(y_test, final_pred))

# Print the average f-score
print("\nAverage F-Score for All Classes:", average_fscore)

# Extract and print accuracy and f-score for each class
print("\nAccuracy and F-Score for Each Class:")
for emotion in report.keys():
    if emotion not in ('accuracy', 'macro avg', 'weighted avg'):
        print(f"Class: {emotion}")
        print(f"  Accuracy: {report[emotion]['precision']}")
        print(f"  F-Score: {report[emotion]['f1-score']}")