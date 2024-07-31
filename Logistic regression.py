import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from scipy.sparse import csr_matrix, hstack

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
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))

# Fit the TF-IDF vectorizer on the training data and transform both train and test sets
X_train_tfidf = tfidf_vectorizer.fit_transform(train_sentences)
X_test_tfidf = tfidf_vectorizer.transform(test_sentences)

y_train = train_labels
y_test = test_labels

# Function to extract punctuation counts
def count_punctuation(sentences, punctuation):
    return [sentence.count(punctuation) for sentence in sentences]

# Extract punctuation features
train_exclamation = count_punctuation(train_sentences, '!')
train_question = count_punctuation(train_sentences, '?')
train_period = count_punctuation(train_sentences, '.')
test_exclamation = count_punctuation(test_sentences, '!')
test_question = count_punctuation(test_sentences, '?')
test_period = count_punctuation(test_sentences, '.')

# Convert punctuation features to sparse matrix format
X_train_punct = csr_matrix([train_exclamation, train_question, train_period]).transpose()
X_test_punct = csr_matrix([test_exclamation, test_question, test_period]).transpose()

# Concatenate TF-IDF features with punctuation features for "joy" and "sadness" labels
X_train_joy_sadness = hstack([X_train_tfidf, X_train_punct]).tocsr()
X_test_joy_sadness = hstack([X_test_tfidf, X_test_punct]).tocsr()

joy_sadness_labels = ['joy', 'sadness']
joy_sadness_train_indices = [i for i, label in enumerate(train_labels) if label in joy_sadness_labels]
joy_sadness_test_indices = [i for i, label in enumerate(test_labels) if label in joy_sadness_labels]

X_train_joy_sadness_subset = X_train_joy_sadness[joy_sadness_train_indices]
y_train_joy_sadness_subset = [train_labels[i] for i in joy_sadness_train_indices]

# Train a general logistic regression model using only TF-IDF features
general_model = LogisticRegression(max_iter=1000)
general_model.fit(X_train_tfidf, y_train)

# Train a specialized logistic regression model using both TF-IDF and punctuation features
# Only for "joy" and "sadness" labels
specialized_model = LogisticRegression(max_iter=1000)
specialized_model.fit(X_train_joy_sadness_subset, y_train_joy_sadness_subset)

# Make predictions on the test set using the general model
general_pred = general_model.predict(X_test_tfidf)

# Make predictions on the test set using the specialized model for "joy" and "sadness"
X_test_joy_sadness_subset = X_test_joy_sadness[joy_sadness_test_indices]
joy_sadness_pred = specialized_model.predict(X_test_joy_sadness_subset)

# Combine predictions: use specialized model's predictions for "joy" and "sadness" labels
final_pred = general_pred.tolist()
for i, idx in enumerate(joy_sadness_test_indices):
    final_pred[idx] = joy_sadness_pred[i]

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
