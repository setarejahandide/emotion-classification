# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report
# from sklearn.preprocessing import LabelEncoder
# from scipy.sparse import csr_matrix, hstack
# import nltk
# from nltk import pos_tag
# from collections import Counter
#
# # Ensure you have the necessary NLTK data
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
#
# train_sentences = []
# train_labels = []
# test_sentences = []
# test_labels = []
#
# with open("isear-train.csv", "r") as file:
#     lines = file.readlines()
#     for line in lines:
#         parts = line.strip().split(',', 1)
#         if len(parts) != 2:
#             continue
#         label = parts[0].strip().strip('"').lower()
#         train_labels.append(label)
#         text = parts[1]
#         train_sentences.append(text)
#
# with open("isear-val.csv", "r") as file:
#     lines = file.readlines()
#     for line in lines:
#         parts = line.strip().split(',', 1)
#         if len(parts) != 2:
#             continue
#         label = parts[0].strip().strip('"').lower()
#         test_labels.append(label)
#         text = parts[1]
#         test_sentences.append(text)
#
# # Convert lists to DataFrame for better visualization
# df_train = pd.DataFrame({'emotion': train_labels, 'sentences': train_sentences})
# print(df_train.head())
#
# # Create a TfidfVectorizer instance with a maximum number of features
# tfidf_vectorizer = TfidfVectorizer()
#
# # Fit the TF-IDF vectorizer on the training data and transform both train and test sets
# X_train_tfidf = tfidf_vectorizer.fit_transform(train_sentences)
# X_test_tfidf = tfidf_vectorizer.transform(test_sentences)
#
# y_train = train_labels
# y_test = test_labels
#
# # Function to extract POS tags
# def pos_tagging(sentences):
#     pos_tags = [pos_tag(nltk.word_tokenize(sentence)) for sentence in sentences]
#     return pos_tags
#
# # Convert POS tags to feature counts
# def pos_tag_features(pos_tags, feature_names):
#     pos_counts = [Counter(tag for word, tag in tags) for tags in pos_tags]
#     pos_df = pd.DataFrame(pos_counts).fillna(0).astype(int)
#     pos_df = pos_df.reindex(columns=feature_names, fill_value=0)
#     return csr_matrix(pos_df)
#
# # Extract POS tags
# train_pos_tags = pos_tagging(train_sentences)
# test_pos_tags = pos_tagging(test_sentences)
#
# # Get all unique POS tags from the training data
# all_pos_tags = set(tag for tags in train_pos_tags for word, tag in tags)
#
# # Convert POS tags to features
# X_train_pos = pos_tag_features(train_pos_tags, all_pos_tags)
# X_test_pos = pos_tag_features(test_pos_tags, all_pos_tags)
#
# # Concatenate TF-IDF features with POS tag features
# X_train_combined = hstack([X_train_tfidf, X_train_pos]).tocsr()
# X_test_combined = hstack([X_test_tfidf, X_test_pos]).tocsr()
#
# # Train a logistic regression model
# model = LogisticRegression(max_iter=2000)
# model.fit(X_train_combined, y_train)
#
# # Make predictions on the test set
# y_pred = model.predict(X_test_combined)
#
# # Evaluate the model
# report = classification_report(y_test, y_pred, output_dict=True)
#
# # Get the average f-score for all the classes
# average_fscore = report['macro avg']['f1-score']
#
# # Print the full classification report
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))
#
# # Print the average f-score
# print("\nAverage F-Score for All Classes:", average_fscore)
#
# # Extract and print accuracy and f-score for each class
# print("\nAccuracy and F-Score for Each Class:")
# for emotion in report.keys():
#     if emotion not in ('accuracy', 'macro avg', 'weighted avg'):
#         print(f"Class: {emotion}")
#         print(f"  Accuracy: {report[emotion]['precision']}")
#         print(f"  F-Score: {report[emotion]['f1-score']}")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, hstack
import nltk
from nltk import pos_tag
from collections import Counter

# Ensure you have the necessary NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

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
tfidf_vectorizer = TfidfVectorizer()

# Fit the TF-IDF vectorizer on the training data and transform both train and test sets
X_train_tfidf = tfidf_vectorizer.fit_transform(train_sentences)
X_test_tfidf = tfidf_vectorizer.transform(test_sentences)

y_train = train_labels
y_test = test_labels

# Function to extract POS tags
def pos_tagging(sentences):
    pos_tags = [pos_tag(nltk.word_tokenize(sentence)) for sentence in sentences]
    return pos_tags


 

# Convert POS tags to feature counts
def pos_tag_features(pos_tags, feature_names):
    pos_counts = [Counter(tag for word, tag in tags) for tags in pos_tags]
    pos_df = pd.DataFrame(pos_counts).fillna(0).astype(int)
    pos_df = pos_df.reindex(columns=feature_names, fill_value=0)
    return csr_matrix(pos_df)

# Extract POS tags
train_pos_tags = pos_tagging(train_sentences)
test_pos_tags = pos_tagging(test_sentences)

# Get all unique POS tags from the training data
all_pos_tags = set(tag for tags in train_pos_tags for word, tag in tags)

# Convert POS tags to features
X_train_pos = pos_tag_features(train_pos_tags, all_pos_tags)
X_test_pos = pos_tag_features(test_pos_tags, all_pos_tags)

# Concatenate TF-IDF features with POS tag features
X_train_combined = hstack([X_train_tfidf, X_train_pos]).tocsr()
X_test_combined = hstack([X_test_tfidf, X_test_pos]).tocsr()

# Train a logistic regression model with combined features
model_combined = LogisticRegression(max_iter=1000)
model_combined.fit(X_train_combined, y_train)

# Make predictions on the test set
y_pred_combined = model_combined.predict(X_test_combined)

# Evaluate the combined model
report_combined = classification_report(y_test, y_pred_combined, output_dict=True)

# Train a logistic regression model with only TF-IDF features
model_tfidf = LogisticRegression(max_iter=1000)
model_tfidf.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred_tfidf = model_tfidf.predict(X_test_tfidf)

# Evaluate the TF-IDF model
report_tfidf = classification_report(y_test, y_pred_tfidf, output_dict=True)

# Train a logistic regression model with only POS features
model_pos = LogisticRegression(max_iter=2000)
model_pos.fit(X_train_pos, y_train)

# Make predictions on the test set
y_pred_pos = model_pos.predict(X_test_pos)

# Evaluate the POS model
report_pos = classification_report(y_test, y_pred_pos, output_dict=True)

# Print the comparison of the three models
print("\nClassification Report - Combined Features:")
print(classification_report(y_test, y_pred_combined))

print("\nClassification Report - TF-IDF Features Only:")
print(classification_report(y_test, y_pred_tfidf))

print("\nClassification Report - POS Features Only:")
print(classification_report(y_test, y_pred_pos))

# Print the average f-score for the three models
print("\nAverage F-Score for All Classes - Combined Features:", report_combined['macro avg']['f1-score'])
print("Average F-Score for All Classes - TF-IDF Features Only:", report_tfidf['macro avg']['f1-score'])
print("Average F-Score for All Classes - POS Features Only:", report_pos['macro avg']['f1-score'])



