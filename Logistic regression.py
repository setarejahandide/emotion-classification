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

df_train = pd.DataFrame({'emotion': train_labels, 'sentences': train_sentences})
print(df_train.head())

tfidf_vectorizer = TfidfVectorizer()

X_train_tfidf = tfidf_vectorizer.fit_transform(train_sentences)
X_test_tfidf = tfidf_vectorizer.transform(test_sentences)

y_train = train_labels
y_test = test_labels

def count_punctuation(sentences, punctuation):
    return [sentence.count(punctuation) for sentence in sentences]

train_exclamation = count_punctuation(train_sentences, '!')
train_question = count_punctuation(train_sentences, '?')
train_period = count_punctuation(train_sentences, '.')
test_exclamation = count_punctuation(test_sentences, '!')
test_question = count_punctuation(test_sentences, '?')
test_period = count_punctuation(test_sentences, '.')

X_train_punct = csr_matrix([train_exclamation, train_question, train_period]).transpose()
X_test_punct = csr_matrix([test_exclamation, test_question, test_period]).transpose()

X_train_emotion = hstack([X_train_tfidf, X_train_punct]).tocsr()
X_test_emotion = hstack([X_test_tfidf, X_test_punct]).tocsr()

emotion_labels = ['joy', 'sadness', 'anger', 'shame']
emotion_train_indices = [i for i, label in enumerate(train_labels) if label in emotion_labels]
emotion_test_indices = [i for i, label in enumerate(test_labels) if label in emotion_labels]

X_train_emotion_subset = X_train_emotion[emotion_train_indices]
y_train_emotion_subset = [train_labels[i] for i in emotion_train_indices]

general_model = LogisticRegression(max_iter=1000)
general_model.fit(X_train_tfidf, y_train)

specialized_model = LogisticRegression(max_iter=1000)
specialized_model.fit(X_train_emotion_subset, y_train_emotion_subset)

general_pred = general_model.predict(X_test_tfidf)

X_test_emotion_subset = X_test_emotion[emotion_test_indices]
emotion_pred = specialized_model.predict(X_test_emotion_subset)

final_pred = general_pred.tolist()
for i, idx in enumerate(emotion_test_indices):
    final_pred[idx] = emotion_pred[i]

report = classification_report(y_test, final_pred, output_dict=True)

average_fscore = report['macro avg']['f1-score']

print("\nClassification Report:")
print(classification_report(y_test, final_pred))

print("\nAverage F-Score for All Classes:", average_fscore)

print("\nAccuracy and F-Score for Each Class:")
for emotion in report.keys():
    if emotion not in ('accuracy', 'macro avg', 'weighted avg'):
        print(f"Class: {emotion}")
        print(f"  Accuracy: {report[emotion]['precision']}")
        print(f"  F-Score: {report[emotion]['f1-score']}")
