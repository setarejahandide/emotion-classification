import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np


train_sentences=[]
train_labels=[]
test_sentences=[]
test_labels=[]

#extracting the sentences and labels for the train data
with open("isear-train.csv", "r") as file:
    lines = file.readlines()
    for line in lines:
        parts = line.strip().split(',', 1)
        # Check if there are at least two parts after splitting
        if len(parts) != 2:
            # Skip this line if it doesn't contain a comma
            continue

        label = parts[0].strip().strip('"').lower()  # Normalize label by removing double quotes and converting to lowercase
        #merge the class shame with guilt
        # if label=='shame':
            #label='guilt'

        train_labels.append(label)
        text = parts[1] 
        train_sentences.append(text)

#extracting the sentences and labels for the test data
with open("isear-val.csv", "r") as file:
    lines = file.readlines()
    for line in lines:
        parts = line.strip().split(',', 1)
        # Check if there are at least two parts after splitting
        if len(parts) != 2:
            # Skip this line if it doesn't contain a comma
            continue

        label = parts[0].strip().strip('"').lower()  # Normalize label by removing double quotes and converting to lowercase
        
        #merging the class shame with guilt
        # if label=='shame':
            #label='guilt'
        test_labels.append(label)
        text = parts[1] 
        test_sentences.append(text)





# Train a Word2Vec model on the training data
from gensim.models import Word2Vec
word2vec_model = Word2Vec(sentences=train_sentences, vector_size=100, window=5, min_count=1, workers=4)

# Get the vocabulary size
vocab_size = len(word2vec_model.wv)

print("Vocabulary Size:", vocab_size)


# Function to convert a sentence to a vector by averaging its word vectors
def sentence_to_vector(sentence, model):
    words = [word for word in sentence if word in model.wv]
    if not words:
        return np.zeros(model.vector_size)
    return np.mean(model.wv[words], axis=0)

# Convert train and test sentences to vectors
X_train = np.array([sentence_to_vector(sentence, word2vec_model) for sentence in train_sentences])
X_test = np.array([sentence_to_vector(sentence, word2vec_model) for sentence in test_sentences])

y_train = train_labels
y_test = test_labels

# Display the shape of the Word2Vec feature matrix
print("\nWord2Vec Feature Matrix Shape (Train):", X_train.shape)
print("Word2Vec Feature Matrix Shape (Test):", X_test.shape)


# Train a logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)




# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
report = classification_report(y_test, y_pred, output_dict=True)

# Get the average f-score for all the classes
average_fscore = report['macro avg']['f1-score']

# Print the full classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Print the average f-score
print("\nAverage F-Score for All Classes:", average_fscore)

# Extract and print accuracy and f-score for each class
print("\nAccuracy and F-Score for Each Class:")
for emotion in report.keys():
    if emotion not in ('accuracy', 'macro avg', 'weighted avg'):
        print(f"Class: {emotion}")
        print(f"  Accuracy: {report[emotion]['precision']}")
        print(f"  F-Score: {report[emotion]['f1-score']}")

        