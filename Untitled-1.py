

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

def predict(input_data):
    if isinstance(input_data, str):
        if input_data.endswith('.csv') and os.path.isfile(input_data):
            with open(input_data, "r") as file:
                lines = file.readlines()
                for line in lines:
                    parts = line.strip().split(',', 1)
                    if len(parts) != 2:
                        continue
                    label = parts[0].strip().strip('"').lower()
                    test_labels.append(label)
                    text = parts[1]
                    test_sentences.append(text)
        else:
            test_sentences.append(input_data)
    
    # Convert lists to dataframes for better visualization
    df_train = pd.DataFrame({'emotion': train_labels, 'sentences': train_sentences})
    
    # Create a TfidfVectorizer instance
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 1))
    
    # Fit the TF-IDF vectorizer and transform both train and test sets
    X_train_tfidf = tfidf_vectorizer.fit_transform(train_sentences)
    X_test_tfidf = tfidf_vectorizer.transform(test_sentences)

    # Convert y_train and y_test to numpy arrays
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)

    # Extract bigrams for specific emotions
    target_labels = ['anger', 'guilt']
    bigram_train_indices = [i for i, label in enumerate(train_labels) if label in target_labels]
    train_sentences_bigrams = [train_sentences[i] for i in bigram_train_indices]

    X_train_bigrams, bigram_vectorizer = extract_bigrams(train_sentences_bigrams)

    X_test_bigrams = bigram_vectorizer.transform(test_sentences)
    test_indices_target = [i for i, label in enumerate(test_labels) if label in target_labels]
    X_test_target_bigrams = X_test_bigrams[test_indices_target]

    if len(bigram_train_indices) > 0 and len(test_indices_target) > 0:
        X_train_combined = hstack([X_train_tfidf[bigram_train_indices], X_train_bigrams]).tocsr()
        X_test_combined = hstack([X_test_tfidf[test_indices_target], X_test_target_bigrams]).tocsr()
    else:
        X_train_combined = csr_matrix((0, X_train_tfidf.shape[1] + X_train_bigrams.shape[1]))
        X_test_combined = csr_matrix((0, X_test_tfidf.shape[1] + X_test_target_bigrams.shape[1]))

    y_train_target_subset = y_train[bigram_train_indices]
    y_test_target_subset = y_test[test_indices_target]

    # Check if there's any data to train and test the model
    if X_train_combined.shape[0] > 0 and X_test_combined.shape[0] > 0:
        specialized_model = LogisticRegression(max_iter=1000)
        specialized_model.fit(X_train_combined, y_train_target_subset)
        target_pred = specialized_model.predict(X_test_combined)
    else:
        target_pred = []

    # Extract punctuation features for specific emotions
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

    X_train_punct = csr_matrix([train_exclamation, train_question, train_period]).transpose()
    X_test_punct = csr_matrix([test_exclamation, test_question, test_period]).transpose()

    if len(joy_sadness_shame_train_indices) > 0 and len(joy_sadness_shame_test_indices) > 0:
        X_train_joy_sadness_shame = hstack([X_train_tfidf[joy_sadness_shame_train_indices], X_train_punct]).tocsr()
        X_test_joy_sadness_shame = hstack([X_test_tfidf[joy_sadness_shame_test_indices], X_test_punct]).tocsr()
    else:
        X_train_joy_sadness_shame = csr_matrix((0, X_train_tfidf.shape[1] + X_train_punct.shape[1]))
        X_test_joy_sadness_shame = csr_matrix((0, X_test_tfidf.shape[1] + X_test_punct.shape[1]))

    if X_train_joy_sadness_shame.shape[0] > 0 and X_test_joy_sadness_shame.shape[0] > 0:
        specialized_model_joy_sadness_shame = LogisticRegression(max_iter=1000)
        specialized_model_joy_sadness_shame.fit(X_train_joy_sadness_shame, y_train[joy_sadness_shame_train_indices])
        joy_sadness_shame_pred = specialized_model_joy_sadness_shame.predict(X_test_joy_sadness_shame)
    else:
        joy_sadness_shame_pred = []

    # Train and predict with the general model if there are samples
    if X_train_tfidf.shape[0] > 0 and X_test_tfidf.shape[0] > 0:
        general_model = LogisticRegression(max_iter=1000)
        general_model.fit(X_train_tfidf, y_train)
        general_pred = general_model.predict(X_test_tfidf)
    else:
        general_pred = []

    final_pred = general_pred.tolist()

    # Replace with predictions from specialized models where applicable
    for i, idx in enumerate(test_indices_target):
        final_pred[idx] = target_pred[i]
    for i, idx in enumerate(joy_sadness_shame_test_indices):
        final_pred[idx] = joy_sadness_shame_pred[i]

    if final_pred:
        report = classification_report(y_test, final_pred, output_dict=True)
        average_fscore = report['macro avg']['f1-score']
        print("\nClassification Report:")
        print(classification_report(y_test, final_pred))
        print("\nAverage F-Score for All Classes:", average_fscore)
    else:
        print("No predictions to report; please check your input data and model training process.")
