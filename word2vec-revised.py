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


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Tokenization and preprocessing function
def preprocess_text(text):
    # Tokenize
    tokens = word_tokenize(text)
    #Lowercase
    tokens = [word.lower() for word in tokens]
    # Remove punctuation
    #tokens = [word for word in tokens if word.isalnum()]
    
    return tokens
    

# Apply preprocessing to sentences
train_sentences = [preprocess_text(sentence) for sentence in train_sentences]
test_sentences = [preprocess_text(sentence) for sentence in test_sentences]


    





# Train a Word2Vec model on the training data
from gensim.models import Word2Vec
word2vec_model = Word2Vec(sentences=train_sentences, vector_size=100, window=5, min_count=1, workers=4)

# Get the vocabulary size
vocab_size = len(word2vec_model.wv)

print("Vocabulary Size:", vocab_size)

# Print some sample words
sample_words = list(word2vec_model.wv.index_to_key)[:10]  
print("Sample Vocabulary Words:", sample_words)

# Get the dimensions of the word vectors
vector_size = word2vec_model.wv.vector_size
print("Word Vector Dimensions:", vector_size)

# Example: Accessing a word vector and its shape
word_vector = word2vec_model.wv['friend']
print("Word Vector for 'cat':", word_vector)
print("Shape of Word Vector for 'friend':", word_vector.shape)

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


from scipy.spatial.distance import cosine

def similarity(word1, word2, model):
    vec1 = model.wv[word1]
    vec2 = model.wv[word2]
    return 1 - cosine(vec1, vec2)  # Cosine similarity

similarity_score = similarity('brother', 'sister', word2vec_model)
print(f"Similarity between 'king' and 'queen': {similarity_score}")


        