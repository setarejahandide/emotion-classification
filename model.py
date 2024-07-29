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
        #if label=='shame':
            #label='guilt'

        train_labels.append(label)
        text = parts[1] 
        train_sentences.append(text)

def predict(test_file): 
    #extracting the sentences and labels for the test data
    with open(test_file, "r") as file:
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
            
    #convert lists to dataframes for better visualization 
    df_train=pd.DataFrame({'emotion':train_labels, 'sentences':train_sentences}) 
    # print(df_train)
    
    # Create a TfidfVectorizer instance with a maximum number of features
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 1))
    
    #convert lists to dataframes for better visualization 
    #df_train=pd.DataFrame({'emotion':train_labels, 'sentences':train_sentences}) 
    #df_test=pd.DataFrame({'emotion':test_labels, 'sentences':test_sentences}) 
    
    # Calculate class frequencies for the training set
    #class_counts_train = df_train['emotion'].value_counts()
    
    # print(class_counts_train.head(20))
    
    # Set a threshold for minimum frequency
    #threshold = 5
    
    # Filter out classes with frequencies less than or equal to the threshold
    #filtered_class_counts_train = class_counts_train[class_counts_train > threshold]
    
    # Display the filtered class frequencies
    #print("\nFiltered Class Frequencies (Training Set):")
    #print(filtered_class_counts_train)
    
    # Get the number of unique classes
    #num_unique_classes = class_counts_train.shape
    #print(f"\nNumber of unique classes: {num_unique_classes}")
    
    # Fit the TF-IDF vectorizer on the training data and transform both train and test sets
    X_train = tfidf_vectorizer.fit_transform(train_sentences)
    X_test = tfidf_vectorizer.transform(test_sentences)
    
    y_train = train_labels
    y_test = test_labels
    
    # Display the shape of the TF-IDF feature matrix
    # #print("\nTF-IDF Feature Matrix Shape (Train):", X_train.shape)
    #print("TF-IDF Feature Matrix Shape (Test):", X_test.shape)
    
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
#print("\nAccuracy and F-Score for Each Class:")
#for emotion in report.keys():
    #if emotion not in ('accuracy', 'macro avg', 'weighted avg'):
        #print(f"Class: {emotion}")
        #print(f"  Accuracy: {report[emotion]['precision']}")
        #print(f"  F-Score: {report[emotion]['f1-score']}")

def predict_sentence(sentence):
    
    # Create a TfidfVectorizer instance with a maximum number of features
    tfidf_vectorizer = TfidfVectorizer()
    
    
    # Fit the TF-IDF vectorizer on the training data and transform both train and test sets
    X_train = tfidf_vectorizer.fit_transform(train_sentences)
    X_test = tfidf_vectorizer.transform(sentence)
    
    y_train = train_labels
    
    
    # Train a logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    print(y_pred)
    
    

#user part
#print('this program takes a file from you and predicts the emotion associated with every sentence in the file')
#input_file = input('please insert the file')
#predict(input_file)

answer= input('please write a sentence ')
answer=[answer]
predict_sentence(answer)


    
