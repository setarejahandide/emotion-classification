import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from scipy.sparse import csr_matrix, hstack
import numpy as np


# Uncomment this part and line 61 to check the keywords impact
# Define emotion keywords
# emotion_keywords = {
#     'anger': ['outrage', 'furious'],
#     'disgust': ['disgusted', 'revolted', 'repulsed'],
#     'fear': ['afraid', 'scared', 'terrified'],
#     'guilt': ['guilty', 'remorseful', 'ashamed'],
#     'joy': ['happy', 'joyful', 'ecstatic'],
#     'sadness': ['sad', 'grief', 'melancholy'],
#     'shame': ['shameful', 'embarrassed', 'humiliated']
# }

# Custom preprocessor function to boost keywords
# def custom_preprocessor(doc):
#     boosted_words = []
#     for emotion, keywords in emotion_keywords.items():
#         for word in keywords:
#             if word in doc:
#                 boosted_words.append(word)
#     return doc + ' ' + ' '.join(boosted_words)




# Read the CSV files into lists
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

def predict_file(test_file): 
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

    #Create a TfidfVectorizer instance without the custom preprocessor
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # , preprocessor=custom_preprocessor)
    
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
print('this program predicts the emotion of sentences. you can either upload a file or a sentence')
answer = input('do you want to upload a file or a sentence? ')
answer=answer.lower().strip()
if answer=='file':
    input_file = input('please insert the file ')
    predict_file(input_file)
elif answer=='sentence':
    sentence= input('please write a sentence ')
    sentence=[sentence]
    predict_sentence(sentence)
else:
    print('this answer is not valid, please try again')


    
