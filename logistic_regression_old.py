import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Read the CSV files into DataFrames
df_train = pd.read_csv('isear-train.csv')
df_test = pd.read_csv('isear-val.csv')

# Assuming the first column is the emotion label and the second column is the sentence
emotion_column_train = df_train.columns[0]
sentence_column_train = df_train.columns[1]

emotion_column_test = df_test.columns[0]
sentence_column_test = df_test.columns[1]

# Rename columns for clarity
df_train = df_train.rename(columns={emotion_column_train: 'emotion', sentence_column_train: 'processed_sentence'})
df_test = df_test.rename(columns={emotion_column_test: 'emotion', sentence_column_test: 'processed_sentence'})

# Remove any unnecessary columns (adjust indices as needed)
df_train = df_train[['emotion', 'processed_sentence']]
df_test = df_test[['emotion', 'processed_sentence']]

# Display the DataFrames to check the structure
print("Training DataFrame:")
print(df_train.head())

print("Testing DataFrame:")
print(df_test.head())

# Handle missing values: replace NaN with an empty string in 'processed_sentence'
df_train['processed_sentence'].fillna('', inplace=True)
df_test['processed_sentence'].fillna('', inplace=True)

# Ensure 'processed_sentence' columns are of type str
df_train['processed_sentence'] = df_train['processed_sentence'].astype(str)
df_test['processed_sentence'] = df_test['processed_sentence'].astype(str)

# Handle missing values in 'emotion' by removing those rows
df_train = df_train.dropna(subset=['emotion'])
df_test = df_test.dropna(subset=['emotion'])


#calculating the frequency of each class for the training set
class_counts_train = df_train['emotion'].value_counts()
print("Class Frequencies (Training Set):")
print(class_counts_train)



# Convert the columns to lists of strings
train_documents = df_train['processed_sentence'].tolist()
train_labels = df_train['emotion'].tolist()

test_documents = df_test['processed_sentence'].tolist()
test_labels = df_test['emotion'].tolist()

# Additional check for NaN values in the lists
print("\nChecking for NaN values in train_documents:", any(pd.isna(doc) for doc in train_documents))
print("Checking for NaN values in test_documents:", any(pd.isna(doc) for doc in test_documents))
print("Checking for NaN values in train_labels:", any(pd.isna(label) for label in train_labels))
print("Checking for NaN values in test_labels:", any(pd.isna(label) for label in test_labels))

# Remove any remaining NaN values within the documents and labels
train_documents = [doc if isinstance(doc, str) else "" for doc in train_documents]
train_labels = [label if isinstance(label, str) else "" for label in train_labels]

test_documents = [doc if isinstance(doc, str) else "" for doc in test_documents]
test_labels = [label if isinstance(label, str) else "" for label in test_labels]

# Create a TfidfVectorizer instance with a maximum number of features
tfidf_vectorizer = TfidfVectorizer(max_features=1000)

# Fit the TF-IDF vectorizer on the training data and transform both train and test sets
X_train = tfidf_vectorizer.fit_transform(train_documents)
X_test = tfidf_vectorizer.transform(test_documents)

y_train = train_labels
y_test = test_labels

# Display the shape of the TF-IDF feature matrix
print("\nTF-IDF Feature Matrix Shape (Train):", X_train.shape)
print("TF-IDF Feature Matrix Shape (Test):", X_test.shape)

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
