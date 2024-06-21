import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Read the CSV file into a DataFrame
df_train = pd.read_csv('isear-train.csv')
df_test = pd.read_csv('isear-val.csv')

# Assuming the first column is the emotion label and the second column is the sentence
train_emotion_column = df_train.columns[0]
train_sentence_column = df_train.columns[1]

test_emotion_column = df_test.columns[0]
test_sentence_column = df_test.columns[1]

# Rename the columns accordingly for training data
df_train = df_train.rename(columns={train_emotion_column: 'emotion', train_sentence_column: 'processed_sentence'})
df_train = df_train[['emotion', 'processed_sentence']]  # Keep only the relevant columns

# Rename the columns accordingly for test data
df_test = df_test.rename(columns={test_emotion_column: 'emotion', test_sentence_column: 'processed_sentence'})
df_test = df_test[['emotion', 'processed_sentence']]  # Keep only the relevant columns

# Display the DataFrame to check the structure
print("Original Training DataFrame:")
print(df_train.head())

print("\nOriginal Test DataFrame:")
print(df_test.head())

#df_train['processed_sentence'].fillna('', inplace=True)  # Replace NaN values with empty string
#df_test['processed_sentence'].fillna('', inplace=True)  # Replace NaN values with empty string

# Handle missing values: drop rows with NaN values in 'emotion' or 'processed_sentence'
df_train = df_train.dropna(subset=['emotion', 'processed_sentence'])
df_test = df_test.dropna(subset=['emotion', 'processed_sentence'])

# Ensure the processed_sentence column is of type str
df_train['processed_sentence'] = df_train['processed_sentence'].astype(str)
df_test['processed_sentence'] = df_train['processed_sentence'].astype(str)

# Verify the data type of the column
print("\nData Types After Conversion:")
print(df_train.dtypes)

# Convert the column to a list of strings
train_documents = df_train['processed_sentence'].tolist()
train_labels = df_train['emotion']

test_documents = df_test['processed_sentence'].tolist()
test_labels = df_test['emotion']

# Additional check for NaN values in the lists
print("\nChecking for NaN values in train_documents:", any(pd.isna(doc) for doc in train_documents))
print("Checking for NaN values in test_documents:", any(pd.isna(doc) for doc in test_documents))

# Remove any remaining NaN values in train and test documents
train_documents = [doc if isinstance(doc, str) else "" for doc in train_documents]
test_documents = [doc if isinstance(doc, str) else "" for doc in test_documents]


# Create a TfidfVectorizer instance without max_features
tfidf_vectorizer = TfidfVectorizer()  # max_features parameter can be adjusted as needed

# Fit and transform the text data into a TF-IDF feature matrix
X_train = tfidf_vectorizer.fit_transform(train_documents)
y_train = train_labels

X_test = tfidf_vectorizer.transform(test_documents)
y_test = test_labels

# Display the shape of the TF-IDF feature matrix
print("\nTF-IDF Feature Matrix Shape:", X_train.shape)
print("\nTF-IDF Feature Matrix Shape:", X_test.shape)


#TF-IDF Feature Matrix Shape: (10699, 7487)
#10699: number of lines in train data
#7487: number of vocab after reduction

# Train a logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)



