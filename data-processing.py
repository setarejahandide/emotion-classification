import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Read the CSV file into a DataFrame
df = pd.read_csv('isear-train.csv')

# the first column is the emotion label and the second column is the sentence
emotion_column = df.columns[0]
sentence_column = df.columns[1]



# If emotion and sentence are already in separate columns, just rename them accordingly
df = df.rename(columns={emotion_column: 'emotion', sentence_column: 'processed_sentence'})

df = df.drop(df.columns[[2,3,4,5,6]], axis=1)

# Display the DataFrame to check the structure
print("Original DataFrame:")
print(df.head())

df['processed_sentence'].fillna('', inplace=True)  # Replace NaN values with empty string


# Ensure the processed_sentence column is of type str
df['processed_sentence'] = df['processed_sentence'].astype(str)

# Verify the data type of the column
print("\nData Types After Conversion:")
print(df.dtypes)

# Convert the column to a list of strings
documents = df['processed_sentence'].tolist()
labels = df['emotion']


# Create a TfidfVectorizer instance without max_features
tfidf_vectorizer = TfidfVectorizer()  # max_features parameter can be adjusted as needed

# Fit and transform the text data into a TF-IDF feature matrix
X = tfidf_vectorizer.fit_transform(documents)
y = labels

# Display the shape of the TF-IDF feature matrix
print("\nTF-IDF Feature Matrix Shape:", X.shape)

