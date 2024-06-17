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

# Display the DataFrame to check the structure
print("Original DataFrame:")
print(df.head())



# If emotion and sentence are already in separate columns, just rename them accordingly
df = df.rename(columns={emotion_column: 'emotion', sentence_column: 'processed_sentence'})

 #Drop the original sentence column if no longer needed
df = df.drop(columns=[sentence_column])

# Reorder the columns to have emotion first and processed_sentence second
df = df[['emotion', 'processed_sentence']]

# Display the updated DataFrame
print("\nProcessed DataFrame:")
print(df.head())

# Create a TfidfVectorizer instance
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Adjust max_features as needed

# Fit and transform the text data into a TF-IDF feature matrix
X = tfidf_vectorizer.fit_transform(df['processed_sentence'])
y = df['emotion']

# Display the shape of the TF-IDF feature matrix
print("\nTF-IDF Feature Matrix Shape:", X.shape)
