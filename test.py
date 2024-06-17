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

# Display the DataFrame to check the structure
print("Original DataFrame:")
print(df.head())