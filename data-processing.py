import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

train_sentences=[]
train_labels=[]
test_sentences=[]
test_labels=[]
with open("isear-train.csv", "r") as file:
    lines = file.readlines()
    for line in lines:
        parts = line.strip().split(',', 1)
        # Check if there are at least two parts after splitting
        if len(parts) != 2:
            # Skip this line if it doesn't contain a comma
            continue

        label = parts[0].strip().strip('"').lower()  # Normalize label by removing double quotes and converting to lowercase
        train_labels.append(label)
        text = parts[1] 
        train_sentences.append(text)

with open("isear-val.csv", "r") as file:
    lines = file.readlines()
    for line in lines:
        parts = line.strip().split(',', 1)
        # Check if there are at least two parts after splitting
        if len(parts) != 2:
            # Skip this line if it doesn't contain a comma
            continue

        label = parts[0].strip().strip('"').lower()  # Normalize label by removing double quotes and converting to lowercase
        test_labels.append(label)
        text = parts[1] 
        test_sentences.append(text)

#convert lists to dataframes for better visualization 
df_train=pd.DataFrame({'emotion':train_labels, 'sentences':train_sentences}) 
df_test=pd.DataFrame({'emotion':test_labels, 'sentences':test_sentences}) 

# Calculate class frequencies for the training set
class_counts_train = df_train['emotion'].value_counts()

print(class_counts_train.head(20))

# Set a threshold for minimum frequency
threshold = 5

# Filter out classes with frequencies less than or equal to the threshold
filtered_class_counts_train = class_counts_train[class_counts_train > threshold]

# Display the filtered class frequencies
print("\nFiltered Class Frequencies (Training Set):")
print(filtered_class_counts_train)

# Get the number of unique classes
num_unique_classes = class_counts_train.shape
print(f"\nNumber of unique classes: {num_unique_classes}")

# Get the total number of rows in the original class counts
total_rows_class_counts_train = class_counts_train.sum()
print(f"\nTotal number of rows in class_counts_train: {total_rows_class_counts_train}")