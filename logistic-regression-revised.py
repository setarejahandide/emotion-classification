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
print(df_train)

# Create a TfidfVectorizer instance with a maximum number of features
tfidf_vectorizer = TfidfVectorizer(max_features=1000)

# Fit the TF-IDF vectorizer on the training data and transform both train and test sets
X_train = tfidf_vectorizer.fit_transform(train_sentences)
X_test = tfidf_vectorizer.transform(test_sentences)

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

        