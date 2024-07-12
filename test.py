import pandas as pd
from sentence_transformers import SentenceTransformer



# Create a DataFrame
#df = pd.DataFrame('isear-train.csv')

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
# Load pre-trained Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode sentences using Sentence-BERT
sentence_embeddings = model.encode(train_sentences)

# Convert embeddings to DataFrame for better visualization
embeddings_df = pd.DataFrame(sentence_embeddings)

# Combine the original DataFrame with the embeddings
result_df = pd.concat([df, embeddings_df], axis=1)

# Display the resulting DataFrame
print(result_df)


