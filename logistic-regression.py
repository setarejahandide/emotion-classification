#original vocabulary 15450
#take a small sample of the training data
import pandas as pd

# Load your full dataset
data = pd.read_csv('isear-train.csv')

# Sample 10% of the data for initial experiments
# sampled_data = data.sample(frac=0.1, random_state=42)

print(data.head())
df = pd.DataFrame(data)
print(df.head())
# Display the shape of the sampled data to verify
#print(sampled_data.shape)

#-------------------------------------------





