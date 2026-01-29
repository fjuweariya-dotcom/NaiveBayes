# import Libraries
import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import Binarizer  # Import Binarizer
import time

# Load dataset
df = pd.read_csv(
    r"C:\Users\User\OneDrive\Desktop\All notes\Naive bayes project\breast cancer\data.csv"'')

# Drop the 'id' column and the 'Unnamed: 32' column
df = df.drop(['id', 'Unnamed: 32'], axis=1)
print(f"DataFrame shape after dropping 'id' and 'Unnamed: 32': {df.shape}")

# Confirm the 'diagnosis' column contains binary values (0 and 1)
print("First 5 rows of the DataFrame:")
print(df.head())
print("\nValue counts for 'diagnosis' column:")
print(df['diagnosis'].value_counts())

# Check if diagnosis column is of object type (string) and map to numerical values
if df['diagnosis'].dtype == 'object':
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Separate features (X) and target (y)
X = df.drop('diagnosis', axis=1).values
y = df['diagnosis'].values

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# --- Binarize features for Bernoulli Naive Bayes ---
# Calculate the mean for each feature (column) in the X_train dataset.
feature_means = X_train.mean(axis=0)

X_train_binarized = np.zeros_like(X_train, dtype=int)
X_test_binarized = np.zeros_like(X_test, dtype=int)

num_features = X_train.shape[1]
for i in range(num_features):
    # Create an instance of Binarizer, setting its threshold to the mean of that specific feature.
    binarizer = Binarizer(threshold=feature_means[i])

    # Apply binarization to X_train and X_test
    X_train_binarized[:, i] = binarizer.transform(
        X_train[:, i].reshape(-1, 1)).flatten()
    X_test_binarized[:, i] = binarizer.transform(
        X_test[:, i].reshape(-1, 1)).flatten()

print("\nFeatures binarized successfully for BernoulliNB.")
print(f"Shape of X_train_binarized: {X_train_binarized.shape}")
print(f"Shape of X_test_binarized: {X_test_binarized.shape}")

# Initialize the Bernoulli Naive Bayes classifier
bnb = BernoulliNB()

# Train the classifier on the binarized data and measure training time
start_time = time.time()
bnb.fit(X_train_binarized, y_train)
end_time = time.time()
print(
    f"\nTraining time (BernoulliNB with binarized features): {(end_time - start_time):.4f} seconds")

# Make predictions on the binarized test set
y_pred = bnb.predict(X_test_binarized)

# Evaluate the classifier's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy (BernoulliNB with binarized features): {accuracy * 100:.2f}%")

# Calculate Precision, Recall, F1 for Bernoulli NB with binarized features
precision = precision_score(y_test, y_pred, pos_label=1)
recall = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)
print(
    f"Precision (BernoulliNB with binarized features): {precision * 100:.2f}%")
print(f"Recall (BernoulliNB with binarized features): {recall * 100:.2f}%")
print(f"F1 Score (BernoulliNB with binarized features): {f1 * 100:.2f}%")
