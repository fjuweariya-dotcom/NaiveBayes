# import Libraries
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
import time

# Load dataset
df = pd.read_csv(
    r"C:\\Users\\User\\OneDrive\\Desktop\\All notes\\Naive bayes project\\breast cancer\\data.csv")

# Drop the 'id' column and the 'Unnamed: 32' column
df = df.drop(['id', 'Unnamed: 32'], axis=1)
print(f"DataFrame shape after dropping 'id' and 'Unnamed: 32': {df.shape}")

# Confirm the 'diagnosis' column contains binary values (0 and 1)
print("First 5 rows of the DataFrame:")
print(df.head())
print("\nValue counts for 'diagnosis' column:")
print(df['diagnosis'].value_counts())

# Check if diagnosis column is of object type (string) and map to numerical values
# This block is moved BEFORE X and y are defined
if df['diagnosis'].dtype == 'object':
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Verify no more missing values
X = df.drop('diagnosis', axis=1).values
y = df['diagnosis'].values

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
# Initialize the Gaussian Naive Bayes classifier
gnb = GaussianNB()

# Train the classifier and measure training time
start_time = time.time()
gnb.fit(X_train, y_train)
end_time = time.time()
print(f"Training time: {(end_time - start_time):.4f} seconds")

# Make predictions on the test set
y_pred = gnb.predict(X_test)
# Evaluate the classifier's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")  # Sklearn Integrated test.

# Calculate Precision, Recall, F1 for Gaussian NB
precision = precision_score(y_test, y_pred, pos_label=1)
recall = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")
