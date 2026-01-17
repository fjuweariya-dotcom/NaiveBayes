# Import Libraries
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import Binarizer
import time

# --- 1. Data Loading and Preprocessing ---
print("--- Data Loading and Preprocessing ---")

# Load dataset
df = pd.read_csv(r"C:\\Users\\User\\OneDrive\\Desktop\\All notes\\Naive bayes project\\breast cancer\\data.csv")


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

print("\nData preprocessing complete. Splitting data into train/test sets.")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# --- 2. Gaussian Naive Bayes ---
print("\n--- Gaussian Naive Bayes ---")

gnb = GaussianNB()

# Train the classifier and measure training time
start_time_gnb = time.time()
gnb.fit(X_train, y_train)
end_time_gnb = time.time()
print(
    f"Training time (GaussianNB): {(end_time_gnb - start_time_gnb):.4f} seconds")

# Make predictions on the test set
y_pred_gnb = gnb.predict(X_test)

# Evaluate the classifier's accuracy
accuracy_gnb = accuracy_score(y_test, y_pred_gnb)
precision_gnb = precision_score(y_test, y_pred_gnb, pos_label=1)
recall_gnb = recall_score(y_test, y_pred_gnb, pos_label=1)
f1_gnb = f1_score(y_test, y_pred_gnb, pos_label=1)

print(f"Accuracy (GaussianNB): {accuracy_gnb * 100:.2f}%")
print(f"Precision (GaussianNB): {precision_gnb * 100:.2f}%")
print(f"Recall (GaussianNB): {recall_gnb * 100:.2f}%")
print(f"F1 Score (GaussianNB): {f1_gnb * 100:.2f}%")

# --- 3. Bernoulli Naive Bayes ---
print("\n--- Bernoulli Naive Bayes (with binarized features) ---")

# Binarize features for Bernoulli Naive Bayes
feature_means = X_train.mean(axis=0)

X_train_binarized = np.zeros_like(X_train, dtype=int)
X_test_binarized = np.zeros_like(X_test, dtype=int)

num_features = X_train.shape[1]
for i in range(num_features):
    binarizer = Binarizer(threshold=feature_means[i])
    X_train_binarized[:, i] = binarizer.transform(
        X_train[:, i].reshape(-1, 1)).flatten()
    X_test_binarized[:, i] = binarizer.transform(
        X_test[:, i].reshape(-1, 1)).flatten()

print("Features binarized successfully for BernoulliNB and MultinomialNB.")
print(f"Shape of X_train_binarized: {X_train_binarized.shape}")
print(f"Shape of X_test_binarized: {X_test_binarized.shape}")

bnb = BernoulliNB()

# Train the classifier on the binarized data and measure training time
start_time_bnb = time.time()
bnb.fit(X_train_binarized, y_train)
end_time_bnb = time.time()
print(
    f"Training time (BernoulliNB): {(end_time_bnb - start_time_bnb):.4f} seconds")

# Make predictions on the binarized test set
y_pred_bnb = bnb.predict(X_test_binarized)

# Evaluate the classifier's accuracy
accuracy_bnb = accuracy_score(y_test, y_pred_bnb)
precision_bnb = precision_score(y_test, y_pred_bnb, pos_label=1)
recall_bnb = recall_score(y_test, y_pred_bnb, pos_label=1)
f1_bnb = f1_score(y_test, y_pred_bnb, pos_label=1)

print(f"Accuracy (BernoulliNB): {accuracy_bnb * 100:.2f}%")
print(f"Precision (BernoulliNB): {precision_bnb * 100:.2f}%")
print(f"Recall (BernoulliNB): {recall_bnb * 100:.2f}%")
print(f"F1 Score (BernoulliNB): {f1_bnb * 100:.2f}%")

# --- 4. Multinomial Naive Bayes ---
print("\n--- Multinomial Naive Bayes (with binarized features) ---")

mnb = MultinomialNB()

# Train the classifier on the binarized data and measure training time
start_time_mnb = time.time()
mnb.fit(X_train_binarized, y_train)
end_time_mnb = time.time()
print(
    f"Training time (MultinomialNB): {(end_time_mnb - start_time_mnb):.4f} seconds")

# Make predictions on the binarized test set
y_pred_mnb = mnb.predict(X_test_binarized)

# Evaluate the classifier's accuracy
accuracy_mnb = accuracy_score(y_test, y_pred_mnb)
precision_mnb = precision_score(y_test, y_pred_mnb, pos_label=1)
recall_mnb = recall_score(y_test, y_pred_mnb, pos_label=1)
f1_mnb = f1_score(y_test, y_pred_mnb, pos_label=1)

print(f"Accuracy (MultinomialNB): {accuracy_mnb * 100:.2f}%")
print(f"Precision (MultinomialNB): {precision_mnb * 100:.2f}%")
print(f"Recall (MultinomialNB): {recall_mnb * 100:.2f}%")
print(f"F1 Score (MultinomialNB): {f1_mnb * 100:.2f}%")
