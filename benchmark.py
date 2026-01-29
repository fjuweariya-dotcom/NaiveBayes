import time
import numpy as np
import pandas as pd
from naivebayes import (   # <-- replace with your actual filename
    BernoulliNaiveBayesFromScratch,
    GaussianNaiveBayesFromScratch,
    MultinomialNaiveBayesFromScratch
)
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB

# --- Manual train/test split ---
def train_test_split_manual(X_data, y_data, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    indices = np.arange(X_data.shape[0])
    np.random.shuffle(indices)
    test_samples = int(X_data.shape[0] * test_size)
    test_indices = indices[:test_samples]
    train_indices = indices[test_samples:]
    X_train, X_test = X_data[train_indices], X_data[test_indices]
    y_train, y_test = y_data[train_indices], y_data[test_indices]
    return X_train, X_test, y_train, y_test

# --- Benchmark function ---
def benchmark_model(model, X_train, y_train, X_test, y_test, name):
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train

    start_pred = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - start_pred

    accuracy = np.mean(y_pred == y_test)

    print(f"\n{name} Results:")
    print(f"  Training Time: {train_time:.4f} sec")
    print(f"  Prediction Time: {pred_time:.4f} sec")
    print(f"  Accuracy: {accuracy*100:.2f}%")

if __name__ == "__main__":
    # --- Load and preprocess dataset ---
    df = pd.read_csv("data.csv")

    # Drop unnecessary columns
    if 'id' in df.columns:
        df = df.drop(['id'], axis=1)
    if 'Unnamed: 32' in df.columns:
        df = df.drop(['Unnamed: 32'], axis=1)

    # Convert diagnosis to binary
    if df['diagnosis'].dtype == 'object':
        df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

    # Continuous copy for Gaussian NB
    df_continuous = df.copy()

    # Binarize features for Bernoulli/Multinomial NB
    for column in df.columns[:-1]:
        mean_val = df[column].mean()
        df[column] = (df[column] > mean_val).astype(int)

    # Feature/target splits
    X = df.drop('diagnosis', axis=1).values
    y = df['diagnosis'].values

    X_cont = df_continuous.drop('diagnosis', axis=1).values
    y_cont = df_continuous['diagnosis'].values

    # Train/test splits
    X_train, X_test, y_train, y_test = train_test_split_manual(X, y)
    X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split_manual(X_cont, y_cont)

    # --- Run benchmarks ---
    benchmark_model(BernoulliNaiveBayesFromScratch(), X_train, y_train, X_test, y_test, "Custom Bernoulli NB")
    benchmark_model(GaussianNaiveBayesFromScratch(), X_train_cont, y_train_cont, X_test_cont, y_test_cont, "Custom Gaussian NB")
    benchmark_model(MultinomialNaiveBayesFromScratch(), X_train, y_train, X_test, y_test, "Custom Multinomial NB")

    benchmark_model(BernoulliNB(), X_train, y_train, X_test, y_test, "Sklearn BernoulliNB")
    benchmark_model(GaussianNB(), X_train_cont, y_train_cont, X_test_cont, y_test_cont, "Sklearn GaussianNB")
    benchmark_model(MultinomialNB(), X_train, y_train, X_test, y_test, "Sklearn MultinomialNB")
