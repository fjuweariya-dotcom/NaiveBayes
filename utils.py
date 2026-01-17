"""
Utility functions for Naive Bayes classification tasks
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, Binarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


def load_dataset(csv_path):
    """
    Load dataset from CSV file using pandas.
    
    Parameters:
    -----------
    csv_path : str or Path
        Path to the CSV file
        
    Returns:
    --------
    pandas.DataFrame
        Loaded dataset
    """
    df = pd.read_csv(csv_path)
    return df


def clean_breast_cancer_data(df, drop_columns=None):
    """
    Clean breast cancer dataset by removing unnecessary columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw dataframe
    drop_columns : list, optional
        Columns to drop. Defaults to ['id', 'Unnamed: 32']
        
    Returns:
    --------
    pandas.DataFrame
        Cleaned dataframe
    """
    if drop_columns is None:
        drop_columns = ['id', 'Unnamed: 32']
    
    return df.drop(drop_columns, axis=1, errors='ignore')


def prepare_features_and_target(df, target_column='diagnosis'):
    """
    Separate features and target variable from dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    target_column : str
        Name of target column
        
    Returns:
    --------
    tuple
        (features array, target array)
    """
    X = df.drop(target_column, axis=1).values
    y = df[target_column].values
    return X, y


def encode_target(y, label_encoder=None):
    """
    Encode target variable using LabelEncoder.
    
    Parameters:
    -----------
    y : array-like
        Target variable
    label_encoder : sklearn.preprocessing.LabelEncoder, optional
        Existing encoder. If None, creates new one.
        
    Returns:
    --------
    tuple
        (encoded target, encoder)
    """
    if label_encoder is None:
        label_encoder = LabelEncoder()
    
    y_encoded = label_encoder.fit_transform(y)
    return y_encoded, label_encoder


def normalize_features(X, method='minmax', scale_factor=100):
    """
    Normalize features using min-max scaling.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    method : str
        Normalization method ('minmax' or 'standard')
    scale_factor : float
        Factor to scale normalized values (for integer conversion)
        
    Returns:
    --------
    numpy.ndarray
        Normalized features
    """
    if method == 'minmax':
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        X_normalized = (X - X_min) / (X_max - X_min + 1e-10)
        return (X_normalized * scale_factor).astype(int)
    elif method == 'standard':
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        return (X - X_mean) / (X_std + 1e-10)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def binarize_features(X_train, X_test, threshold_type='mean'):
    """
    Binarize features for Bernoulli Naive Bayes.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training features
    X_test : numpy.ndarray
        Testing features
    threshold_type : str
        Type of threshold ('mean' or custom array)
        
    Returns:
    --------
    tuple
        (binarized X_train, binarized X_test)
    """
    if threshold_type == 'mean':
        thresholds = X_train.mean(axis=0)
    else:
        thresholds = threshold_type
    
    X_train_binarized = np.zeros_like(X_train, dtype=int)
    X_test_binarized = np.zeros_like(X_test, dtype=int)
    
    for i in range(X_train.shape[1]):
        binarizer = Binarizer(threshold=thresholds[i])
        X_train_binarized[:, i] = binarizer.transform(
            X_train[:, i].reshape(-1, 1)).flatten()
        X_test_binarized[:, i] = binarizer.transform(
            X_test[:, i].reshape(-1, 1)).flatten()
    
    return X_train_binarized, X_test_binarized


def split_data(X, y, test_size=0.2, random_state=42, stratify=True):
    """
    Split data into training and testing sets.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Features
    y : numpy.ndarray
        Target
    test_size : float
        Proportion of test set
    random_state : int
        Random seed
    stratify : bool
        Whether to stratify split by target
        
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    stratify_arg = y if stratify else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify_arg)


def calculate_metrics(y_true, y_pred, average='weighted'):
    """
    Calculate classification metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    average : str
        Averaging method for multi-class metrics
        
    Returns:
    --------
    dict
        Dictionary containing accuracy, precision, recall, f1-score
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
    }


def print_metrics(metrics, y_true, y_pred):
    """
    Print classification metrics and confusion matrix.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of metrics from calculate_metrics()
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    """
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_true, y_pred)}")
    print(f"\nClassification Report:\n{classification_report(y_true, y_pred)}")


def find_breast_cancer_dataset(base_path=None):
    """
    Find breast cancer dataset in common locations.
    
    Parameters:
    -----------
    base_path : str or Path, optional
        Base path to search. Defaults to current file's parent directory.
        
    Returns:
    --------
    Path or None
        Path to data.csv if found, None otherwise
    """
    if base_path is None:
        base_path = Path(__file__).parent
    else:
        base_path = Path(base_path)
    
    common_paths = [
        base_path / "breast cancer" / "data.csv",
        base_path / "Naive bayes project" / "breast cancer" / "data.csv",
    ]
    
    for path in common_paths:
        if path.exists():
            return path
    
    return None


class HashTable:
    """
    Simple hash table implementation for storing key-value pairs.
    Used in from-scratch Naive Bayes implementations.
    """
    def __init__(self, size=100):
        self.size = size
        self.table = [[] for _ in range(self.size)]
    
    def _hash(self, key):
        """Compute hash value for key"""
        return hash(key) % self.size
    
    def insert(self, key, value):
        """Insert key-value pair into hash table"""
        index = self._hash(key)
        for i, (k, v) in enumerate(self.table[index]):
            if k == key:
                self.table[index][i] = (key, value)
                return
        self.table[index].append((key, value))
    
    def search(self, key):
        """Search for value by key"""
        index = self._hash(key)
        for k, v in self.table[index]:
            if k == key:
                return v
        return None
    
    def delete(self, key):
        """Delete key-value pair from hash table"""
        index = self._hash(key)
        self.table[index] = [(k, v) for k, v in self.table[index] if k != key]
