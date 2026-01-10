import pandas as pd
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB

# Load dataset from CSV file
def load_dataset(csv_path):
    """Load dataset from CSV file using pandas"""
    df = pd.read_csv(csv_path)
    return df

# Example: Loading the breast cancer dataset
dataset_path = Path(__file__).parent / "breast cancer" / "data.csv"
if dataset_path.exists():
    data = load_dataset(str(dataset_path))
    print(f"Dataset loaded successfully!")
    
    # Prepare data for training
    # Remove unnecessary columns
    data_clean = data.drop(['id', 'Unnamed: 32'], axis=1)
    
    # Separate features and target
    X = data_clean.drop('diagnosis', axis=1).values
    y = data_clean['diagnosis'].values
    
    # Encode target variable (M=1, B=0)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Normalize features to be suitable for Multinomial NB (convert to counts)
    # Multinomial NB expects non-negative integers, so we'll scale the data
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_normalized = ((X - X_min) / (X_max - X_min + 1e-10) * 100).astype(int)
    
    print(f"Features shape: {X_normalized.shape}")
    print(f"Target shape: {y_encoded.shape}")
    print(f"Classes: {le.classes_} -> {np.unique(y_encoded)}")
    print(f"Class distribution:\n{pd.Series(y_encoded).value_counts().sort_index()}")
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\nTrain set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
else:
    print(f"Dataset not found at {dataset_path}")
    exit(1)

class MultinomialNaiveBayesFromScratch:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing parameter
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.n_features = X.shape[1]
        
        self.class_priors = np.zeros(self.n_classes, dtype=np.float64)
        self.feature_counts = np.zeros((self.n_classes, self.n_features), dtype=np.float64)
        self.class_totals = np.zeros(self.n_classes, dtype=np.float64)
        
        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.class_priors[i] = X_c.shape[0] / X.shape[0]
            
            # Sum of all feature counts in this class
            self.feature_counts[i, :] = np.sum(X_c, axis=0)
            self.class_totals[i] = np.sum(self.feature_counts[i, :])
    
    def predict(self, X):
        predictions = [self._predict_sample(x) for x in X]
        return np.array(predictions)
    
    def _predict_sample(self, x):
        log_posteriors = []
        
        for i, c in enumerate(self.classes):
            log_prior = np.log(self.class_priors[i])
            log_likelihood = 0.0
            
            for j in range(self.n_features):
                # Calculate log probability with Laplace smoothing
                count = self.feature_counts[i, j]
                total = self.class_totals[i]
                prob = (count + self.alpha) / (total + self.n_features * self.alpha)
                
                # Multiply by feature value (assuming integer counts)
                if x[j] > 0:
                    log_likelihood += x[j] * np.log(prob)
            
            log_posteriors.append(log_prior + log_likelihood)
        
        return self.classes[np.argmax(log_posteriors)]

# Training the models
print(f"\n" + "="*80)
print("MODEL TRAINING")
print("="*80)

# Train custom Multinomial Naive Bayes
print("\n1. Training Custom Multinomial Naive Bayes...")
start_time = time.time()
mnb_custom = MultinomialNaiveBayesFromScratch(alpha=1.0)
mnb_custom.fit(X_train, y_train)
custom_train_time = time.time() - start_time
print(f"   Training time: {custom_train_time:.4f} seconds")

# Make predictions with custom model
y_pred_custom = mnb_custom.predict(X_test)

# Train sklearn Multinomial Naive Bayes
print("\n2. Training sklearn Multinomial Naive Bayes...")
start_time = time.time()
mnb_sklearn = MultinomialNB(alpha=1.0)
mnb_sklearn.fit(X_train, y_train)
sklearn_train_time = time.time() - start_time
print(f"   Training time: {sklearn_train_time:.4f} seconds")

# Make predictions with sklearn model
y_pred_sklearn = mnb_sklearn.predict(X_test)

# Evaluation
print(f"\n" + "="*80)
print("MODEL EVALUATION RESULTS")
print("="*80)

# Custom Model Metrics
print("\n--- CUSTOM MULTINOMIAL NAIVE BAYES ---")
custom_accuracy = accuracy_score(y_test, y_pred_custom)
custom_precision = precision_score(y_test, y_pred_custom)
custom_recall = recall_score(y_test, y_pred_custom)
custom_f1 = f1_score(y_test, y_pred_custom)

print(f"Accuracy:  {custom_accuracy:.4f} ({custom_accuracy*100:.2f}%)")
print(f"Precision: {custom_precision:.4f}")
print(f"Recall:    {custom_recall:.4f}")
print(f"F1-Score:  {custom_f1:.4f}")

# sklearn Model Metrics
print("\n--- SKLEARN MULTINOMIAL NAIVE BAYES ---")
sklearn_accuracy = accuracy_score(y_test, y_pred_sklearn)
sklearn_precision = precision_score(y_test, y_pred_sklearn)
sklearn_recall = recall_score(y_test, y_pred_sklearn)
sklearn_f1 = f1_score(y_test, y_pred_sklearn)

print(f"Accuracy:  {sklearn_accuracy:.4f} ({sklearn_accuracy*100:.2f}%)")
print(f"Precision: {sklearn_precision:.4f}")
print(f"Recall:    {sklearn_recall:.4f}")
print(f"F1-Score:  {sklearn_f1:.4f}")

# Comparison
print(f"\n--- COMPARISON ---")
print(f"Custom Model Accuracy:  {custom_accuracy*100:.2f}%")
print(f"sklearn Model Accuracy: {sklearn_accuracy*100:.2f}%")
print(f"Difference: {abs(custom_accuracy - sklearn_accuracy)*100:.2f}%")

# Confusion Matrices
print(f"\n--- CONFUSION MATRICES ---")
cm_custom = confusion_matrix(y_test, y_pred_custom)
cm_sklearn = confusion_matrix(y_test, y_pred_sklearn)

print(f"\nCustom Model Confusion Matrix:")
print(f"             Predicted B  Predicted M")
print(f"Actual B     {cm_custom[0,0]:>4}          {cm_custom[0,1]:>4}")
print(f"Actual M     {cm_custom[1,0]:>4}          {cm_custom[1,1]:>4}")

print(f"\nsklearn Model Confusion Matrix:")
print(f"             Predicted B  Predicted M")
print(f"Actual B     {cm_sklearn[0,0]:>4}          {cm_sklearn[0,1]:>4}")
print(f"Actual M     {cm_sklearn[1,0]:>4}          {cm_sklearn[1,1]:>4}")

# Classification Report
print(f"\n--- CUSTOM MODEL CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred_custom, target_names=le.classes_))

print(f"--- SKLEARN MODEL CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred_sklearn, target_names=le.classes_))

print(f"\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
        