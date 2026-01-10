# Naive Bayes Classification Project

This repository contains implementations of various Naive Bayes classifiers from scratch and comparisons with scikit-learn implementations.

## 📋 Project Overview

Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem with the assumption of feature independence. This project implements and evaluates three variants:

- **Bernoulli Naive Bayes** - For binary feature problems
- **Gaussian Naive Bayes** - For continuous feature problems
- **Multinomial Naive Bayes** - For count-based feature problems

## 📁 Project Structure

```
NaiveBayes/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── Multinomial                        # Multinomial NB implementation
├── Gaussian.py                        # Gaussian NB implementation
├── Gaussian.ipynb                     # Jupyter notebook for Gaussian NB
├── BernoulliNB.ipynb                  # Jupyter notebook for Bernoulli NB
├── BernoulliGausianNB pd.ipy          # Interactive Python script
├── breast cancer/
│   └── data.csv                       # Dataset: Breast Cancer Wisconsin
└── Naive bayes project/               # Additional implementations
    ├── BernoulliGausianNB pd.ipy
    ├── BernoulliNB.ipynb
    ├── Gaussian.py
    └── breast cancer/
        └── data.csv
```

## 🗂️ Dataset

**Breast Cancer Wisconsin Dataset**
- **Total Samples:** 569
- **Features:** 30 cancer-related measurements
- **Classes:** 2 (Benign, Malignant)
- **Training Set:** 455 samples (80%)
- **Test Set:** 114 samples (20%)

### Features Include:
- Radius, texture, perimeter, area measurements
- Smoothness, compactness, concavity measurements
- Symmetry and fractal dimension measurements
- Mean, standard error, and worst case statistics

## 📦 Installation

### Prerequisites
- Python 3.12+
- Virtual environment (optional but recommended)

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/fjuweariya-dotcom/NaiveBayes.git
cd NaiveBayes
```

2. **Create a virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## 📚 Dependencies

```
pandas          # Data manipulation and analysis
numpy           # Numerical computing
scikit-learn    # Machine learning algorithms
```

## 🚀 Usage

### Running the Multinomial Naive Bayes Model

```bash
python Multinomial
```

### Running Jupyter Notebooks

```bash
jupyter notebook Gaussian.ipynb
jupyter notebook BernoulliNB.ipynb
```

### Running Python Scripts

```bash
python Gaussian.py
```

## 📊 Model Results

### Multinomial Naive Bayes Performance

| Metric | Custom Model | sklearn Model | Result |
|--------|------------|--------------|--------|
| **Accuracy** | 81.58% | 81.58% | ✅ Match |
| **Precision** | 0.7333 | 0.7333 | ✅ Match |
| **Recall** | 0.7857 | 0.7857 | ✅ Match |
| **F1-Score** | 0.7586 | 0.7586 | ✅ Match |

### Confusion Matrix (Multinomial)
```
             Predicted B  Predicted M
Actual B       60            12
Actual M        9            33
```

### Classification Report
```
              precision    recall  f1-score   support

       Benign       0.87      0.83      0.85        72
   Malignant       0.73      0.79      0.76        42

    accuracy                           0.82       114
   macro avg       0.80      0.81      0.80       114
weighted avg       0.82      0.82      0.82       114
```

### Performance Metrics
- **Training Time (Custom):** 0.0002 seconds
- **Training Time (sklearn):** 0.0021 seconds
- **Model Consistency:** Custom implementation produces identical results to scikit-learn

## 🔍 Implementation Details

### Multinomial Naive Bayes
The Multinomial Naive Bayes classifier is particularly useful for:
- Text classification
- Count-based data
- Document classification

**Key Parameters:**
- `alpha` (Laplace smoothing): Default = 1.0

**Algorithm:**
1. Calculate class priors: P(Class)
2. Calculate feature probabilities with Laplace smoothing
3. Apply log probabilities to prevent numerical underflow
4. Select class with maximum posterior probability

### Feature Preprocessing
- Features normalized from continuous values to count values (0-100 scale)
- Label encoding for target variable (Benign=0, Malignant=1)
- Data split: 80% training, 20% testing with stratification

## 📈 Key Findings

1. **Model Accuracy:** Custom Multinomial NB achieves 81.58% accuracy
2. **Implementation Correctness:** Custom implementation matches scikit-learn output exactly
3. **Performance:** Both models are fast (< 3ms training time)
4. **Class Balance:** Better performance on Benign class (87% precision) vs Malignant (73% precision)

## 🎯 Model Interpretation

**Strengths:**
- Fast training and prediction
- Good overall accuracy (81.58%)
- Excellent precision on Benign classification (87%)
- Interpretable probability-based decisions

**Areas for Improvement:**
- 12 false positives (Benign classified as Malignant)
- 9 false negatives (Malignant classified as Benign)
- Consider ensemble methods or feature engineering for higher recall on Malignant class

## 🧪 Experiments

The repository includes experiments comparing:
- Custom implementations vs scikit-learn
- Different naive Bayes variants (Bernoulli, Gaussian, Multinomial)
- Feature preprocessing techniques
- Model performance metrics

## 📖 Learning Resources

### Naive Bayes Concepts
- Bayes' Theorem: P(A|B) = P(B|A)P(A) / P(B)
- Conditional Independence Assumption
- Laplace Smoothing for handling zero probabilities
- Log-probability to prevent numerical underflow

### Notebooks
All notebooks include detailed explanations and visualizations of:
- Data exploration
- Model training
- Performance evaluation
- Result interpretation

## 🤝 Contributing

Feel free to contribute by:
1. Improving model implementations
2. Adding new classifier variants
3. Enhancing documentation
4. Optimizing performance
5. Adding new datasets

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

**fjuweariya-dotcom**

Created: January 2026

## 📞 Support

For questions or issues, please open an issue on GitHub.

## 🔗 References

- [Scikit-learn Naive Bayes Documentation](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [Naive Bayes Theorem](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
- [Breast Cancer Wisconsin Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))

---

**Last Updated:** January 10, 2026
