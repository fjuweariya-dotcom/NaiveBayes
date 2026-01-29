class GaussianNaiveBayesFromScratch:
    def __init__(self):
        # Initialize HashTable for means and variances
        self._mean_table = HashTable()
        self._variance_table = HashTable()

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # Calculate prior for each class
        self._priors = np.zeros(n_classes)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            n_c = X_c.shape[0]
            self._priors[idx] = n_c / n_samples

            for j in range(n_features):
                # Calculate mean and variance for each feature in each class
                mean_val = np.mean(X_c[:, j])
                # Add a small epsilon to variance to prevent division by zero
                variance_val = np.var(X_c[:, j]) + 1e-9  # Add epsilon

                # Store mean and variance in hash tables
                self._mean_table.insert((idx, j), mean_val)
                self._variance_table.insert((idx, j), variance_val)

    def predict(self, X):
        return [self._predict(x) for x in X]

    def _predict(self, x):
        posteriors = []
        n_features = x.shape[0]

        # Calculate posterior probability for each class
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            # Calculate likelihood using Gaussian PDF: N(x; mu, sigma^2)
            class_conditional = np.sum(self._gaussian_pdf(idx, x, n_features))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]

    def _gaussian_pdf(self, class_idx, x, n_features):
        log_likelihoods = []
        for j in range(n_features):
            mean = self._mean_table.search((class_idx, j))
            variance = self._variance_table.search((class_idx, j))

            if mean is None or variance is None:
                # Handle case where key might not be found (shouldn't happen with correct logic)
                # or if variance is 0 (after epsilon addition, it should not be)
                raise ValueError(
                    f"Mean or Variance not found for class_idx {class_idx}, feature_idx {j}")

            numerator = np.exp(- (x[j] - mean)**2 / (2 * variance))
            denominator = np.sqrt(2 * np.pi * variance)

            # Prevent log(0) if numerator becomes extremely small after calculation
            likelihood_val = numerator / denominator
            if likelihood_val == 0:  # Extremely small number, treat as very low log-likelihood
                # A very small number instead of 0
                log_likelihoods.append(np.log(1e-300))
            else:
                log_likelihoods.append(np.log(likelihood_val))

        # Return log likelihood to avoid underflow with very small numbers
        return np.array(log_likelihoods)
