class MultinomialNaiveBayesFromScratch:
    def _init_(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing parameter

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.n_features = X.shape[1]

        self.class_priors = np.zeros(self.n_classes, dtype=np.float64)
        self.feature_counts = np.zeros(
            (self.n_classes, self.n_features), dtype=np.float64)
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
                prob = (count + self.alpha) / \
                    (total + self.n_features * self.alpha)

                # Multiply by feature value (assuming integer counts)
                if x[j] > 0:
                    log_likelihood += x[j] * np.log(prob)

            log_posteriors.append(log_prior + log_likelihood)

        return self.classes[np.argmax(log_posteriors)]
