class BernoulliNaiveBayesFromScratch:
    def __init__(self):
        # Initialize HashTable for likelihoods
        self._likelihood_table = HashTable()

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # Calculate P(class) - prior probabilities
        self._priors = np.zeros(n_classes)
        for idx, c in enumerate(self._classes):
            self._priors[idx] = np.sum(y == c) / n_samples

        # Calculate P(feature|class) - likelihoods using HashTable
        # Add Laplace smoothing to prevent zero probabilities
        alpha = 1  # Laplace smoothing parameter

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            n_c = X_c.shape[0]

            # Calculate P(feature_j=1 | class_c) for all features
            p_feature_given_class_1 = (
                np.sum(X_c, axis=0) + alpha) / (n_c + alpha * 2)

            for j in range(n_features):
                # Store P(feature_j=1 | class_c) in hash table
                self._likelihood_table.insert(
                    (idx, j, 1), p_feature_given_class_1[j])
                # Store P(feature_j=0 | class_c) in hash table
                self._likelihood_table.insert(
                    (idx, j, 0), 1 - p_feature_given_class_1[j])

    def predict(self, X):
        return [self._predict(x) for x in X]

    def _predict(self, x):
        posteriors = []
        n_features = x.shape[0]

        # Calculate posterior probability for each class
        for idx, c_label in enumerate(self._classes):
            prior = np.log(self._priors[idx])

            class_conditional_log_likelihood = 0.0
            for j in range(n_features):
                feature_value = x[j]

                # Retrieve likelihood from HashTable
                if feature_value == 1:
                    likelihood = self._likelihood_table.search((idx, j, 1))
                else:  # feature_value == 0
                    likelihood = self._likelihood_table.search((idx, j, 0))

                # Add log likelihood to the sum
                class_conditional_log_likelihood += np.log(likelihood)

            posterior = prior + class_conditional_log_likelihood
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]
