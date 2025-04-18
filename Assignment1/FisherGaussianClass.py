import numpy as np


class FisherGaussianLDAClassifier:
    def __init__(self):
        self.coefs_ = None  # coefficient matrix
        self.intercepts_ = None  # intercepts
        self.classes_ = None  # class labels
        self.Sigma_inv = None  # inverse of covariant matrix
        self.mean_vecs = None  # mean vectors of the classes

    def train(self, X_train, y_train):
        # get classes
        self.classes_ = np.unique(y_train)
        n_classes = len(self.classes_)
        n_samples, n_features = X_train.shape

        # compute for each class
        self.mean_vecs = [np.mean(X_train[y_train == cls], axis=0) for cls in self.classes_]
        class_counts = [np.sum(y_train == cls) for cls in self.classes_]

        # Sw
        Sw = np.zeros((n_features, n_features))
        for cls, mean_vec in zip(self.classes_, self.mean_vecs):
            class_samples = X_train[y_train == cls]
            Sw += (class_samples - mean_vec).T @ (class_samples - mean_vec)
        Sw += 1e-6 * np.eye(n_features)  # regularization

        # covariant matrix and inverse
        Sigma = Sw / (n_samples - n_classes)
        self.Sigma_inv = np.linalg.inv(Sigma)

        # prior probabilities
        priors = np.array(class_counts) / n_samples

        # coefficient matrix (gaussian distribution assumption)
        self.coefs_ = [self.Sigma_inv @ mean_vector for mean_vector in self.mean_vecs]
        self.intercepts_ = [
            -0.5 * mean_vector @ self.Sigma_inv @ mean_vector + np.log(prior)
            for mean_vector, prior in zip(self.mean_vecs, priors)
        ]

        return self

    def predict(self, X):
        if self.coefs_ is None:
            raise ValueError("Please fit the model first.")
        coef_matrix = np.array(self.coefs_)  # (n_classes, n_features)
        intercepts = np.array(self.intercepts_)  # (n_classes,)
        scores = X @ coef_matrix.T + intercepts  # (n_samples, n_classes)
        return self.classes_[np.argmax(scores, axis=1)]

    def compute_accuracy(self, X_test, y_test):
        prediction = self.predict(X_test)
        if y_test is not None:
            mask = np.array(prediction == y_test)
            acc = mask.astype(float).mean()
            return acc, mask, prediction
        else:
            return None, None, prediction

