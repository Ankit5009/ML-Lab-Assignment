# # File: knn_classifier.py
# # This cell contains the implementation of the KNN model.

# import numpy as np
# from collections import Counter

# def euclidean_distance(x1, x2):
#     """
#     Calculates the scalar Euclidean distance between two data points.
    
#     Formula: sqrt(sum((x1 - x2)^2))

#     Args:
#         x1 (np.ndarray): The first data point.
#         x2 (np.ndarray): The second data point.

#     Returns:
#         float: The Euclidean distance between the two points.
#     """
#     return np.sqrt(np.sum((x1 - x2)**2))


# class KNNClassifier:
#     """
#     K-Nearest Neighbors classifier implemented from scratch.
#     """
#     def __init__(self, k=3):
#         """
#         Constructor method for the classifier.
        
#         Args:
#             k (int): The number of neighbors to use for classification.
#         """
#         self.k = k

#     def fit(self, X_train, y_train):
#         """
#         Stores the training data. For KNN, this is the entire "training" process.

#         Args:
#             X_train (np.ndarray): The training feature data.
#             y_train (np.ndarray): The training labels.
#         """
#         self.X_train = X_train
#         self.y_train = y_train

#     def predict(self, X_test):
#         """
#         Generates predictions for a set of test data.

#         Args:
#             X_test (np.ndarray): The feature data to predict on.

#         Returns:
#             np.ndarray: An array containing the predicted labels for each sample in X_test.
#         """
#         # Iterate through each sample in X_test and get its prediction
#         predictions = [self._predict(x) for x in X_test]
#         return np.array(predictions)

#     def _predict(self, x):
#         """
#         Private helper method to predict the class for a single data point.

#         Args:
#             x (np.ndarray): A single sample from the test set.

#         Returns:
#             The predicted label for the sample.
#         """
#         # a. Calculate the distance from x to every point in self.X_train
#         distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        
#         # b. Identify the indices of the k training samples with the smallest distances
#         # np.argsort returns the indices that would sort the array
#         k_nearest_indices = np.argsort(distances)[:self.k]
        
#         # c. Retrieve the labels corresponding to these k indices
#         k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]
        
#         # d. Determine the most frequently occurring label (majority vote)
#         # Counter(...).most_common(1) returns a list like [('label', count)]
#         most_common = Counter(k_nearest_labels).most_common(1)
        
#         return most_common[0][0]


# File: knn_classifier.py
import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    """Compute Euclidean distance between two vectors."""
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNNClassifier:
    """Custom implementation of K-Nearest Neighbors classifier."""

    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        # Step 1: Compute all distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Step 2: Find indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Step 3: Collect labels of those neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Step 4: Return the majority vote
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
