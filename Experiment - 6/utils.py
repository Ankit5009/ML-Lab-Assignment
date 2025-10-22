# # File: utils.py
# # This cell is for general utility functions.

# import numpy as np

# def train_test_split(X, y, test_size=0.2, random_state=None):
#     """
#     Splits the data into training and testing sets from scratch.

#     Args:
#         X (np.ndarray): The feature matrix (samples, features).
#         y (np.ndarray): The target vector (samples,).
#         test_size (float): The proportion of the dataset to allocate to the test set.
#         random_state (int, optional): Seed for the random number generator for reproducibility.

#     Returns:
#         tuple: A tuple containing the split arrays in the order:
#                (X_train, X_test, y_train, y_test).
#     """
#     # 1. Set the random seed for reproducibility if provided
#     if random_state is not None:
#         np.random.seed(random_state)

#     # 2. Get the total number of samples in the dataset
#     num_samples = X.shape[0]

#     # 3. Generate a shuffled sequence of indices
#     indices = np.arange(num_samples)
#     np.random.shuffle(indices)

#     # 4. Calculate the split point
#     # This is the index where the training data ends and test data begins
#     split_point = int(num_samples * (1 - test_size))

#     # 5. Partition the shuffled indices into training and testing sets
#     train_indices = indices[:split_point]
#     test_indices = indices[split_point:]

#     # 6. Use the partitioned indices to slice the X and y arrays
#     X_train = X[train_indices]
#     X_test = X[test_indices]
#     y_train = y[train_indices]
#     y_test = y[test_indices]
    
#     # 7. Return the four resulting NumPy arrays
#     return X_train, X_test, y_train, y_test


# File: utils.py
import numpy as np

def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Splits arrays into random train and test subsets.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels array.
        test_size (float): Fraction of data for testing.
        random_state (int, optional): Seed for reproducibility.

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    if random_state is not None:
        np.random.seed(random_state)

    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    split_point = int(num_samples * (1 - test_size))
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
