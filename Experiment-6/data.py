# import pandas as pd
# import numpy as np
# from ucimlrepo import fetch_ucirepo

# def load_and_prepare_dataset(dataset_id=53):
#     """
#     Loads a dataset from the UCI repository, preprocesses it, and returns the
#     features and target labels as NumPy arrays, along with their names.
#     """
#     dataset = fetch_ucirepo(id=dataset_id)

#     X_df = dataset.data.features
#     y_df = dataset.data.targets
    
#     # --- THIS IS THE CORRECTED SECTION FOR THE ERROR ---
#     # Safely get the feature names.
#     feature_names = dataset.metadata.get('feature_names')
#     if feature_names is None:
#         # Fallback: If metadata is empty, get names from the DataFrame columns
#         feature_names = X_df.columns.tolist()
    
#     # Safely get the target name.
#     target_names_list = dataset.metadata.get('target_names')
#     if target_names_list:
#         target_name = target_names_list[0]
#     else:
#         target_name = y_df.columns[0]
#     # --- END OF CORRECTION ---

#     # --- Conditional Preprocessing based on dataset ---
#     if dataset_id == 53: # Iris dataset
#         # FIX FOR WARNING: Use .loc for safer assignment
#         y_df.loc[:, target_name] = y_df[target_name].str.replace('Iris-', '')
    
#     X_numpy = X_df.values
#     y_numpy = y_df.values.ravel()

#     print(f"✅ Loaded '{dataset.metadata.get('name')}' dataset.")
#     return X_numpy, y_numpy, feature_names, target_name



# File: data.py
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo

def load_and_prepare_dataset(dataset_id=53):
    """
    Loads a dataset from the UCI repository, preprocesses it, 
    and returns NumPy arrays for features and labels.

    Args:
        dataset_id (int): UCI dataset ID (53 for Iris, 109 for Wine).

    Returns:
        tuple: (X, y, feature_names, target_name)
    """
    dataset = fetch_ucirepo(id=dataset_id)
    X_df = dataset.data.features
    y_df = dataset.data.targets

    # Feature names
    feature_names = dataset.metadata.get('feature_names')
    if feature_names is None:
        feature_names = X_df.columns.tolist()

    # Target name
    target_names_list = dataset.metadata.get('target_names')
    target_name = target_names_list[0] if target_names_list else y_df.columns[0]

    # --- Dataset-specific preprocessing ---
    if dataset_id == 53:
        # For Iris dataset — remove the "Iris-" prefix
        y_df.loc[:, target_name] = y_df[target_name].str.replace('Iris-', '')
    elif dataset_id == 109:
        # For Wine dataset — ensure numeric target (if not already)
        y_df[target_name] = y_df[target_name].astype(str)

    # Convert to NumPy arrays
    X_numpy = X_df.values
    y_numpy = y_df.values.ravel()

    print(f"✅ Loaded '{dataset.metadata.get('name')}' dataset.")
    return X_numpy, y_numpy, feature_names, target_name
