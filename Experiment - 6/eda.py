# # File: eda.py
# # This cell contains a more general script for exploratory data analysis.

# import matplotlib.pyplot as plt
# import numpy as np
# import itertools

# def visualize_feature_pairs(X, y, feature_names, target_name):
#     """
#     Generates scatter plots for every unique pair of features in a dataset.

#     Args:
#         X (np.ndarray): The feature data.
#         y (np.ndarray): The target labels.
#         feature_names (list): The names of the features for axis labels.
#         target_name (str): The name of the target variable for the legend.
#     """
#     unique_classes = np.unique(y)
#     num_features = X.shape[1]
    
#     # Generate all unique pairs of feature indices
#     feature_pairs = list(itertools.combinations(range(num_features), 2))
    
#     # Calculate grid size for subplots (e.g., 4 features -> 6 plots -> 2x3 grid)
#     n_rows = int(np.ceil(len(feature_pairs) / 3))
#     n_cols = 3
    
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
#     axes = axes.flatten()

#     for i, (feat1_idx, feat2_idx) in enumerate(feature_pairs):
#         ax = axes[i]
#         for cls in unique_classes:
#             ax.scatter(X[y == cls, feat1_idx], X[y == cls, feat2_idx], label=cls)
        
#         ax.set_xlabel(feature_names[feat1_idx])
#         ax.set_ylabel(feature_names[feat2_idx])
#         ax.set_title(f'{feature_names[feat1_idx]} vs. {feature_names[feat2_idx]}')
#         ax.grid(True)

#     # Hide any unused subplots
#     for j in range(i + 1, len(axes)):
#         axes[j].set_visible(False)
        
#     handles, labels = axes[0].get_legend_handles_labels()
#     fig.legend(handles, labels, loc='upper right', title=target_name)
#     plt.tight_layout(rect=[0, 0, 0.9, 1])
#     plt.show()

# File: eda.py
import matplotlib.pyplot as plt
import numpy as np
import itertools

def visualize_feature_pairs(X, y, feature_names, target_name):
    """
    Generates scatter plots for all feature pairs in the dataset.

    Args:
        X (np.ndarray): Feature data.
        y (np.ndarray): Labels.
        feature_names (list): List of feature names.
        target_name (str): Label name for the legend.
    """
    unique_classes = np.unique(y)
    num_features = X.shape[1]
    feature_pairs = list(itertools.combinations(range(num_features), 2))

    n_rows = int(np.ceil(len(feature_pairs) / 3))
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes = axes.flatten()

    for i, (f1, f2) in enumerate(feature_pairs):
        ax = axes[i]
        for cls in unique_classes:
            ax.scatter(
                X[y == cls, f1], X[y == cls, f2],
                label=cls, alpha=0.7
            )
        ax.set_xlabel(feature_names[f1])
        ax.set_ylabel(feature_names[f2])
        ax.set_title(f'{feature_names[f1]} vs. {feature_names[f2]}')
        ax.grid(True)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', title=target_name)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()
