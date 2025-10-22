# # File: main.py
# import numpy as np
# import matplotlib.pyplot as plt

# # Import your custom functions and classes from the other files
# from data import load_and_prepare_dataset
# from utils import train_test_split
# from knn_classifier import KNNClassifier
# from eda import visualize_feature_pairs

# print("--- [START] Machine Learning Pipeline for Iris Dataset ---")

# # --- Step 1: Assemble the Pipeline ---
# X, y, feature_names, target_name = load_and_prepare_dataset(dataset_id=53)
# visualize_feature_pairs(X, y, feature_names, target_name)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # --- Step 2 & 3: Hyperparameter Tuning and Analysis ---
# print("\n[INFO] Performing Hyperparameter Tuning for k...")
# k_values = [1, 3, 5, 7, 9, 11, 15]
# accuracies = {}

# for k in k_values:
#     knn = KNNClassifier(k=k)
#     knn.fit(X_train, y_train)
#     predictions = knn.predict(X_test)
#     accuracy = np.sum(predictions == y_test) / len(y_test)
#     accuracies[k] = accuracy
#     print(f"  - Accuracy for k = {k}: {accuracy:.4f}")

# plt.figure(figsize=(10, 6))
# plt.plot(list(accuracies.keys()), list(accuracies.values()), marker='o', linestyle='--')
# plt.title('Accuracy vs. k-value for Iris Dataset')
# plt.xlabel('k (Number of Neighbors)')
# plt.ylabel('Model Accuracy')
# plt.xticks(k_values)
# plt.grid(True)
# plt.show()

# best_k = max(accuracies, key=accuracies.get)
# print(f"\n[INFO] Best k-value identified: {best_k} with an accuracy of {accuracies[best_k]:.4f}")
# print("--- [END] Iris Dataset Pipeline Finished ---\n")


# # --- Step 4: Generalization to a New Dataset ---
# print("="*60)
# print("--- [START] Generalizing to Wine Dataset ---")

# X_wine, y_wine, _, _ = load_and_prepare_dataset(dataset_id=109)
# X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(
#     X_wine, y_wine, test_size=0.2, random_state=42
# )
# print(f"\n[INFO] Evaluating model on Wine dataset using the best k ({best_k})...")
# knn_wine = KNNClassifier(k=best_k)
# knn_wine.fit(X_train_wine, y_train_wine)
# predictions_wine = knn_wine.predict(X_test_wine)
# accuracy_wine = np.sum(predictions_wine == y_test_wine) / len(y_test_wine)

# print(f"\n✅ Final Accuracy on the Wine Dataset (with k={best_k}): {accuracy_wine:.4f} ({accuracy_wine * 100:.2f}%)")
# print("--- [END] Generalization Finished ---")

# File: main.py
import numpy as np
import matplotlib.pyplot as plt

from data import load_and_prepare_dataset
from utils import train_test_split
from knn_classifier import KNNClassifier
from eda import visualize_feature_pairs

print("--- [START] Machine Learning Pipeline for Iris Dataset ---")

# --- Step 1: Load and visualize data ---
X, y, feature_names, target_name = load_and_prepare_dataset(dataset_id=53)
visualize_feature_pairs(X, y, feature_names, target_name)

# --- Step 2: Split the data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 3: Hyperparameter tuning ---
print("\n[INFO] Performing Hyperparameter Tuning for k...")
k_values = [1, 3, 5, 7, 9, 11, 15]
accuracies = {}

for k in k_values:
    knn = KNNClassifier(k=k)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    accuracy = np.sum(preds == y_test) / len(y_test)
    accuracies[k] = accuracy
    print(f" - Accuracy for k = {k}: {accuracy:.4f}")

# --- Step 4: Plot accuracy vs k ---
plt.figure(figsize=(8, 5))
plt.plot(list(accuracies.keys()), list(accuracies.values()), marker='o', linestyle='--')
plt.title('Accuracy vs. k-value (Iris Dataset)')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

best_k = max(accuracies, key=accuracies.get)
print(f"\n[INFO] Best k-value identified: {best_k} with accuracy = {accuracies[best_k]:.4f}")
print("--- [END] Iris Dataset Pipeline Finished ---")

# --- Step 5: Generalization to Wine dataset ---
print("=" * 60)
print("--- [START] Generalizing to Wine Dataset ---")

X_wine, y_wine, _, _ = load_and_prepare_dataset(dataset_id=109)
X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(X_wine, y_wine, test_size=0.2, random_state=42)

print(f"\n[INFO] Evaluating model on Wine dataset using k = {best_k}...")
knn_wine = KNNClassifier(k=best_k)
knn_wine.fit(X_train_w, y_train_w)
preds_wine = knn_wine.predict(X_test_w)

acc_wine = np.sum(preds_wine == y_test_w) / len(y_test_w)
print(f"\n✅ Final Accuracy on Wine Dataset: {acc_wine:.4f} ({acc_wine*100:.2f}%)")
print("--- [END] Generalization Finished ---")
