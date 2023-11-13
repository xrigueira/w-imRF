# Load a Random Forest model
import pickle
filename = f'models/rf_model_med_{1}.sav'
loaded_model_med = pickle.load(open(filename, 'rb'))

# Access on of its estimators (trees)
tree = loaded_model_med.estimators_[1]

# Get the distance from each note to the root node (depth)
print(tree.tree_.compute_node_depths())

# I think that the key to clusteting would be getting the distance from each node to the others, not just to the root node.
# Then I could try to apply a clustering algorithm.

# Also I have to look into using dendograms as David said.

# Random Forest docs: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# Decision Tree Classifier (estimator_) docs: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
# This is key: Decision Tree Structure docs: https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
import numpy as np

# Create a synthetic dataset (replace this with your actual data)
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_clusters_per_class=1, random_state=42)

# Assume you have trained a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# Assuming anomalies are instances where the predicted class is 1 (adjust as needed)
anomaly_indices = np.where(y == 1)[0]

# Create an empty list to store decision paths
decision_paths = []

# Traverse each tree in the Random Forest
for tree in clf.estimators_:
    # Get the decision path for each anomaly instance
    tree_decision_paths = tree.decision_path(X[anomaly_indices]).toarray()
    decision_paths.append(tree_decision_paths)

# Concatenate the decision paths along the columns
all_decision_paths = np.concatenate(decision_paths, axis=1)

# Convert the binary decision paths to integers
integer_decision_paths = all_decision_paths.dot(1 << np.arange(all_decision_paths.shape[-1] - 1, -1, -1))

# Reshape the integer decision paths to have one row per anomaly instance
integer_decision_paths = integer_decision_paths.reshape(len(anomaly_indices), -1)
print(integer_decision_paths)
# Apply K-Means clustering
num_clusters = 3  # Adjust as needed
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_assignments = kmeans.fit_predict(integer_decision_paths)

# Print the cluster assignments for each anomaly instance
for i, cluster in enumerate(cluster_assignments):
    print(f"Anomaly instance {anomaly_indices[i]} belongs to cluster {cluster}")
