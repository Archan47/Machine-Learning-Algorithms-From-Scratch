from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
import numpy as np
import plotly.graph_objects as go
from tree import DecisionTree


data = datasets.load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names
class_names = data.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf = DecisionTree()
clf.fit(X_train, y_train)

clf_prediction = clf.predict(X_test)
clf_accuracy = accuracy_score(y_test, clf_prediction)

print(f"Decision Tree Accuracy: {clf_accuracy:.2f}")

# ---------- Plot 1: Confusion Matrix ----------
cm = confusion_matrix(y_test, clf_prediction)

fig_cm = go.Figure(data=go.Heatmap(
    z=cm,
    x=[f"Predicted: {c}" for c in class_names],
    y=[f"Actual: {c}" for c in class_names],
    text=cm,
    texttemplate="%{text}",
    colorscale="Blues"
))
fig_cm.update_layout(
    title=f"Confusion Matrix (Accuracy: {clf_accuracy:.2%})",
    xaxis_title="Predicted label",
    yaxis_title="True label"
)
fig_cm.show()

# ---------- Plot 2: Feature importance (split frequency) ----------
def compute_feature_importance(node, importance):
    if node is None or node.is_leaf_node():
        return
    importance[node.feature] += 1
    compute_feature_importance(node.left, importance)
    compute_feature_importance(node.right, importance)

importance_counts = np.zeros(X.shape[1])
compute_feature_importance(clf.root, importance_counts)

importance_norm = (
    importance_counts / importance_counts.sum()
    if importance_counts.sum() > 0
    else importance_counts
)

top_n = 10
sorted_idx = np.argsort(importance_norm)[::-1][:top_n]

fig_importance = go.Figure(go.Bar(
    x=importance_norm[sorted_idx][::-1],
    y=[feature_names[i] for i in sorted_idx][::-1],
    orientation="h"
))
fig_importance.update_layout(
    title=f"Feature Importance — top {top_n} (by split frequency)",
    xaxis_title="Relative importance",
    yaxis_title="Feature"
)
fig_importance.show()

# ---------- Plot 3: Correct vs. incorrect predictions (PCA projection) ----------
pca = PCA(n_components=2)
X_test_2d = pca.fit_transform(X_test)
correct = clf_prediction == y_test

fig_pca = go.Figure()
fig_pca.add_trace(go.Scatter(
    x=X_test_2d[correct, 0], y=X_test_2d[correct, 1],
    mode="markers", name="Correct",
    marker=dict(color="green", size=8, opacity=0.7)
))
fig_pca.add_trace(go.Scatter(
    x=X_test_2d[~correct, 0], y=X_test_2d[~correct, 1],
    mode="markers", name="Incorrect",
    marker=dict(color="red", size=11, symbol="x")
))
fig_pca.update_layout(
    title="Test Set Predictions — PCA Projection",
    xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
    yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)"
)
fig_pca.show()