from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from knn import KNN


iris = load_iris()

X = iris.data
y = iris.target

#print(X[:5])
#print(y[:5])

# train_X = X[:120]
# train_y = y[:120]

# test_X = X[120:]
# test_y = y[120:]


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------- This is for my custom KNN Model -------------------------------------
points = {
    "Setosa": [],
    "Versicolor": [],
    "Virginica": []
}

for features, label in zip(X_train, y_train):

    if label == 0:
        points["Setosa"].append(features.tolist())

    elif label == 1:
        points["Versicolor"].append(features.tolist())

    else:
        points["Virginica"].append(features.tolist())

classifier = KNN(k=5)
classifier.fit(points)

correct = 0

for flower, actual in zip(X_test,y_test):

    prediction = classifier.predict(flower)

    actual_name = iris.target_names[actual]

    if prediction.lower() == actual_name.lower():
        correct += 1

accuracy = correct / len(y_test)

print(f"Custom KNN Accuracy: {accuracy:.2f}")

print(prediction)


# ----------------------- Now Implementing scikit-learn KNN classifier -----------------------------

print()
print("Implementing KNN from sklearn")

skl_knn_model = KNeighborsClassifier()
skl_knn_model.fit(X_train,y_train)

skl_knn_model_predictions = skl_knn_model.predict(X_test)

skl_knn_model_accuracy = accuracy_score(y_test,skl_knn_model_predictions)

print(f"Sklearn KNN Accuracy: {skl_knn_model_accuracy:.2f}")
