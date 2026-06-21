from sklearn.datasets import load_iris
from knn import KNN

iris = load_iris()

X = iris.data
y = iris.target

#print(X[:5])
#print(y[:5])

train_X = X[:120]
train_y = y[:120]

test_X = X[120:]
test_y = y[120:]

points = {
    "Setosa": [],
    "Versicolor": [],
    "Virginica": []
}

for features, label in zip(train_X, train_y):

    if label == 0:
        points["Setosa"].append(features.tolist())

    elif label == 1:
        points["Versicolor"].append(features.tolist())

    else:
        points["Virginica"].append(features.tolist())

classifier = KNN(k=5)
classifier.fit(points)

correct = 0

for flower, actual in zip(test_X, test_y):

    prediction = classifier.predict(flower)

    actual_name = iris.target_names[actual]

    if prediction.lower() == actual_name.lower():
        correct += 1

accuracy = correct / len(test_y)

print(f"Accuracy: {accuracy:.2f}")

print(prediction)