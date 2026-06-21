from collections import Counter

import numpy as np


points = {
    "Blue" : [[2,4], [1,3], [2,3], [3,2], [2,1], [2,2], [4,1] ],
    "Red" : [[5,6], [4,5], [4,6], [6,6], [5,4], [3,4], [12,10], ]
}

test_point = [15,12]

def euclideanDistance(p,q):
    
    disstance = np.sqrt(np.sum((np.array(p) - np.array(q)) ** 2))

    return disstance


class KNN:
    def __init__(self,k):
        self.k = k
        self.point = None

    def fit(self,points):
        self.points = points

    def predict(self,new_point):
        distances = []

        for category in self.points:
            for point in self.points[category]:
                d = euclideanDistance(point,new_point)
                distances.append([d,category,point])
                
        categories = [category[1] for category in sorted(distances)[:self.k]]

        # To know the category and distance of top k data points 

        distances.sort()

        for distance, category, point in distances[:self.k]:
            print(f"Point: {point}")
            print(f"Distance: {distance:.3f}")
            print(f"Category: {category}\n")

        result = Counter(categories).most_common(1)[0][0]
        
        return result

def evaluate(predictions, true_labels):
    correct = sum(p == t for p, t in zip(predictions, true_labels))
    total = len(true_labels)
    accuracy = correct / total
    precision = correct / len(predictions) if predictions else 0
    recall = correct / len(true_labels) if true_labels else 0
    return accuracy, precision, recall

classifier = KNN(k=3)
classifier.fit(points)
predictions = []
true_labels = []

for point, true_category in test_point:
    predicted_category = classifier.predict(point)
    print(f"Point: {point}")
    print(f"True Category: {true_category}")
    print(f"Predicted Category: {predicted_category}\n")
    predictions.append(predicted_category)
    true_labels.append(true_category)

accuracy, precision, recall = evaluate(predictions, true_labels)
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")


