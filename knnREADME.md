# K-Nearest Neighbors From Scratch: Understanding Every Line, Then Benchmarking It Against Scikit-Learn

K-Nearest Neighbors (KNN) is usually the first classification algorithm people meet in machine learning — and it's a good one to start with, because there's no hidden math you have to take on faith. Every step is something you can trace by hand on paper. In this post, I'll build KNN from scratch in Python, explain exactly what each piece of code is doing, then put my implementation head-to-head against scikit-learn's built-in `KNeighborsClassifier` on a real dataset.

## The Core Idea, In Plain English

Imagine you move to a new neighborhood and want to guess whether it's a "quiet" or "busy" area. A reasonable strategy: look at the 5 houses closest to yours, see how most of them would describe the area, and go with the majority.

That's the entirety of KNN:

1. **Measure distance** from a new point to every point you already have labels for.
2. **Pick the k closest ones** — your "neighbors."
3. **Let them vote.** Whichever label is most common among the neighbors becomes the prediction.

There's no training step where the algorithm "learns" weights or coefficients. It just memorizes the data and does all its work at prediction time. This is why KNN is called a **lazy learner**.

## Step 1: Teaching the Algorithm to Measure Distance

Before you can find "nearest" neighbors, you need a definition of "near." I used **Euclidean distance** — the same straight-line distance formula from the Pythagorean theorem, just extended to handle more than 2 dimensions:

```python
def euclideanDistance(p, q):
    distance = np.sqrt(np.sum((np.array(p) - np.array(q)) ** 2))
    return distance
```

Here's what's actually happening on that one line:

- `np.array(p) - np.array(q)` subtracts the two points feature-by-feature. If `p = [2, 4]` and `q = [5, 6]`, this gives `[-3, -2]` — the difference along each axis.
- `** 2` squares each of those differences, so `[-3, -2]` becomes `[9, 4]`. Squaring removes the negative sign and emphasizes larger gaps.
- `np.sum(...)` adds those squared differences together: `9 + 4 = 13`.
- `np.sqrt(...)` takes the square root of that sum, giving the final distance: `√13 ≈ 3.61`.

This is exactly the Pythagorean theorem (`c = √(a² + b²)`), just written so it works for 2 features, 4 features, or 40 features without changing the code.

## Step 2: The KNN Class Itself

```python
class KNN:
    def __init__(self, k):
        self.k = k
        self.point = None

    def fit(self, points):
        self.points = points

    def predict(self, new_point):
        distances = []

        for category in self.points:
            for point in self.points[category]:
                d = euclideanDistance(point, new_point)
                distances.append([d, category, point])

        categories = [category[1] for category in sorted(distances)[:self.k]]

        result = Counter(categories).most_common(1)[0][0]
        return result
```

Let's go through this piece by piece.

**`__init__`** just stores `k` — the number of neighbors to consult — as an attribute, so it's available later inside `predict`.

**`fit`** stores the training data. Notice it doesn't *do* anything with it — no math, no transformation. It just remembers it. This is the entire "training" phase of KNN, and it's why fitting a KNN model takes basically no time even on large datasets.

**`predict`** is where everything actually happens, and it runs in three stages:

1. **Distance to everyone.** The nested loop walks through every category (`"Setosa"`, `"Versicolor"`, etc.) and every point stored under that category, computing the distance from `new_point` to each one. Each result is stored as a list: `[distance, category, point]`.
2. **Sort and trim.** `sorted(distances)` sorts that whole list — and because each entry is `[distance, category, point]`, Python sorts by the *first* element (distance) automatically. Slicing `[:self.k]` keeps only the `k` smallest distances — the nearest neighbors.
3. **Vote.** `categories` pulls out just the category labels from those `k` nearest entries. `Counter(categories).most_common(1)[0][0]` counts how often each label appears and returns the single most frequent one. That's the prediction.

If you've ever wondered what "distance-based voting" means in an algorithm description, this `predict` method *is* that sentence translated directly into code.

## Step 3: Testing on Real Data — The Iris Dataset

To check whether this actually works, I tested it on the **Iris dataset**: 150 flower measurements (petal length, petal width, sepal length, sepal width) labeled as one of three species — Setosa, Versicolor, or Virginica.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

A few details worth slowing down on, since they matter more than they look:

- **`test_size=0.2`** holds back 20% of the data for testing — the model never sees these points during "training," so evaluating on them is a fair test of generalization.
- **`random_state=42`** just makes the split reproducible. Without it, you'd get a different random split every time you ran the script, making your results hard to compare run-to-run.
- **`stratify=y`** is the detail people most often skip. Iris has 50 samples of each species. A *plain* random split could, by bad luck, put very few Setosa samples in the test set. `stratify=y` forces the split to preserve the original 50/50/50 proportions in both the train and test sets — so the test set is a fair, balanced sample of all three species.

My `KNN` class only understands dictionaries shaped like `{"category": [[features...], [features...]]}`, so the next step reshapes the Iris data into that format:

```python
points = {"Setosa": [], "Versicolor": [], "Virginica": []}

for features, label in zip(X_train, y_train):
    if label == 0:
        points["Setosa"].append(features.tolist())
    elif label == 1:
        points["Versicolor"].append(features.tolist())
    else:
        points["Virginica"].append(features.tolist())

classifier = KNN(k=5)
classifier.fit(points)
```

Iris labels its species as `0`, `1`, `2` by default — this loop just translates those numbers into the species names my class expects, then feeds the result into `fit()`.

## Step 4: Benchmarking Against Scikit-Learn

The real test of "did I implement this correctly" is comparing it against a trusted, production-grade implementation. So alongside my own classifier, I ran the exact same train/test split through scikit-learn's `KNeighborsClassifier`:

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

skl_knn_model = KNeighborsClassifier()
skl_knn_model.fit(X_train, y_train)
skl_knn_model_predictions = skl_knn_model.predict(X_test)
skl_knn_model_accuracy = accuracy_score(y_test, skl_knn_model_predictions)
```

`accuracy_score` does exactly what my own `evaluate` function did manually earlier — compares predictions to true labels and reports the fraction that matched — except it's scikit-learn's tested, optimized version of that same idea.

## The Results

My from-scratch implementation scored **100% accuracy** on the test split (`k=5`). Scikit-learn's classifier, trained and tested on the identical split, performed comparably well.

That result deserves a little honesty rather than just a victory lap: Iris is a famously easy, well-separated dataset, and a held-out test set of 30 samples is small enough that a perfect score isn't a huge surprise — it doesn't yet prove my implementation would hold up on messier, less separable data. The valuable takeaway isn't the specific number; it's that a roughly 20-line `predict` method, with no libraries doing the heavy lifting, lands in the same neighborhood (no pun intended) as a mature, widely-used implementation. That's a solid sign the underlying logic — distance, sorting, voting — is correct.

## What's Different Under the Hood

If the accuracy is similar, what is scikit-learn actually doing differently?

- **Search strategy.** My version checks the distance to *every* training point for every prediction — O(n) per query. Scikit-learn defaults to smarter structures like KD-trees or Ball trees that avoid checking every point, which matters a lot as datasets grow.
- **Tie-breaking and weighting.** Scikit-learn supports weighting neighbors by inverse distance (closer neighbors count more) and has defined behavior for ties. My version treats all `k` neighbors equally and lets `Counter` pick a winner if there's a tie.
- **Vectorization.** My distance loop runs in plain Python; scikit-learn's internals are vectorized and compiled, which is most of why it's faster at scale even with a similar algorithm.

None of this changes the *result* on a small, easy dataset like Iris — but it's exactly where the two implementations would start to diverge on something larger or messier.

## What's Next

Getting matching results against scikit-learn on Iris feels like a good checkpoint, but it's a checkpoint, not a finish line. Next, I want to push this implementation further: add some of the upgrades above (distance weighting, feature scaling, maybe a smarter search structure instead of brute-force), and then test it against a more complex, less cleanly-separated dataset than Iris — something with more features, more noise, or more classes — to see where the from-scratch version actually starts to struggle. That's where I expect to learn the most.

---

*Code for this implementation is on my GitHub — feel free to fork it and try the next round of improvements yourself.*