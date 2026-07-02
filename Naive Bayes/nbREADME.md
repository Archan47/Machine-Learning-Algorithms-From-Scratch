# I Built Gaussian Naive Bayes From Scratch — Here's What the Numbers Actually Taught Me

### Less than 50 lines of Python, a 96.5% accurate cancer classifier, and a learning curve that tells a surprisingly honest story.

---

Most machine learning tutorials hand you a `model.fit()` call and move on. This one goes the other direction — we're going to build Gaussian Naive Bayes ourselves, run it against scikit-learn's version on the breast cancer dataset, and then look at what the learning curves actually reveal about how this algorithm thinks.

No prior ML experience needed. If you know what a mean is and you've written a Python class before, you're good.

---

## Why Naive Bayes? Why Gaussian?

You might already know the basic Naive Bayes idea: use Bayes' Theorem to compute the probability of each class, multiply in some per-feature likelihoods, and pick the winner. That works perfectly when your features are categorical — like weather labels (Sunny/Rainy/Overcast) in the classic Play Tennis example.

But what happens when your features are continuous numbers — like tumor measurements in millimetres? You can't just count how many training samples had `mean_radius = 14.23`. That exact value might never repeat.

**Gaussian Naive Bayes** solves this by fitting a normal distribution (bell curve) to each feature, per class. Once you have that fitted curve, you can plug in *any* value and get back a probability density. That density is then used as the likelihood, and the rest of the algorithm proceeds exactly as before.

The log-likelihood of a single value `x` given a Gaussian with mean `μ` and variance `σ²` is:

```
log N(x | μ, σ²) = -0.5 × log(2π σ²) - (x - μ)² / (2σ²)
```

Two things to notice here: first, the exponent contains `(x - μ)²` — the squared distance from the mean. Values close to the class mean score high; values far away score low. Second, we work in log-space throughout, which turns all those tiny multiplications into additions and avoids floating-point underflow.

---

## The Dataset

We're using the Wisconsin Breast Cancer dataset, which comes built into scikit-learn. It has **569 samples**, each described by **30 numerical features** — things like the mean radius, texture, and smoothness of a cell nucleus. The label is binary: malignant (`0`) or benign (`1`).

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 455 training samples, 114 test samples
```

---

## Building the Classifier

### Step 1: The `fit` method — just counting and averaging

During training, Gaussian Naive Bayes doesn't optimize anything. It just computes three things for each class:

- **Prior probability** — what fraction of training examples belong to this class
- **Mean** — the average value of each feature, within this class
- **Variance** — how spread out each feature is, within this class

```python
import numpy as np

class GaussianNB:
    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        self._classes_ = np.unique(y)
        n_features = X.shape[1]
        n_classes = len(self._classes_)

        self._mean     = np.zeros((n_classes, n_features))
        self._variance = np.zeros((n_classes, n_features))
        self._priors   = np.zeros(n_classes)

        for i, c in enumerate(self._classes_):
            X_class = X[y == c]
            self._mean[i]     = X_class.mean(axis=0)
            self._variance[i] = X_class.var(axis=0)
            self._priors[i]   = X_class.shape[0] / X.shape[0]

        return self
```

After `fit`, we have a `(2 × 30)` table of means and another of variances — that's literally the entire trained model. No weights, no loss function, no epochs.

---

### Step 2: The `log_gaussian` method — evaluating those bell curves

At prediction time, for each test sample we need to compute how well it fits the distribution of each class. This is where the Gaussian formula comes in:

```python
    def log_gaussian(self, x):
        # x shape: (n_samples, n_features)
        # self._mean shape: (n_classes, n_features)
        diff = x[:, None, :] - self._mean   # → (n_samples, n_classes, n_features)
        num  = -(diff ** 2) / (2 * self._variance)
        log_prob = num - 0.5 * np.log(2 * np.pi * self._variance)
        return log_prob.sum(axis=2)         # → (n_samples, n_classes)
```

The `x[:, None, :]` trick inserts a class axis so NumPy can broadcast the subtraction cleanly across all classes at once — no loops needed. After summing across features, we get one log-likelihood score per sample per class.

---

### Step 3: The `predict` method — adding the prior and picking a winner

```python
    def predict(self, X):
        X = np.asarray(X)
        log_likelihood = self.log_gaussian(X)
        log_prior      = np.log(self._priors)
        return self._classes_[np.argmax(log_likelihood + log_prior, axis=1)]
```

Adding `log_prior` (in log-space, this is addition, not multiplication) shifts the scores to account for class imbalance before we take the argmax. The class with the highest combined score wins.

---

## Results: Scratch vs. Scikit-Learn

```
My model accuracy   : 96.49%
Sklearn accuracy    : 97.37%
Prediction agreement: 99.12%
```

Two things stand out here. First, the accuracy gap between our implementation and sklearn's is tiny — just under 1%. For 114 test samples, that's literally one prediction different. Second, and more interesting: the two models agreed on **99.1%** of individual predictions. They're effectively the same algorithm; the marginal difference traces back to sklearn applying a small `var_smoothing` constant (`1e-9`) that stabilizes variance estimates on very flat features. Our implementation doesn't do that, which occasionally nudges one borderline sample the other way.

---

## Reading the Learning Curves

This is where things get genuinely interesting. A learning curve shows how accuracy changes as you give the model more training data — and the shape of that curve tells you things about the algorithm that a single accuracy number never could.

### Our Gaussian NB

![Learning Curve — Custom Implementation]

A few things jump out immediately:

**The test accuracy flatlines at 96.5% from around 150 samples onwards.** This is a signature behaviour of Naive Bayes — it's a generative model, so it just needs enough samples to estimate its Gaussians reliably. Once those estimates stabilize, adding more data barely changes anything.

**The train accuracy wiggles noisily and ends *below* the test accuracy.** At first glance this looks backwards — shouldn't a model always do better on the data it trained on? With Naive Bayes, not necessarily. The algorithm fits a single Gaussian per feature per class; when the real distribution of training data is messier than a bell curve, the Gaussian approximation can actually generalize better than it memorizes. The test set, being a fixed holdout, gives it a cleaner signal to lock onto.

**The early instability (50–150 samples)** comes from variance estimates being unreliable with small samples. With only 50 rows, one unusual sample can wildly shift the estimated spread of a feature.

### Sklearn's Gaussian NB

![Learning Curve — Scikit-Learn]

The sklearn curve is smoother overall, but tells the same fundamental story. The test ceiling is marginally higher (97.3%) and the train curve is less volatile, largely due to that variance smoothing constant doing quiet stabilizing work in the background.

One notable difference: sklearn's test accuracy actually *rises* slightly at the far right rather than flattening completely. This is because with more training data, the variance estimates themselves become better, which tightens the Gaussian fits and nudges a few borderline predictions the right way.

---

## What These Curves Tell You About the Algorithm

If you were looking at a learning curve for a decision tree or a neural network, you'd expect to see train accuracy stay high while test accuracy slowly climbs to meet it — the classic overfitting-to-generalization convergence. Naive Bayes doesn't really do that.

Because it makes a strong structural assumption (features are independent given the class, and each follows a Gaussian), it can't "overfit" in the traditional sense — there's no complexity to crank up. What it can do is have *misspecified* assumptions — the Gaussians might not be the right shape for some features. More data helps, but only up to the point where the estimates are stable. After that, the ceiling is set by the quality of the assumptions, not the quantity of data.

This is precisely why Naive Bayes is often described as a **high-bias, low-variance** model. It has a ceiling it can't break through without changing the model family — but it reaches that ceiling fast, and it's stable once it gets there.

---

## The Full Code

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB as SkGaussianNB
import plotly.graph_objects as go

class GaussianNB:
    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        self._classes_ = np.unique(y)
        n_features = X.shape[1]
        n_classes = len(self._classes_)
        self._mean     = np.zeros((n_classes, n_features))
        self._variance = np.zeros((n_classes, n_features))
        self._priors   = np.zeros(n_classes)
        for i, c in enumerate(self._classes_):
            X_class = X[y == c]
            self._mean[i]     = X_class.mean(axis=0)
            self._variance[i] = X_class.var(axis=0)
            self._priors[i]   = X_class.shape[0] / X.shape[0]
        return self

    def log_gaussian(self, x):
        diff     = x[:, None, :] - self._mean
        num      = -(diff ** 2) / (2 * self._variance)
        log_prob = num - 0.5 * np.log(2 * np.pi * self._variance)
        return log_prob.sum(axis=2)

    def predict(self, X):
        X = np.asarray(X)
        log_likelihood = self.log_gaussian(X)
        log_prior      = np.log(self._priors)
        return self._classes_[np.argmax(log_likelihood + log_prior, axis=1)]


X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb_classifier = GaussianNB().fit(X_train, y_train)
my_pred = nb_classifier.predict(X_test)
print(f"My model accuracy : {accuracy_score(my_pred, y_test):.4f}")

sk_nb = SkGaussianNB()
sk_nb.fit(X_train, y_train)
sk_pred = sk_nb.predict(X_test)
print(f"Sklearn accuracy  : {accuracy_score(sk_pred, y_test):.4f}")
print(f"Prediction agreement: {np.mean(my_pred == sk_pred):.4f}")

# Learning curves
train_sizes = np.linspace(0.1, 1.0, 10)
my_train_scores, my_test_scores = [], []
sk_train_scores, sk_test_scores = [], []

for frac in train_sizes:
    n = int(frac * len(X_train))
    X_sub, y_sub = X_train[:n], y_train[:n]

    my_model = GaussianNB().fit(X_sub, y_sub)
    my_train_scores.append(accuracy_score(y_sub, my_model.predict(X_sub)))
    my_test_scores.append(accuracy_score(y_test, my_model.predict(X_test)))

    sk_model = SkGaussianNB().fit(X_sub, y_sub)
    sk_train_scores.append(accuracy_score(y_sub, sk_model.predict(X_sub)))
    sk_test_scores.append(accuracy_score(y_test, sk_model.predict(X_test)))

x_axis = (train_sizes * len(X_train)).astype(int)

for title, train_s, test_s in [
    ("Learning Curve — My Gaussian Naive Bayes",    my_train_scores, my_test_scores),
    ("Learning Curve — Sklearn Gaussian Naive Bayes", sk_train_scores, sk_test_scores),
]:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_axis, y=train_s, mode="lines+markers", name="Train Accuracy"))
    fig.add_trace(go.Scatter(x=x_axis, y=test_s,  mode="lines+markers", name="Test Accuracy"))
    fig.update_layout(title=title, xaxis_title="Training Samples",
                      yaxis_title="Accuracy", template="plotly_white")
    fig.show()
```

---

## Key Takeaways

- Gaussian Naive Bayes extends the classic categorical version to continuous features by fitting a normal distribution per feature per class during training.
- The entire model is just a table of means, variances, and priors — nothing is optimised iteratively.
- Working in log-space converts the product of probabilities into a sum, which is faster and numerically stable.
- Building from scratch and matching 99% of sklearn's predictions confirms the implementation is correct — and that the 1% gap is an implementation detail (variance smoothing), not a conceptual one.
- The learning curves reveal the true personality of this algorithm: it plateaus early, stays stable, and its ceiling is determined by the quality of the Gaussian assumption rather than the volume of data.

The next time someone tells you that "simple models can't compete," point them at 96.5% accuracy on a 30-feature medical dataset — built in an afternoon, from scratch, with no gradient in sight.

---

*The complete code is available above. Try swapping the breast cancer dataset for a different one and watching how the learning curve shape changes — that alone teaches you more about bias-variance trade-offs than most textbooks.*