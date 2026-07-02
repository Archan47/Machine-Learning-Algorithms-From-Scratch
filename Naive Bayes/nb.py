import numpy as np
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import plotly.express as px
from sklearn.naive_bayes import GaussianNB as SkGaussianNb
import plotly.graph_objects as go


class GaussianNB:

    def fit(self,X,y):
        X,y = np.asarray(X), np.asarray(y)
        self._classes_ = np.unique(y)
        n_features = X.shape[1]
        n_classes = len(self._classes_)

        self._mean = np.zeros((n_classes,n_features))
        self._variance = np.zeros((n_classes,n_features))
        self._priors = np.zeros(n_classes)

        # now iterating through all the classes i = index and c = class
        for i, c in enumerate(self._classes_):
            X_class = X[y == c]
            self._mean[i] =  X_class.mean(axis=0)
            self._variance[i] = X_class.var(axis=0)
            self._priors[i] = X_class.shape[0] / X.shape[0]
        
        return self
    
    #log N(x | μ, σ²) = -0.5 * log(2π σ²)  -  (x - μ)² / (2σ²)

    def log_gaussian(self, x):

        diff = x[:, None, :] - self._mean          # broadcasts to (n_samples, n_classes,n_features)
        num = - (diff ** 2) / (2 * self._variance)
        log_prob = num - 0.5 * np.log(2 * np.pi * self._variance)
        return log_prob.sum(axis=2)


    def predict(self,X):
        X = np.asarray(X)
        log_likelihood = self.log_gaussian(X)
        log_prior = np.log(self._priors)
        
        return self._classes_[np.argmax(log_likelihood + log_prior, axis=1)]



X, y = load_breast_cancer(return_X_y=True)
print(len(X))
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

nb_classifier = GaussianNB().fit(X_train,y_train)

my_pred = nb_classifier.predict(X_test)

print(f"My model accuracy : {accuracy_score(my_pred,y_test)}")


# Scikit-learn implementation
sk_nb = SkGaussianNb()
sk_nb.fit(X_train, y_train)
sk_pred = sk_nb.predict(X_test)

print("Sklearn Accuracy: ", accuracy_score(y_test, sk_pred))

same_predictions = np.mean(my_pred == sk_pred)

print("Prediction Agreement:", same_predictions)

#----------------------------- Learning Curve comparison -------------------------------------------

train_sizes = np.linspace(0.1, 1.0, 10)

my_train_scores = []
my_test_scores = []

sk_train_scores = []
sk_test_scores = []

for frac in train_sizes:

    n = int(frac * len(X_train))

    X_sub = X_train[:n]
    y_sub = y_train[:n]

    #My Gaussian NB
    my_model = GaussianNB()
    my_model.fit(X_sub, y_sub)

    my_train_scores.append(
        accuracy_score(y_sub, my_model.predict(X_sub))
    )

    my_test_scores.append(
        accuracy_score(y_test, my_model.predict(X_test))
    )

    #Sklearn Gaussian NB
    sk_model = SkGaussianNb()
    sk_model.fit(X_sub, y_sub)

    sk_train_scores.append(
        accuracy_score(y_sub, sk_model.predict(X_sub))
    )

    sk_test_scores.append(
        accuracy_score(y_test, sk_model.predict(X_test))
    )


fig1 = go.Figure()

fig1.add_trace(go.Scatter(
    x=(train_sizes * len(X_train)).astype(int),
    y=my_train_scores,
    mode="lines+markers",
    name="Train Accuracy"
))

fig1.add_trace(go.Scatter(
    x=(train_sizes * len(X_train)).astype(int),
    y=my_test_scores,
    mode="lines+markers",
    name="Test Accuracy"
))

fig1.update_layout(
    title="Learning Curve - My Gaussian Naive Bayes",
    xaxis_title="Training Samples",
    yaxis_title="Accuracy",
    template="plotly_white"
)

fig1.show()

fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=(train_sizes * len(X_train)).astype(int),
    y=sk_train_scores,
    mode="lines+markers",
    name="Train Accuracy"
))

fig2.add_trace(go.Scatter(
    x=(train_sizes * len(X_train)).astype(int),
    y=sk_test_scores,
    mode="lines+markers",
    name="Test Accuracy"
))

fig2.update_layout(
    title="Learning Curve - Sklearn Gaussian Naive Bayes",
    xaxis_title="Training Samples",
    yaxis_title="Accuracy",
    template="plotly_white"
)

fig2.show()