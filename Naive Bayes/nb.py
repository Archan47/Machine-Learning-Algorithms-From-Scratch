import numpy as np
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import plotly.express as px

                                                     
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
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

nb_classifier = GaussianNB().fit(X_train,y_train)

y_pred = nb_classifier.predict(X_test)

print(accuracy_score(y_pred,y_test))

# results = pd.DataFrame({
#     "Actual": y_test,
#     "Predicted": y_pred
# })

# fig = px.scatter(
#     results,
#     x="Actual",
#     y="Predicted"
# )

# fig.show()
