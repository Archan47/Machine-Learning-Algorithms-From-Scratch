# Building a Decision Tree From Scratch

Decision trees are one of the most intuitive machine learning models — arguably the most intuitive. There's no gradient descent, no distance metric, no matrix multiplication. Just a sequence of yes/no questions that eventually land you on a prediction. That simplicity is exactly why it's a good algorithm to implement from scratch: every part of it maps to a formula you can compute by hand.

This is the third entry in my "ML from scratch" series, after KNN and Gaussian Naive Bayes. This time: a decision tree classifier built with nothing but NumPy, walked through calculation by calculation.

## The Core Idea

At every node in the tree, we ask one question: *"is feature X ≤ some threshold?"* Samples that answer yes go left, samples that answer no go right. We keep doing this, recursively, until the samples left at a node are (almost) all the same class — at which point that node becomes a **leaf** that just predicts the majority class.

The entire algorithm boils down to one problem, repeated at every node: **out of all possible (feature, threshold) questions we could ask, which one splits the data best?**

To answer that, we need a way to measure how "good" a split is. That's entropy and information gain.

## Entropy: Measuring Disorder

Entropy comes from information theory, and it measures how mixed up a set of labels is:

```
E(S) = -Σ p(x) · log(p(x))
```

Where `p(x)` is the proportion of class `x` in set `S`, and the sum runs over all classes.

**Worked example.** Say a node has 10 samples: 6 belong to class 0, 4 belong to class 1.

```
p(0) = 6/10 = 0.6
p(1) = 4/10 = 0.4

E = -(0.6 · log(0.6) + 0.4 · log(0.4))
  = -(0.6 · (-0.511) + 0.4 · (-0.916))
  = -(-0.3065 + -0.3665)
  = 0.673
```

Compare that to a perfectly pure node — 10 samples, all class 0:

```
p(0) = 1.0
E = -(1.0 · log(1.0)) = -(1.0 · 0) = 0
```

Entropy of 0 means zero disorder — the node is pure, no more splitting needed. Entropy is at its **maximum** when classes are perfectly balanced (a 50/50 split for binary classification gives `E = log(2) ≈ 0.693` using natural log), and decreases toward 0 as one class comes to dominate. That's the whole intuition: entropy is a disorder score, and we want splits that drive it down.

In code, this is a direct translation:

```python
def _entropy(self, y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log(p) for p in ps if p > 0])
```

`np.bincount(y)` counts how many samples belong to each class label, dividing by `len(y)` turns those counts into proportions `p(x)`, and the `if p > 0` guard exists because `log(0)` is undefined (and a class with 0 samples contributes nothing to entropy anyway).

## Information Gain: Scoring a Split

Entropy tells us how mixed a *single* set of labels is. Information gain tells us how much a *split* reduces that mixing, by comparing the parent's entropy to a weighted average of the two children's entropy:

```
IG = E(parent) - [ (n_left/n) · E(left) + (n_right/n) · E(right) ]
```

The weighting by `n_left/n` and `n_right/n` matters — a split that produces a large, pure partition and a tiny, mixed one should score differently than a split producing two medium, moderately pure partitions. Weighting by size accounts for that.

**Worked example**, continuing the 10-sample node above (6 class-0, 4 class-1, parent entropy `E = 0.673`). Suppose a candidate split sends 5 samples left (all class 0) and 5 samples right (1 class-0, 4 class-1):

```
E(left)  = 0                     (pure)
E(right) = -(0.2·log(0.2) + 0.8·log(0.8))
         = -(0.2·(-1.609) + 0.8·(-0.223))
         = -(-0.322 + -0.179)
         = 0.500

Weighted child entropy = (5/10)·0 + (5/10)·0.500 = 0.250

IG = 0.673 - 0.250 = 0.423
```

That's a strong split — it isolated a pure group and left a much cleaner remainder. A weak split (say, one that barely changes the class ratio on either side) would produce an `IG` close to 0.

In code:

```python
def _information_gain(self, y, X_column, threshold):
    parent_entropy = self._entropy(y)
    left_idxs, right_idxs = self._split(X_column, threshold)

    if len(left_idxs) == 0 or len(right_idxs) == 0:
        return 0

    n = len(y)
    n_l, n_r = len(left_idxs), len(right_idxs)
    e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
    child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

    return parent_entropy - child_entropy
```

The early return handles the degenerate case where a threshold doesn't actually split anything (everything goes to one side) — that split contributes nothing, so we score it 0 rather than divide by an empty set.

## Finding the Best Split

Now the search problem: for a given node, which (feature, threshold) pair maximizes information gain? The approach is exhaustive — try every feature, and for each feature, try every unique value in that feature's column as a candidate threshold:

```python
def best_split(self, X, y, feat_idxs):
    best_gain = -1
    split_idx, split_threshold = None, None

    for feat_idx in feat_idxs:
        X_column = X[:, feat_idx]
        thresholds = np.unique(X_column)

        for thr in thresholds:
            gain = self._information_gain(y, X_column, thr)
            if gain > best_gain:
                best_gain = gain
                split_idx = feat_idx
                split_threshold = thr

    return split_idx, split_threshold
```

Why only *unique* values as thresholds, rather than scanning a continuous range? Because the only thresholds that can possibly change which samples fall left vs. right are the actual observed feature values — anything between two consecutive observed values produces an identical split. So checking every unique value covers every meaningfully different split, with no wasted computation.

This is also where **feature subsampling** happens — `feat_idxs` can be a random subset of all available features rather than every feature, controlled by the `n_features` parameter. On its own this just adds randomness to a single tree, but it's the exact mechanism a Random Forest relies on to decorrelate the trees in its ensemble.

The splitting itself is a simple boolean mask:

```python
def _split(self, X_column, split_thresh):
    left_idxs = np.argwhere(X_column <= split_thresh).flatten()
    right_idxs = np.argwhere(X_column > split_thresh).flatten()
    return left_idxs, right_idxs
```

## Growing the Tree

With `best_split` in hand, growing the tree is straightforward recursion. At each call, first check whether we should stop:

```python
def grow_tree(self, X, y, depth=0):
    n_samples, n_feats = X.shape
    n_labels = len(np.unique(y))

    if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
        leaf_value = self.most_common_label(y)
        return Node(value=leaf_value)

    feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
    best_feature, best_threshold = self.best_split(X, y, feat_idxs)

    left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
    left = self.grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
    right = self.grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)

    return Node(best_feature, best_threshold, left, right)
```

Three stopping conditions, any one of which turns a node into a leaf:

- **`depth >= max_depth`** — a depth cap, to prevent trees from growing indefinitely and overfitting to individual samples.
- **`n_labels == 1`** — the node is already pure; there's nothing left to split.
- **`n_samples < min_samples_split`** — too few samples left to justify splitting further (and to avoid splits based on statistical noise from tiny subsets).

If none of those trigger, the node picks its best split, partitions the data, and recurses on each partition with `depth + 1`. The recursion bottoms out at leaves, and each call builds and returns its own `Node` — which is what lets the parent call wire `left` and `right` into its own returned node, all the way back up to the root.

## Making Predictions

Once built, prediction is just tree traversal — no computation, only comparisons:

```python
def predict(self, X):
    return np.array([self._traverse_tree(x, self.root) for x in X])

def _traverse_tree(self, x, node):
    if node.is_leaf_node():
        return node.value

    if x[node.feature] <= node.threshold:
        return self._traverse_tree(x, node.left)

    return self._traverse_tree(x, node.right)
```

For each sample, start at the root. If the current node is a leaf, return its stored class label. Otherwise, compare the sample's value for `node.feature` against `node.threshold`, and recurse left or right accordingly. This is O(depth) per sample — one of the appeals of decision trees at inference time, regardless of how large the training set was.

## Results

Trained and evaluated on `sklearn.datasets.load_breast_cancer` (80/20 split, `random_state=1234`, default hyperparameters):

**Accuracy: 94%**

That's competitive with sklearn's own `DecisionTreeClassifier` on the same split — expected, since the underlying math (entropy, information gain, greedy recursive splitting) is identical. sklearn's version is faster because its split-finding is more optimized, and it supports additional criteria (Gini impurity) and pruning options this implementation doesn't.

To go beyond the accuracy number, I plotted a confusion matrix to see where the errors landed, a feature-importance chart based on how often each feature was used to split (which features the tree actually relied on), and a PCA projection of the test set colored by correct vs. incorrect predictions, to see whether misclassifications clustered in any particular region of the data.

## What This Implementation Doesn't Do

Worth being explicit about the gaps, since they're exactly what separates a from-scratch learning exercise from a production-ready implementation:

- **No pruning.** A fully-grown tree (especially with `max_depth=100`) can overfit badly on noisier datasets. Real implementations add pre-pruning (stricter stopping criteria) or post-pruning (cost-complexity pruning, i.e. `ccp_alpha` in sklearn).
- **No Gini impurity option.** Entropy is one valid splitting criterion; Gini impurity is another, cheaper to compute (no logarithms) and commonly used as the default in practice.
- **Exhaustive threshold search.** Trying every unique value in every feature is fine for a learning implementation, but scales poorly — sklearn's C-optimized version does the same search far more efficiently.
- **No feature importance built into the model itself** — the version I computed for the plots above is a simple split-frequency count, not the weighted (samples × information gain) importance sklearn reports.

## Next Steps

The natural extension from here is a **Random Forest** — since `n_features` subsampling is already built in, most of the ensemble machinery is really just: train many of these trees on bootstrapped samples of the data, and aggregate their predictions by majority vote. That's next in this series.

*Code for this and the rest of the "ML from scratch" series is on GitHub.*