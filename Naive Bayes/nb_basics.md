# Naive Bayes Explained



---

**Naive Bayes** is usually one of the first "real" algorithms you meet that isn't just a black box. It's built entirely on a 250-year-old idea — **Bayes' Theorem** — and once you see it work through one example by hand, it stops feeling like magic and starts feeling like common sense.

In this post, we'll build it up from scratch using a classic toy dataset: deciding whether to **play tennis** based on the weather. By the end, you'll be able to compute a Naive Bayes prediction yourself, with a pen and paper, no library required.

---

## 1. It All Starts With Bayes' Theorem

Bayes' Theorem answers a very specific kind of question: *"Given that something happened, what's the probability something else caused it?"*

The formula looks like this:

```
P(B|A) = [ P(B) × P(A|B) ] / P(A)
```

In plain English: the probability of B being true, given that A happened, depends on three things — how likely B was *before* we knew anything (`P(B)`), how likely A is *if* B were true (`P(A|B)`), and how likely A is overall (`P(A)`).

That's it. The whole theorem is just a way of "flipping" a conditional probability around. Naive Bayes is what happens when you point this formula at a machine learning classification problem.

---

## 2. From Bayes' Theorem to a Classifier

**Naive Bayes** clasifier is a "probabilistic classifier based on applying Bayes theorem with strong (naive) independence assusmptions between features.

In a typical supervised classification problem, you have some input features (`X1, X2, X3, ... Xn`) and you're trying to predict a label `Y` — say, `Yes` or `No`.

Using Bayes' Theorem, we can write:

```
P(Y | X1, X2, ..., Xn) = [ P(Y) × P(X1, X2, ..., Xn | Y) ] / P(X1, X2, ..., Xn)
```

There's one big problem with this: calculating `P(X1, X2, ..., Xn | Y)` directly means figuring out how all the features behave *together*, for every class. With more than a couple of features, that becomes computationally painful — and you'd need a huge amount of data to estimate it reliably.

This is where the **"naive"** part comes in.

### The Naive Assumption

Naive Bayes makes a bold simplifying assumption: **every feature is independent of every other feature, given the class.** Knowing the temperature, in other words, doesn't change what you'd expect the humidity to be — once you already know whether the day was a "Yes" or "No" day.

This assumption is almost never *exactly* true in the real world. But it turns out to work remarkably well in practice, and it lets us simplify the messy joint probability into a simple product:

```
P(X1, X2, ..., Xn | Y) ≈ P(X1|Y) × P(X2|Y) × P(X3|Y) × ... × P(Xn|Y)
```

So the full formula becomes:

```
P(Y | X1, X2, ..., Xn) = [ P(Y) × P(X1|Y) × P(X2|Y) × ... × P(Xn|Y) ] / P(X1, X2, ..., Xn)
```

Here's a handy shortcut: when we're comparing `P(Yes | ...)` against `P(No | ...)`, the denominator `P(X1, X2, ..., Xn)` is **the same number in both cases.** It doesn't depend on the class at all — so we can completely ignore it and just compare:

```
P(Y | X1, ..., Xn)  ∝  P(Y) × P(X1|Y) × P(X2|Y) × ... × P(Xn|Y)
```

Whichever class scores higher wins. We only normalize back into real percentages at the very end, if we want them.

---

## 3. The Dataset: Play Tennis or Not?

Here's our training data — 14 days, each with some weather conditions and whether tennis was played:

| Day | Outlook | Temperature | Humidity | Wind | Play Tennis |
|-----|----------|-------------|----------|--------|--------------|
| D1 | Sunny | Hot | High | Weak | No |
| D2 | Sunny | Hot | High | Strong | No |
| D3 | Overcast | Hot | High | Weak | Yes |
| D4 | Rain | Mild | High | Weak | Yes |
| D5 | Rain | Cool | Normal | Weak | Yes |
| D6 | Rain | Cool | Normal | Strong | No |
| D7 | Overcast | Cool | Normal | Strong | Yes |
| D8 | Sunny | Mild | High | Weak | No |
| D9 | Sunny | Cool | Normal | Weak | Yes |
| D10 | Rain | Mild | Normal | Weak | Yes |
| D11 | Sunny | Mild | Normal | Strong | Yes |
| D12 | Overcast | Mild | High | Strong | Yes |
| D13 | Overcast | Hot | Normal | Weak | Yes |
| D14 | Rain | Mild | High | Strong | No |

To keep things simple (and match the worked example), we'll focus on just two features — **Outlook** and **Temperature** — as our `X1` and `X2`.

---

## 4. Step 1: Calculate the Prior Probabilities

Before looking at any weather conditions at all, how often did people play tennis in this data?

Out of 14 days: **9 said Yes, 5 said No.**

```
P(Yes) = 9/14
P(No)  = 5/14
```

These are called **priors** — your starting beliefs, before any new evidence comes in.

---

## 5. Step 2: Build the Likelihood Tables

Now we count how each feature value splits across Yes and No. These are called **likelihoods** — `P(feature value | class)`.

### Outlook

| Outlook | Yes | No | P(Outlook \| Yes) | P(Outlook \| No) |
|----------|-----|-----|------------------|-----------------|
| Sunny | 2 | 3 | 2/9 | 3/5 |
| Overcast | 4 | 0 | 4/9 | 0/5 |
| Rain | 3 | 2 | 3/9 | 2/5 |
| **Total** | **9** | **5** | | |

### Temperature

| Temperature | Yes | No | P(Temp \| Yes) | P(Temp \| No) |
|--------------|-----|-----|---------------|--------------|
| Hot | 2 | 2 | 2/9 | 2/5 |
| Mild | 4 | 2 | 4/9 | 2/5 |
| Cool | 3 | 1 | 3/9 | 1/5 |
| **Total** | **9** | **5** | | |

These two tables are all we need. They're the entire "model" — there's no gradient descent, no iterations, just counting.

---

## 6. Step 3: Predicting a New Day — "Sunny and Hot"

Suppose today's weather is **Outlook = Sunny, Temperature = Hot.** Should you play tennis?

We compute a score for each class using our simplified formula:

```
P(Yes | Sunny, Hot) ∝ P(Yes) × P(Sunny|Yes) × P(Hot|Yes)
                     = (9/14) × (2/9) × (2/9)
                     = 2/63
                     ≈ 0.0317

P(No | Sunny, Hot)  ∝ P(No) × P(Sunny|No) × P(Hot|No)
                     = (5/14) × (3/5) × (2/5)
                     = 3/35
                     ≈ 0.0857
```

### Step 4: Normalize Into Real Probabilities

These raw scores aren't proper probabilities yet (they don't add up to 1), so we normalize by dividing each by their sum:

```
P(Yes | Sunny, Hot) = 0.0317 / (0.0317 + 0.0857) ≈ 27%
P(No  | Sunny, Hot) = 0.0857 / (0.0317 + 0.0857) ≈ 73%
```

**No** scores higher. The model's prediction: **don't play tennis today.**

That's the entire algorithm. No neural networks, no fancy optimizers — just multiplying probabilities together and comparing.

---

## 7. A Twist: What About "Overcast and Mild"?

Let's try a second day: **Outlook = Overcast, Temperature = Mild.**

```
P(Yes | Overcast, Mild) ∝ (9/14) × (4/9) × (4/9) = 8/63 ≈ 0.127

P(No | Overcast, Mild)  ∝ (5/14) × (0/5) × (2/5) = 0
```

Notice what happened: `P(Overcast | No)` is **zero**, because in our entire 14-day dataset, it was *never* overcast on a "No" day. The moment a single factor in the product is zero, the *entire* No score collapses to zero — no matter how likely the other features were.

After normalizing, this gives **100% Yes, 0% No.** The model is certain you should play tennis.

### The catch: the "zero-frequency problem"

This result happens to make intuitive sense here — overcast days really were great tennis days in this data. But mathematically, this is a known weakness of Naive Bayes called the **zero-frequency problem.** If a feature value simply never appeared with a particular class in training — even just because the dataset was small — the model will declare that combination *impossible* forever, even if more data would have proven otherwise.

The standard fix is called **Laplace smoothing** (or add-one smoothing): you add a small constant (usually 1) to every count before computing probabilities, so nothing is ever exactly zero. For example, with Outlook having 3 possible values:

```
P(Overcast | No) without smoothing = 0/5        = 0
P(Overcast | No) with smoothing    = (0+1)/(5+3) = 0.125
```

It's a small adjustment, but it keeps the model from making overconfident, brittle predictions on combinations it simply hasn't seen enough of.

---

## 8. Why Call It "Naive"?

Just to drive the point home: Naive Bayes assumes the Outlook and Temperature of a day tell you nothing about each other once you know whether tennis was played. In reality, weather features are often correlated — hot days might be more likely to be sunny, for instance. The algorithm doesn't care. It treats every feature as if it's voting completely independently, multiplies all those "votes" together, and picks the class with the highest score.

This assumption is the "naive" part — and it's almost never perfectly true. Yet despite that, Naive Bayes remains a genuinely strong baseline for text classification, spam filtering, and sentiment analysis, largely because it's fast, needs very little data, and the independence assumption hurts the *final decision* less often than you'd expect.

---

## Key Takeaways

- **Bayes' Theorem** flips a conditional probability around: `P(B|A)` in terms of `P(A|B)`.
- **Naive Bayes** applies this to classification by assuming features are independent given the class — turning one hard joint probability into a simple product of easy ones.
- You don't need to compute the denominator at all — it's constant across classes, so just compare the numerators.
- **Priors** (`P(Yes)`, `P(No)`) come from the overall class split; **likelihoods** (`P(feature|class)`) come from counting within each class.
- Multiply prior × likelihoods for each class, then normalize at the end if you want actual percentages.
- Watch out for the **zero-frequency problem** — a single unseen feature/class combination can zero out an entire prediction. **Laplace smoothing** fixes this.

Once you've worked through one example by hand like this, reading a Naive Bayes implementation in scikit-learn (or writing your own from scratch) becomes much less intimidating — it's really just these tables and a few multiplications, done quickly and at scale.

---
