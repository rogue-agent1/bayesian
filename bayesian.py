#!/usr/bin/env python3
"""Bayesian inference utilities. Zero dependencies."""

def bayes_theorem(prior, likelihood, evidence):
    return prior * likelihood / evidence if evidence > 0 else 0

def bayes_update(prior, likelihood_true, likelihood_false):
    evidence = prior * likelihood_true + (1 - prior) * likelihood_false
    return bayes_theorem(prior, likelihood_true, evidence)

class NaiveBayesClassifier:
    def __init__(self, alpha=1.0):
        self.alpha = alpha; self.class_counts = {}
        self.feature_counts = {}; self.total = 0

    def fit(self, X, y):
        self.total = len(y)
        vocab = set()
        for xi in X:
            for f in xi: vocab.add(f)
        self.vocab_size = len(vocab)
        for xi, yi in zip(X, y):
            self.class_counts[yi] = self.class_counts.get(yi, 0) + 1
            if yi not in self.feature_counts: self.feature_counts[yi] = {}
            for f in xi:
                self.feature_counts[yi][f] = self.feature_counts[yi].get(f, 0) + 1
        return self

    def predict(self, features):
        import math
        best_class = None; best_score = float("-inf")
        for c in self.class_counts:
            score = math.log(self.class_counts[c] / self.total)
            total_features = sum(self.feature_counts[c].values())
            for f in features:
                count = self.feature_counts[c].get(f, 0)
                score += math.log((count + self.alpha) / (total_features + self.alpha * self.vocab_size))
            if score > best_score: best_score = score; best_class = c
        return best_class

def beta_posterior(prior_a, prior_b, successes, failures):
    return prior_a + successes, prior_b + failures

def beta_mean(a, b):
    return a / (a + b)

def credible_interval(a, b, level=0.95):
    """Approximate credible interval using normal approximation."""
    import math
    mean = a / (a + b)
    var = (a * b) / ((a + b)**2 * (a + b + 1))
    z = 1.96 if level == 0.95 else 2.576
    return (mean - z * math.sqrt(var), mean + z * math.sqrt(var))

if __name__ == "__main__":
    # Medical test: P(disease|positive)
    p = bayes_update(prior=0.01, likelihood_true=0.95, likelihood_false=0.05)
    print(f"P(disease|positive): {p:.4f}")
