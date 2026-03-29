#!/usr/bin/env python3
"""bayesian - Bayesian inference with conjugate priors."""
import sys, math

def bayes_theorem(prior, likelihood, evidence):
    return prior * likelihood / evidence if evidence > 0 else 0.0

class BetaBinomial:
    """Beta-Binomial conjugate prior for coin flips."""
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta
    def update(self, successes, failures):
        self.alpha += successes
        self.beta += failures
    def mean(self):
        return self.alpha / (self.alpha + self.beta)
    def mode(self):
        if self.alpha > 1 and self.beta > 1:
            return (self.alpha - 1) / (self.alpha + self.beta - 2)
        return self.mean()
    def variance(self):
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b)**2 * (a + b + 1))
    def credible_interval(self, level=0.95):
        # approximate using normal
        m = self.mean()
        s = math.sqrt(self.variance())
        z = 1.96 if abs(level - 0.95) < 0.01 else 2.576
        return max(0, m - z*s), min(1, m + z*s)

class NormalNormal:
    """Normal-Normal conjugate prior."""
    def __init__(self, mu0=0, sigma0=1):
        self.mu = mu0
        self.sigma = sigma0
    def update(self, data, data_sigma):
        n = len(data)
        data_mean = sum(data) / n
        precision0 = 1 / self.sigma**2
        precision_data = n / data_sigma**2
        new_precision = precision0 + precision_data
        self.mu = (precision0 * self.mu + precision_data * data_mean) / new_precision
        self.sigma = math.sqrt(1 / new_precision)

def test():
    # Bayes theorem: disease test
    # P(disease) = 0.01, P(+|disease) = 0.99, P(+|no disease) = 0.05
    p_pos = 0.01 * 0.99 + 0.99 * 0.05
    p_disease_given_pos = bayes_theorem(0.01, 0.99, p_pos)
    assert abs(p_disease_given_pos - 0.1667) < 0.01
    # Beta-Binomial
    bb = BetaBinomial(1, 1)  # uniform prior
    bb.update(7, 3)  # 7 heads, 3 tails
    assert abs(bb.mean() - 8/12) < 0.01
    lo, hi = bb.credible_interval()
    assert lo < bb.mean() < hi
    # Normal-Normal
    nn = NormalNormal(0, 10)  # vague prior
    nn.update([5, 6, 4, 5, 5], 1)
    assert abs(nn.mu - 5.0) < 0.1
    assert nn.sigma < 10  # posterior should be tighter
    print("OK: bayesian")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test()
    else:
        print("Usage: bayesian.py test")
