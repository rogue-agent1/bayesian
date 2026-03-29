from bayesian import bayes_theorem, bayes_update, NaiveBayesClassifier, beta_posterior, beta_mean, credible_interval
p = bayes_update(0.01, 0.95, 0.05)
assert 0.1 < p < 0.2  # ~16%
p2 = bayes_update(p, 0.95, 0.05)  # second positive test
assert p2 > p
nbc = NaiveBayesClassifier()
X = [["good","great"],["bad","awful"],["good","nice"],["bad","terrible"]]
y = ["pos","neg","pos","neg"]
nbc.fit(X, y)
assert nbc.predict(["good","nice"]) == "pos"
assert nbc.predict(["bad","awful"]) == "neg"
a, b = beta_posterior(1, 1, 7, 3)
assert a == 8 and b == 4
assert abs(beta_mean(8, 4) - 0.667) < 0.01
lo, hi = credible_interval(8, 4)
assert lo < 0.667 < hi
print("bayesian tests passed")
