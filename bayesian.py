#!/usr/bin/env python3
"""bayesian - Bayesian inference and probability calculator."""
import argparse, math, json

def bayes(prior, likelihood, evidence):
    return prior * likelihood / evidence

def bayes_table(hypotheses, priors, likelihoods):
    evidence = sum(p * l for p, l in zip(priors, likelihoods))
    posteriors = [bayes(p, l, evidence) for p, l in zip(priors, likelihoods)]
    return posteriors, evidence

def beta_distribution(alpha, beta_param, x):
    from math import gamma as gamma_fn
    B = gamma_fn(alpha) * gamma_fn(beta_param) / gamma_fn(alpha + beta_param)
    return x**(alpha-1) * (1-x)**(beta_param-1) / B

def naive_bayes_classify(features, classes, training_data):
    best_class, best_prob = None, -1
    for cls in classes:
        cls_data = [d for d in training_data if d[-1] == cls]
        prior = len(cls_data) / len(training_data)
        likelihood = prior
        for i, f in enumerate(features):
            matches = sum(1 for d in cls_data if d[i] == f)
            likelihood *= (matches + 1) / (len(cls_data) + 2)  # Laplace smoothing
        if likelihood > best_prob:
            best_prob = likelihood; best_class = cls
    return best_class, best_prob

def main():
    p = argparse.ArgumentParser(description="Bayesian inference")
    sub = p.add_subparsers(dest="cmd")
    b = sub.add_parser("bayes"); b.add_argument("--prior", type=float, required=True)
    b.add_argument("--likelihood", type=float, required=True)
    b.add_argument("--evidence", type=float)
    b.add_argument("--false-positive", type=float)
    t = sub.add_parser("table"); t.add_argument("--hypotheses", nargs="+", required=True)
    t.add_argument("--priors", nargs="+", type=float, required=True)
    t.add_argument("--likelihoods", nargs="+", type=float, required=True)
    d = sub.add_parser("demo")
    args = p.parse_args()
    if args.cmd == "bayes":
        if args.evidence:
            post = bayes(args.prior, args.likelihood, args.evidence)
        elif args.false_positive is not None:
            evidence = args.prior * args.likelihood + (1-args.prior) * args.false_positive
            post = bayes(args.prior, args.likelihood, evidence)
        else:
            print("Need --evidence or --false-positive"); return
        print(f"Posterior: {post:.6f}")
    elif args.cmd == "table":
        posteriors, evidence = bayes_table(args.hypotheses, args.priors, args.likelihoods)
        print(f"Evidence: {evidence:.6f}")
        for h, pr, l, po in zip(args.hypotheses, args.priors, args.likelihoods, posteriors):
            print(f"  {h}: prior={pr:.4f} likelihood={l:.4f} posterior={po:.4f}")
    elif args.cmd == "demo":
        print("Medical test (1% disease prevalence, 99% sensitivity, 5% false positive):")
        evidence = 0.01*0.99 + 0.99*0.05
        post = bayes(0.01, 0.99, evidence)
        print(f"  P(disease|positive) = {post:.4f} ({post*100:.1f}%)")

if __name__ == "__main__":
    main()
