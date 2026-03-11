#!/usr/bin/env python3
"""Naive Bayes text classifier."""
import sys, math, collections, re
class NaiveBayes:
    def __init__(self): self.class_counts=collections.Counter(); self.word_counts=collections.defaultdict(collections.Counter); self.vocab=set()
    def train(self,text,label):
        words=re.findall(r'\w+',text.lower()); self.class_counts[label]+=1
        for w in words: self.word_counts[label][w]+=1; self.vocab.add(w)
    def predict(self,text):
        words=re.findall(r'\w+',text.lower()); total=sum(self.class_counts.values()); best=None; best_s=-float('inf')
        for c in self.class_counts:
            s=math.log(self.class_counts[c]/total)
            n=sum(self.word_counts[c].values()); v=len(self.vocab)
            for w in words: s+=math.log((self.word_counts[c][w]+1)/(n+v))
            if s>best_s: best_s=s; best=c
        return best,best_s
nb=NaiveBayes()
train=[("great movie loved it","pos"),("terrible waste of time","neg"),("amazing performance","pos"),
       ("awful acting boring","neg"),("brilliant screenplay","pos"),("worst film ever","neg")]
for text,label in train: nb.train(text,label)
tests=["great acting","terrible movie","loved the performance","boring waste"]
for t in tests: label,score=nb.predict(t); print(f"  '{t}' → {label} ({score:.2f})")
