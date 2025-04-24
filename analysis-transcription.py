import matplotlib.pyplot as plt
import seaborn as sns
from jiwer import wer, cer
from sklearn.metrics import confusion_matrix
import numpy as np

# Load data
with open("data/ground_truth.txt", "r", encoding="utf-8") as f:
    ground_truth = [line.strip().lower() for line in f.readlines()]

with open("data/hypothesis.txt", "r", encoding="utf-8") as f:
    hypothesis = [line.strip().lower() for line in f.readlines()]

assert len(ground_truth) == len(hypothesis), "Jumlah baris pada ground truth dan hipotesis harus sama."

# Word Error Rate (WER)
total_wer = wer(ground_truth, hypothesis)
print(f"Word Error Rate (WER): {total_wer:.2%}")

# Character Error Rate (CER)
total_cer = cer(ground_truth, hypothesis)
print(f"Character Error Rate (CER): {total_cer:.2%}")

# Perbandingan kata
gt_words = " ".join(ground_truth).split()
hyp_words = " ".join(hypothesis).split()

same_words = sum(1 for gt, hyp in zip(gt_words, hyp_words) if gt == hyp)
total_words = len(gt_words)
different_words = total_words - same_words

print(f"\nJumlah kata pada Ground Truth     : {len(gt_words)}")
print(f"Jumlah kata pada Hypothesis       : {len(hyp_words)}")
print(f"Jumlah kata yang sama             : {same_words}")
print(f"Jumlah kata yang tidak sama       : {different_words}")