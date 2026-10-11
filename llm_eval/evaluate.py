"""Step 3: compare Claude zero-shot with TF-IDF + LR on the same 100 comments,
and build the human-review sheet (50 comments) for the rubric in rubric.md.

Run from the project root:  python llm_eval/evaluate.py
"""
import os

import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

LABELS = ["TikTok", "Instagram Reels", "YouTube Shorts"]
REVIEW_PATH = "llm_eval/human_review.csv"
REVIEW_SIZE = 50

sample = pd.read_csv("llm_eval/sample_100.csv")
preds = pd.read_csv("llm_eval/llm_predictions.csv")
df = sample.merge(preds, on="row_id")
print(f"Comments with an LLM prediction: {len(df)} / {len(sample)}")
print(f"Refused: {(df['llm_pred'] == 'REFUSED').sum()}\n")

rows = {}
for name, col in [("TF-IDF + LR", "lr_pred"), ("Claude zero-shot", "llm_pred")]:
    p, r, f1, _ = precision_recall_fscore_support(df["true_platform"], df[col], labels=LABELS,
                                                  average="weighted", zero_division=0)
    rows[name] = {"Accuracy": accuracy_score(df["true_platform"], df[col]),
                  "Precision": p, "Recall": r, "F1 Score": f1}
print(pd.DataFrame(rows).T.round(3).to_string())

for name, col in [("TF-IDF + LR", "lr_pred"), ("Claude zero-shot", "llm_pred")]:
    print(f"\n=== {name} ===")
    print(classification_report(df["true_platform"], df[col], labels=LABELS, zero_division=0))
    cm = pd.DataFrame(confusion_matrix(df["true_platform"], df[col], labels=LABELS),
                      index=[f"true {l}" for l in LABELS], columns=[f"pred {l}" for l in LABELS])
    print(cm.to_string())

both_right = ((df["lr_pred"] == df["true_platform"]) & (df["llm_pred"] == df["true_platform"])).sum()
only_lr = ((df["lr_pred"] == df["true_platform"]) & (df["llm_pred"] != df["true_platform"])).sum()
only_llm = ((df["lr_pred"] != df["true_platform"]) & (df["llm_pred"] == df["true_platform"])).sum()
print(f"\nBoth right: {both_right} | only LR right: {only_lr} | only Claude right: {only_llm} | "
      f"both wrong: {len(df) - both_right - only_lr - only_llm}")
print(f"LR and Claude agree on {(df['lr_pred'] == df['llm_pred']).mean():.0%} of comments")

# Human-review sheet: created once, never overwritten (your scores live there).
if os.path.exists(REVIEW_PATH):
    print(f"\n{REVIEW_PATH} already exists; not overwriting your scores.")
else:
    review = df.sample(n=min(REVIEW_SIZE, len(df)), random_state=42)
    review = review[["row_id", "comment", "true_platform", "llm_pred", "llm_reason"]].assign(
        cues_real="", reason_supports_answer="", overconfident="", error_type="", note="")
    review.to_csv(REVIEW_PATH, index=False, encoding="utf-8-sig")
    print(f"\nSaved {REVIEW_PATH} — fill in the empty columns following llm_eval/rubric.md")
