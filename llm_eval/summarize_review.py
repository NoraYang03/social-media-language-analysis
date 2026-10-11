"""Step 4: summarize the human review of 50 Claude predictions (rubric.md).

Run from the project root:  python llm_eval/summarize_review.py
"""
import pandas as pd

d = pd.read_csv("llm_eval/human_review.csv", encoding="utf-8-sig")
d["error_type"] = d["error_type"].str.strip().str.lower()
d["correct"] = d["true_platform"] == d["llm_pred"]
wrong = d[~d["correct"]]

print(f"Reviewed: {len(d)} | correct: {d['correct'].sum()} | wrong: {len(wrong)}\n")
for col, label in [("cues_real", "Cues in the reason actually present"),
                   ("reason_supports_answer", "Reason supports the chosen platform"),
                   ("overconfident", "Overconfident on a generic comment")]:
    print(f"{label:<40} {int(d[col].sum()):>2}/{len(d)}   "
          f"(correct: {int(d.loc[d['correct'], col].sum())}/{d['correct'].sum()}, "
          f"wrong: {int(wrong[col].sum())}/{len(wrong)})")

print("\nError types among wrong predictions:")
print(wrong["error_type"].value_counts().to_string())

print("\nWrong predictions by (true -> predicted):")
print(wrong.groupby(["true_platform", "llm_pred"]).size().sort_values(ascending=False).to_string())
