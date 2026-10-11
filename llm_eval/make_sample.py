"""Step 1: rebuild the notebook's exact test split, train the TF-IDF + LR
baseline, and save a fixed 100-comment sample for the LLM evaluation.

Run from the project root:  python llm_eval/make_sample.py
"""
import re

import emoji
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

SAMPLE_SIZE = 100
SEED = 42

# --- Same loading and cleaning as the notebook ---
youtube = pd.read_csv("data/youtube.txt", sep="\t").rename(columns={"textDisplay": "Comment"})
tiktok = pd.read_csv("data/tiktok.txt", sep="\t")
instagram = pd.read_csv("data/instagram.txt", sep="\t").rename(columns={"Comment Text": "Comment"})

data = pd.concat(
    [df[["Comment", "platform"]] for df in (youtube, tiktok, instagram)],
    ignore_index=True,
)
data["platform"] = data["platform"].replace({
    "tik tok": "TikTok",
    "Youtube Shorts": "YouTube Shorts",
    "Instagram Reels": "Instagram Reels",
})
data = data.dropna(subset=["Comment"]).copy()
data["Comment"] = data["Comment"].astype(str).str.strip()
data = data[data["Comment"] != ""].reset_index(drop=True)


def preprocess_text(text):
    text = str(text).lower()
    text = emoji.demojize(text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


data["processed_comment"] = data["Comment"].apply(preprocess_text)

# --- Same 80/10/10 split as the notebook ---
# Split row indices (not text) so we can recover the raw comment for the LLM.
idx_train, idx_temp = train_test_split(data.index, test_size=0.20, random_state=42)
idx_dev, idx_test = train_test_split(idx_temp, test_size=0.50, random_state=42)

train, test = data.loc[idx_train], data.loc[idx_test]

lr_model = Pipeline([
    ("vectorizer", CountVectorizer(ngram_range=(1, 2))),
    ("tfidf", TfidfTransformer()),
    ("classifier", LogisticRegression(max_iter=1000)),
])
lr_model.fit(train["processed_comment"], train["platform"])
test = test.assign(lr_pred=lr_model.predict(test["processed_comment"]))


def report(name, y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    print(f"{name:<40} acc={accuracy_score(y_true, y_pred):.3f}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}")


print(f"Train / dev / test: {len(idx_train)} / {len(idx_dev)} / {len(idx_test)}\n")
print("Full test set:")
report("TF-IDF + LR", test["platform"], test["lr_pred"])
majority = train["platform"].value_counts().idxmax()
report(f"Majority baseline (always '{majority}')", test["platform"], [majority] * len(test))

# --- Fixed sample for the LLM ---
sample = test.sample(n=SAMPLE_SIZE, random_state=SEED)
print(f"\n{SAMPLE_SIZE}-comment sample:")
report("TF-IDF + LR", sample["platform"], sample["lr_pred"])
print("\nPlatform counts in sample:")
print(sample["platform"].value_counts().to_string())

out = sample.reset_index().rename(columns={"index": "row_id", "Comment": "comment", "platform": "true_platform"})
out[["row_id", "comment", "true_platform", "lr_pred"]].to_csv("llm_eval/sample_100.csv", index=False)
print("\nSaved llm_eval/sample_100.csv")
