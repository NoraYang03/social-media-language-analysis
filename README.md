# Cross-Platform Social Media Language Analysis

Do people write differently on TikTok, Instagram Reels, and YouTube Shorts? This project analyzes **12,376 user comments** across the three platforms, compares emoji usage, and trains text classifiers to predict which platform a comment came from.

## Results

| Model | Accuracy | Precision | Recall | Weighted F1 |
|---|---|---|---|---|
| Multinomial Naive Bayes (baseline) | 0.616 | 0.671 | 0.616 | 0.571 |
| **Logistic Regression** | **0.638** | 0.660 | **0.638** | **0.622** |

Unigram + bigram TF-IDF features, 80/10/10 train/dev/test split (`random_state=42`), weighted averages on the 1,238-comment test set.

**Takeaway:** platform-specific language signals exist, but overlap between platforms is large, so a comment's platform can be predicted only moderately well from its text.

## LLM evaluation: zero-shot Claude vs. Logistic Regression

Can a general-purpose LLM, with no training data, tell the platforms apart as well as a model trained on this dataset?

- **Sample:** 100 comments drawn at random from the same test set.
- **LLM:** Claude Opus 5.5 (`claude-opus-5-5`), zero-shot. It sees only the raw comment (emojis included) and the three platform names, and returns structured JSON: the platform plus a one-to-two-sentence reason.
- **Human review:** 50 of the 100 outputs reviewed by hand against a rubric ([`llm_eval/rubric.md`](llm_eval/rubric.md)) that judges the *reason*, not only the answer.

| Model (same 100 comments) | Accuracy | Precision | Recall | Weighted F1 |
|---|---|---|---|---|
| **Logistic Regression** | **0.720** | **0.752** | **0.720** | **0.700** |
| Claude zero-shot | 0.560 | 0.584 | 0.560 | 0.566 |

The two agree on 62% of comments: both right on 48, only Logistic Regression on 24, only Claude on 8, neither on 20. On the smallest class, YouTube Shorts, Claude recalls 5 of 13 comments versus 3 of 13 for Logistic Regression.

| Human review (50 outputs) | Result |
|---|---|
| Cues named in the reason are actually in the comment | 50/50 |
| Reason supports the platform chosen | 37/50 (23/26 correct, 14/24 wrong) |
| Overconfident on a comment with no real platform signal | 15/50 |

Error types among the 24 wrong predictions: platform stereotype (11), no platform signal in the comment (10), misread (1), other (2).

- **No hallucinated evidence:** every reason cited words or emojis that were really in the comment; errors come from how cues were interpreted.
- **Stereotypes drive the errors:** a plain, fully punctuated sentence is read as "YouTube style" and a bare `@mention` as "Instagram style", but both habits appear on every platform here. Claude predicted YouTube Shorts 18 times and was right 5 times.
- **Many comments carry no signal:** a lone emoji or "me fr" fits any platform.

**Takeaway:** on this task a small model trained on in-domain data beats a large zero-shot model by 16 accuracy points. The LLM's reasons are faithful to the text but lean on general platform stereotypes that do not hold for this dataset, a failure that shows up only when the reasoning is reviewed, not just the accuracy.

## Data

- Comments collected from comparable videos on one topic (carsickness) on each platform, to reduce topic differences.
- 12,391 comments collected; 12,376 after removing empty or missing comments.

| Platform | Comments |
|---|---|
| TikTok | 5,722 |
| Instagram Reels | 4,932 |
| YouTube Shorts | 1,722 |

The raw data is **not included** because comments may contain user-identifying information. The notebook expects three tab-separated files in `data/` (`youtube.txt`, `tiktok.txt`, `instagram.txt`), each with a comment column and a `platform` column.

## Method

1. **Cleaning:** standardize column names across platform exports, drop empty comments, normalize platform labels.
2. **Preprocessing:** lowercase, convert emojis to text descriptions (`emoji.demojize`) so emoji signal is kept, remove non-word characters, normalize whitespace.
3. **Emoji analysis:** extract emojis from the original comments and compare the most frequent emojis per platform.
4. **Classification:** scikit-learn pipelines (`CountVectorizer(ngram_range=(1, 2))` → `TfidfTransformer` → classifier), comparing Multinomial Naive Bayes and Logistic Regression on the same features.

## Limitations

- **Class imbalance:** comment counts differ substantially across platforms, particularly for YouTube Shorts.
- **Dataset scope:** comments come from a limited set of comparable content, so they may not represent platform-wide language.
- **Informal language:** emojis, slang, and short expressions are highly variable.
- **Feature representation:** unigram/bigram TF-IDF captures lexical patterns but not context or meaning.

Results show detectable differences within this dataset; this is not a general-purpose system for identifying a comment's platform.

## Future work

- A larger, more balanced dataset across topics and creators
- Feature analysis: which words, n-grams, and emojis drive each prediction
- Additional text representations and classification methods, including a fine-tuned neural classifier in PyTorch
- Few-shot prompting, to test whether a handful of in-domain examples closes the gap between the LLM and the supervised model

## How to run

    pip install -r requirements.txt
    jupyter notebook social_media_language_analysis.ipynb

Requirements: `pandas`, `scikit-learn`, `emoji`, `anthropic`.

LLM evaluation (needs your own `ANTHROPIC_API_KEY`):

    python llm_eval/make_sample.py       # rebuild the split, baseline, and 100-comment sample
    python llm_eval/run_llm.py           # Claude zero-shot predictions
    python llm_eval/evaluate.py          # metrics + human-review sheet
    python llm_eval/summarize_review.py  # rubric summary

## Files

| File | Contents |
|---|---|
| `social_media_language_analysis.ipynb` | Full analysis: data loading, preprocessing, emoji analysis, classification, results |
| `llm_eval/` | LLM evaluation scripts and the human-review rubric |
| `requirements.txt` | Python dependencies |
| `data/README.md` | Notes on the (not included) data files |

## Author

Jing Yang（please hire me)
