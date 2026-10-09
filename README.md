# Cross-Platform Social Media Language Analysis

Do people write differently on TikTok, Instagram Reels, and YouTube Shorts? This project analyzes **12,376 user comments** across the three platforms, compares emoji usage, and trains text classifiers to predict which platform a comment came from.

## Results

| Model | Accuracy | Precision | Recall | Weighted F1 |
|---|---|---|---|---|
| Multinomial Naive Bayes (baseline) | 0.616 | 0.671 | 0.616 | 0.571 |
| **Logistic Regression** | **0.638** | 0.660 | **0.638** | **0.622** |

Unigram + bigram TF-IDF features, 80/10/10 train/dev/test split (`random_state=42`), weighted averages on the 1,238-comment test set.

**Takeaway:** platform-specific language signals exist, but overlap between platforms is large, so a comment's platform can be predicted only moderately well from its text.

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
- Additional text representations and classification methods

## How to run

    pip install -r requirements.txt
    jupyter notebook social_media_language_analysis.ipynb

Requirements: `pandas`, `scikit-learn`, `emoji`.

## Files

| File | Contents |
|---|---|
| `social_media_language_analysis.ipynb` | Full analysis: data loading, preprocessing, emoji analysis, classification, results |
| `requirements.txt` | Python dependencies |
| `data/README.md` | Notes on the (not included) data files |

## Author

Jing Yang（please hire me)
