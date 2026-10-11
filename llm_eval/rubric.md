# Human review rubric

Score each of the 50 rows in `human_review.csv` (open it in Excel or Numbers).
Judge the **reason**, not just whether the platform is right — a correct answer
can come from a bad reason, and a wrong answer can come from a sensible one.

| Column | Values | Question to ask |
|---|---|---|
| `cues_real` | 1 / 0 | Are the cues the reason mentions actually in the comment? (0 = it describes words, emojis, or features that are not there) |
| `reason_supports_answer` | 1 / 0 | If those cues are real, do they point to the platform it chose? |
| `overconfident` | 1 / 0 | Is the comment generic (could be any platform) but the reason sounds certain anyway? |
| `error_type` | see below | Only for wrong predictions; leave blank when correct |
| `note` | free text | Anything interesting — good candidates for the README's error examples |

`error_type` values:
- `generic` — the comment has no real platform signal (e.g. "lol", a single emoji)
- `stereotype` — relied on a platform stereotype that did not hold (e.g. "young slang → TikTok")
- `misread` — misunderstood the comment
- `other` — anything else (describe it in `note`)
