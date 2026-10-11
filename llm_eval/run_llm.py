"""Step 2: zero-shot platform classification of the 100-comment sample with Claude.

The model sees only the raw comment (emojis included) and the three platform
names — no examples, no training data. Results are appended row by row to
llm_eval/llm_predictions.csv, so an interrupted run picks up where it stopped.

Needs ANTHROPIC_API_KEY in the environment.
Run from the project root:  python llm_eval/run_llm.py
"""
import csv
import os
from typing import Literal

import anthropic
import pandas as pd
from pydantic import BaseModel

MODEL = "claude-opus-5-5"
SAMPLE_PATH = "llm_eval/sample_100.csv"
OUT_PATH = "llm_eval/llm_predictions.csv"

SYSTEM_PROMPT = """You classify short social media comments by the platform they were posted on.

The three possible platforms are:
- TikTok
- Instagram Reels
- YouTube Shorts

You see one comment at a time, with no video, username, or other context. Many comments \
are short or generic and could plausibly come from any platform; still pick the single most \
likely one. Base your guess on cues in the text itself: vocabulary, slang, emoji use, length, \
tone, and references to platform features.

In "reason", explain in one or two sentences which cues in the comment led to your choice."""


class Prediction(BaseModel):
    platform: Literal["TikTok", "Instagram Reels", "YouTube Shorts"]
    reason: str


def classify(client, comment):
    response = client.messages.parse(
        model=MODEL,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Comment:\n{comment}"}],
        output_format=Prediction,
        output_config={"effort": "low"},
        # If a safety classifier declines, retry on Anthropic's recommended fallback model.
        extra_headers={"anthropic-beta": "server-side-fallback-2026-07-01"},
        extra_body={"fallbacks": "default"},
    )
    if response.stop_reason == "refusal":
        return "REFUSED", "", response.model
    return response.parsed_output.platform, response.parsed_output.reason, response.model


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set. Run: export ANTHROPIC_API_KEY=sk-ant-...")

    sample = pd.read_csv(SAMPLE_PATH)
    done = set()
    if os.path.exists(OUT_PATH):
        done = set(pd.read_csv(OUT_PATH)["row_id"])

    client = anthropic.Anthropic()
    new_file = not os.path.exists(OUT_PATH)
    with open(OUT_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["row_id", "llm_pred", "llm_reason", "served_by"])
        for i, row in enumerate(sample.itertuples(), start=1):
            if row.row_id in done:
                continue
            pred, reason, served_by = classify(client, row.comment)
            writer.writerow([row.row_id, pred, reason, served_by])
            f.flush()
            print(f"[{i}/{len(sample)}] {pred:<16} (true: {row.true_platform})")

    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
