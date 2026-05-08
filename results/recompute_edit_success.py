#!/usr/bin/env python3
"""
Recompute edit_success for all eval questions using each question's own answer
(from the original dataset), instead of blindly using target_new.

Answer source per question type:
  - paraphrase: q["answer"]          (usually == target_new)
  - context:    q["answer"]          (may differ from target_new!)
  - core:       q["answer"]          (usually == target_new)
  - hop:        q["new_answer"]      (always differs from target_new)

For main questions, target_new is still the correct answer.

This script:
  1. Loads the original dataset to build an answer lookup
  2. For each method's labeled result file:
     - Recomputes edit_success for main question (using target_new)
     - Recomputes edit_success for each eval question (using its own answer)
     - If edit_success flips 1->0, also clears issue_type (since it's now a fail)
     - If edit_success flips 0->1, sets issue_type="" (needs manual review)
  3. Prints a summary of changes
  4. Writes updated files (with --apply flag)

Usage:
  python results/recompute_edit_success.py          # preview only
  python results/recompute_edit_success.py --apply   # write changes
"""

import argparse
import json
import os
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = SCRIPT_DIR
ORIG_PATH = os.path.join(SCRIPT_DIR, "..", "data", "merged_eval_dataset.json")

METHODS = ["prompt_v2", "ROME", "FT-M", "GRACE", "WISE"]
METHOD_DISPLAY = {
    "prompt_v2": "IKE", "ROME": "ROME", "FT-M": "FT-M",
    "GRACE": "GRACE", "WISE": "WISE",
}

Q_TYPE_PREFIXES = ["paraphrase", "context", "core", "hop"]


# ── Answer lookup builder ──────────────────────────────────────────

def build_answer_lookup(orig_data):
    """
    Returns a dict: (sample_idx, eval_key) -> expected_answer

    eval_key examples: "paraphrase_1", "context_3", "hop_12"

    Answer source:
      - paraphrases:        q["answer"]
      - context_questions:  q["answer"]
      - core_questions:     q["answer"]
      - hop_questions:      q["new_answer"]
    """
    lookup = {}
    field_map = {
        "paraphrase": ("paraphrases", "answer"),
        "context": ("context_questions", "answer"),
        "core": ("core_questions", "answer"),
        "hop": ("hop_questions", "new_answer"),
    }

    for idx, sample in enumerate(orig_data):
        for prefix, (field_name, answer_key) in field_map.items():
            questions = sample.get(field_name, [])
            for qi, q in enumerate(questions, 1):
                eval_key = f"{prefix}_{qi}"
                answer = q.get(answer_key, "")
                lookup[(idx, eval_key)] = answer

    return lookup


# ── Pattern matching ───────────────────────────────────────────────

def contains_target(response: str, target: str) -> bool:
    """
    Check if response contains the target answer.
    Uses exact substring match first, then falls back to
    token-level match for tokens longer than 3 characters.
    """
    if not target or not response:
        return False
    resp_lower = response.lower()
    tgt_lower = target.lower().strip()

    # Exact substring match
    if tgt_lower in resp_lower:
        return True

    # Token-level fallback: any token >3 chars found in response
    for tok in tgt_lower.split():
        if len(tok) > 3 and tok in resp_lower:
            return True

    return False


# ── Main logic ─────────────────────────────────────────────────────

def recompute_method(method_key, orig_data, answer_lookup, apply=False):
    path = os.path.join(DATA_DIR, f"merged_eval_dataset_{method_key}_post.json")
    with open(path) as f:
        data = json.load(f)

    display = METHOD_DISPLAY[method_key]

    # Counters
    changes = Counter()  # (qtype, "1->0" or "0->1") -> count
    totals = Counter()   # qtype -> count

    for idx, sample in enumerate(data):
        post = sample.get("post", {})
        target_new = sample.get("target_new", "")

        # ── Main question ──
        main_resp = post.get("response", "")
        old_es = post.get("edit_success")
        new_es = 1 if contains_target(main_resp, target_new) else 0
        totals["main"] += 1

        if (old_es in (1, True)) and new_es == 0:
            changes[("main", "1->0")] += 1
            if apply:
                post["edit_success"] = 0
                post["issue_type"] = ""
        elif (old_es not in (1, True)) and new_es == 1:
            changes[("main", "0->1")] += 1
            if apply:
                post["edit_success"] = 1
                if not post.get("issue_type"):
                    post["issue_type"] = ""
        else:
            if apply:
                post["edit_success"] = new_es

        # ── Eval questions ──
        ev = post.get("eval", {})
        for eval_key, v in ev.items():
            if not isinstance(v, dict):
                continue

            prefix = eval_key.split("_")[0]
            if prefix not in Q_TYPE_PREFIXES:
                continue

            expected_answer = answer_lookup.get((idx, eval_key), "")
            if not expected_answer:
                # Fallback to target_new if no answer found
                expected_answer = target_new

            resp = v.get("response", "")
            old_es = v.get("edit_success")
            new_es = 1 if contains_target(resp, expected_answer) else 0
            totals[prefix] += 1

            if (old_es in (1, True)) and new_es == 0:
                changes[(prefix, "1->0")] += 1
                if apply:
                    v["edit_success"] = 0
                    v["issue_type"] = ""
            elif (old_es not in (1, True)) and new_es == 1:
                changes[(prefix, "0->1")] += 1
                if apply:
                    v["edit_success"] = 1
                    if not v.get("issue_type"):
                        v["issue_type"] = ""
            else:
                if apply:
                    v["edit_success"] = new_es

    # Print summary
    print(f"\n{'='*60}")
    print(f"  {display} ({method_key})")
    print(f"{'='*60}")
    print(f"  {'Category':<14} {'Total':>6} {'1->0':>8} {'0->1':>8} {'Net':>8}")
    print(f"  {'-'*46}")

    for qtype in ["main"] + Q_TYPE_PREFIXES:
        n = totals[qtype]
        flip_down = changes.get((qtype, "1->0"), 0)
        flip_up = changes.get((qtype, "0->1"), 0)
        net = flip_up - flip_down
        if n > 0:
            print(f"  {qtype:<14} {n:>6} {flip_down:>8} {flip_up:>8} {net:>+8}")

    total_all = sum(totals.values())
    total_down = sum(v for (_, d), v in changes.items() if d == "1->0")
    total_up = sum(v for (_, d), v in changes.items() if d == "0->1")
    print(f"  {'-'*46}")
    print(f"  {'TOTAL':<14} {total_all:>6} {total_down:>8} {total_up:>8} {total_up-total_down:>+8}")

    if apply:
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  >> Written to {path}")
    else:
        print(f"  >> Preview only. Use --apply to write.")

    return changes, totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="Actually write changes to result files")
    args = parser.parse_args()

    with open(ORIG_PATH) as f:
        orig_data = json.load(f)

    answer_lookup = build_answer_lookup(orig_data)

    print(f"Original dataset: {len(orig_data)} samples")
    print(f"Answer lookup: {len(answer_lookup)} eval answers loaded")

    if args.apply:
        print("\n*** APPLY MODE: changes will be written ***")
    else:
        print("\n*** PREVIEW MODE: no files will be modified ***")

    for mk in METHODS:
        recompute_method(mk, orig_data, answer_lookup, apply=args.apply)

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
