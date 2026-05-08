#!/usr/bin/env python3
"""
Comprehensive review of issue_type labels for eval probes with edit_success=1.

Uses the original dataset answers (not just target_new) for more accurate detection.
Applies stricter heuristics for repeat, mixture, and flags potential hallucination.

For each probe with edit_success=1:
  1. Re-check repeat with strict thresholds
  2. Check mixture using both target_new AND target_true/old_answer
  3. Flag potential new_hallucination (long responses with fabricated details)
  4. Generate issue_details for each detected issue

Usage:
  python review_eval_labels.py                    # preview all methods
  python review_eval_labels.py --method ROME      # preview one method
  python review_eval_labels.py --apply            # write changes
  python review_eval_labels.py --verbose          # show all reviewed probes
"""

import argparse
import json
import os
import re
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = SCRIPT_DIR
ORIG_PATH = os.path.join(SCRIPT_DIR, "..", "data", "merged_eval_dataset.json")

METHODS_ALL = ["ROME", "GRACE", "WISE"]


# ── Load original dataset answers ──────────────────────────────────

def build_answer_lookup(orig_data):
    """Build lookup: (sample_idx, eval_key) -> {new_answer, old_answer}"""
    lookup = {}
    field_map = {
        "paraphrase": ("paraphrases", "answer", None),
        "context": ("context_questions", "answer", None),
        "core": ("core_questions", "answer", None),
        "hop": ("hop_questions", "new_answer", "old_answer"),
    }
    for idx, sample in enumerate(orig_data):
        for prefix, (field_name, new_key, old_key) in field_map.items():
            for qi, q in enumerate(sample.get(field_name, []), 1):
                eval_key = f"{prefix}_{qi}"
                entry = {"new_answer": q.get(new_key, "")}
                if old_key:
                    entry["old_answer"] = q.get(old_key, "")
                else:
                    entry["old_answer"] = sample.get("target_true", "")
                lookup[(idx, eval_key)] = entry
    return lookup


# ── Detection functions ────────────────────────────────────────────

def contains_answer(response, answer):
    if not answer or not response:
        return False
    resp = response.lower()
    ans = answer.lower().strip()
    if ans in resp:
        return True
    for tok in ans.split():
        if len(tok) > 3 and tok in resp:
            return True
    return False


def detect_repeat(response, min_length=30):
    """Strict repeat detection."""
    if not response or len(response) < min_length:
        return False, ""

    words = response.lower().split()
    if len(words) < 8:
        return False, ""

    # Consecutive word repetition (4+)
    streak = 1
    for i in range(1, len(words)):
        if words[i] == words[i-1] and len(words[i]) > 1:
            streak += 1
            if streak >= 4:
                return True, f"consecutive repeat: '{words[i]}' x{streak}"
        else:
            streak = 1

    # Phrase-level repetition
    for n in range(2, 7):
        if len(words) < n * 3:
            continue
        ngram_counts = Counter()
        for i in range(len(words) - n + 1):
            ngram = tuple(words[i:i+n])
            ngram_counts[ngram] += 1
        for ngram, count in ngram_counts.items():
            coverage = count * len(ngram) / len(words)
            if count >= 3 and coverage >= 0.25:
                phrase = " ".join(ngram)
                return True, f"phrase repeat: '{phrase}' x{count} ({coverage:.0%} coverage)"

    # Low unique-word ratio
    if len(words) >= 20:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.25:
            return True, f"low unique ratio: {unique_ratio:.2f} ({len(set(words))}/{len(words)} unique)"

    # Character-level repetition
    text = response.lower()
    for window in range(5, 30):
        if len(text) < window * 3:
            continue
        chunk = text[:window]
        if chunk * 3 in text:
            return True, f"char-level repeat: '{chunk[:20]}...' x3+"

    return False, ""


def detect_mixture(response, target_new, target_true):
    """Detect mixture of old and new facts."""
    if not target_new or not target_true:
        return False, ""
    if target_new.lower().strip() == target_true.lower().strip():
        return False, ""

    has_new = contains_answer(response, target_new)
    has_old = contains_answer(response, target_true)

    if has_new and has_old:
        return True, f"contains both new='{target_new}' and old='{target_true}'"
    return False, ""


def detect_hallucination_heuristic(response, expected_answer, question):
    """
    Heuristic hallucination detection.
    Flags responses that are very long or contain suspicious patterns.
    This is a rough filter — results should be manually reviewed.
    """
    if not response or len(response) < 50:
        return False, ""

    resp_lower = response.lower()
    details = []

    # If response is extremely long (>500 chars) for a factual question,
    # it's likely padding with potentially fabricated details
    if len(response) > 500:
        details.append(f"very long response ({len(response)} chars)")

    # Check for fabricated years (4-digit numbers that look like years)
    years = re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', response)
    if years:
        details.append(f"mentions years: {', '.join(years)}")

    # Check for institution/organization names that might be fabricated
    org_patterns = [
        r'University of \w+', r'Institute of \w+', r'College of \w+',
        r'Academy of \w+', r'School of \w+',
    ]
    for pat in org_patterns:
        matches = re.findall(pat, response)
        if matches:
            details.append(f"mentions institutions: {', '.join(matches[:3])}")
            break

    if len(details) >= 2:
        return True, "; ".join(details)

    return False, ""


# ── Main review logic ──────────────────────────────────────────────

def review_method(method_key, orig_data, answer_lookup, apply=False, verbose=False):
    path = os.path.join(DATA_DIR, f"merged_eval_dataset_{method_key}_post.json")
    with open(path) as f:
        data = json.load(f)

    changes = Counter()
    flagged_for_review = []
    total_reviewed = 0

    for idx, sample in enumerate(data):
        post = sample.get("post", {})
        target_new = sample.get("target_new", "")
        target_true = sample.get("target_true", "")
        subject = sample.get("subject", "")
        seed = sample.get("seed_id", idx)

        ev = post.get("eval", {})
        for eval_key, v in ev.items():
            if not isinstance(v, dict):
                continue
            if v.get("edit_success") not in (1, True):
                continue

            prefix = eval_key.split("_")[0]
            if prefix not in ("paraphrase", "core", "context", "hop"):
                continue

            total_reviewed += 1
            response = v.get("response", "")
            old_issue = (v.get("issue_type", "") or "").strip()
            question = v.get("question", "")

            # Get correct answers from original dataset
            ans_info = answer_lookup.get((idx, eval_key), {})
            expected_new = ans_info.get("new_answer", target_new)
            expected_old = ans_info.get("old_answer", target_true)

            # Run detections
            is_repeat, repeat_detail = detect_repeat(response)
            is_mixture, mixture_detail = detect_mixture(response, target_new, target_true)

            # For hop: also check mixture with hop-specific old answer
            if prefix == "hop" and not is_mixture:
                is_mixture, mixture_detail = detect_mixture(response, expected_new, expected_old)

            is_halluc, halluc_detail = detect_hallucination_heuristic(
                response, expected_new, question)

            # Determine new issue_type
            issues = []
            details = []
            if is_repeat:
                issues.append("repeat")
                details.append(repeat_detail)
            if is_mixture:
                issues.append("mixture")
                details.append(mixture_detail)
            if is_halluc and not is_repeat:
                flagged_for_review.append({
                    "seed": seed, "subject": subject, "eval_key": eval_key,
                    "question": question[:80], "response": response[:200],
                    "flag": halluc_detail, "current_issue": old_issue,
                })

            if issues:
                new_issue = "+".join(issues)
                new_details = "; ".join(details)
            else:
                new_issue = old_issue if old_issue else "ok"
                new_details = ""

            # Only change if we detected something new
            if issues and new_issue != old_issue:
                changes[(prefix, old_issue, new_issue)] += 1
                if apply:
                    v["issue_type"] = new_issue
                    v["issue_details"] = new_details
                if verbose:
                    print(f"  seed={seed} {eval_key}: '{old_issue}' -> '{new_issue}'")
                    print(f"    detail: {new_details}")
                    print(f"    resp: {response[:120]}...")
                    print()

    if apply:
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  {method_key}")
    print(f"{'='*60}")
    print(f"  Total edit_success=1 eval probes reviewed: {total_reviewed}")

    if changes:
        print(f"\n  Issue type changes:")
        print(f"  {'Category':<14} {'Old':<20} {'New':<20} {'Count':>6}")
        print(f"  {'-'*62}")
        for (prefix, old, new), count in sorted(changes.items()):
            print(f"  {prefix:<14} {old or '(empty)':<20} {new:<20} {count:>6}")
        total_changed = sum(changes.values())
        print(f"  {'-'*62}")
        print(f"  {'TOTAL':<14} {'':20} {'':20} {total_changed:>6}")
    else:
        print(f"  No issue type changes detected.")

    if flagged_for_review:
        print(f"\n  Flagged for manual review (potential hallucination): {len(flagged_for_review)}")
        # Show top examples
        for item in flagged_for_review[:10]:
            print(f"    seed={item['seed']} {item['eval_key']}: "
                  f"current='{item['current_issue']}' flag='{item['flag']}'")
            print(f"      Q: {item['question']}")
            print(f"      R: {item['response'][:150]}...")
            print()

    if apply:
        print(f"\n  >> Written to {path}")

    return changes, flagged_for_review


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--method", type=str, default=None,
                        help="Review only one method (ROME/GRACE/WISE)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print every change")
    args = parser.parse_args()

    with open(ORIG_PATH) as f:
        orig_data = json.load(f)
    answer_lookup = build_answer_lookup(orig_data)

    methods = [args.method] if args.method else METHODS_ALL

    for mk in methods:
        review_method(mk, orig_data, answer_lookup,
                      apply=args.apply, verbose=args.verbose)

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
