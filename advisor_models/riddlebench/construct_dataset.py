"""Dataset construction for RiddleBench domain.

Implements the 3-step verifier flow from the proposal:
  1. Student attempts each riddle with no guidance → initial_response stored
  2. Advisor sees riddle + initial_response → generates <diagnosis><advice>
  3. Student revises with advice → final answer → reward

Each training question is emitted twice: once with null_advice=False (advised)
and once with null_advice=True (counterfactual null). Both rows share the same
initial_response — only ONE API call per question. The paired structure enables
true per-question counterfactual advantage estimation when
advantage_batch_normalize=True in the GRPO trainer.

Val split contains only advised rows (null_advice=False) for clean evaluation.

Example (dry run — no API calls, safe for testing):
    python advisor_models/riddlebench/construct_dataset.py --dry_run --limit 5

Example (small real test — 10 items):
    python advisor_models/riddlebench/construct_dataset.py --limit 10 --output_dir /tmp/rb_test

Example (full run):
    python advisor_models/riddlebench/construct_dataset.py --output_dir data/riddlebench
"""

from __future__ import annotations

import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import datasets
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from config import (
    STUDENT_INITIAL_SYSTEM_PROMPT,
    ADVISOR_SYSTEM_PROMPT,
    ADVISOR_INSTRUCTION,
    extract_answer,
    compute_score,
)


def _parse_attempts(raw: str) -> str:
    """Extract and reformat both attempts from the model's single response.

    Looks for 'Attempt 1:' and 'Attempt 2:' markers. Falls back to returning
    the full raw response labelled as Attempt 1 if parsing fails.
    """
    attempt1 = attempt2 = ""

    m1 = re.search(r"Attempt\s*1\s*[:\-]?\s*(.*?)(?=Attempt\s*2|$)", raw, re.DOTALL | re.IGNORECASE)
    m2 = re.search(r"Attempt\s*2\s*[:\-]?\s*(.*?)$", raw, re.DOTALL | re.IGNORECASE)

    if m1:
        attempt1 = m1.group(1).strip()
    if m2:
        attempt2 = m2.group(1).strip()

    if attempt1 and attempt2:
        return f"Attempt 1:\n{attempt1}\n\nAttempt 2:\n{attempt2}"
    # fallback: treat the whole response as a single attempt
    return f"Attempt 1:\n{raw.strip()}"


def get_initial_response(
    problem: str,
    model: str,
    api_base: Optional[str] = None,
) -> str:
    """Get two independent student attempts in one API call.

    Prompts the model to try two different approaches and parses both
    from the single response. Returns a formatted string:
        'Attempt 1:\\n...\\n\\nAttempt 2:\\n...'
    """
    try:
        import litellm
        litellm.drop_params = True
        kwargs: Dict[str, Any] = dict(
            model=model,
            messages=[
                {"role": "system", "content": STUDENT_INITIAL_SYSTEM_PROMPT},
                {"role": "user", "content": problem},
            ],
            temperature=0.7,
        )
        if api_base:
            kwargs["api_base"] = api_base
        response = litellm.completion(**kwargs)
        raw = response.choices[0].message.content or ""
        return _parse_attempts(raw)
    except Exception as e:
        print(f"[get_initial_response] API call failed: {e}")
        return ""


def build_advisor_prompt(
    riddle_type: str,
    problem: str,
    initial_response: str,
) -> List[Dict[str, str]]:
    """Advisor prompt that shows the student's initial wrong attempt."""
    user_content = ADVISOR_INSTRUCTION.format(
        problem=problem,
        riddle_type=riddle_type,
        initial_response=initial_response,
    )
    return [
        {"role": "system", "content": ADVISOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def process_item(
    item: Dict[str, Any],
    model: str,
    api_base: Optional[str],
    dry_run: bool,
    include_null: bool = True,
) -> List[Dict[str, Any]]:
    """Process one RiddleBench item with ONE API call, returning 1 or 2 rows.

    Always returns an advised row (null_advice=False). When include_null=True
    also returns a null row (null_advice=True) sharing the same initial_response.
    """
    problem = item["question"]
    ground_truth = item["answer"]
    riddle_type = item["type"]

    if dry_run:
        initial_response = "[DRY RUN — no API call made]"
    else:
        initial_response = get_initial_response(problem, model, api_base)

    initial_reward = compute_score(extract_answer(initial_response), ground_truth)
    prompt = build_advisor_prompt(riddle_type, problem, initial_response)

    base = {
        "prompt": prompt,
        "env_class": "riddlebench",
        "reward_spec": {"ground_truth": ground_truth},
        "original_question": problem,
        "original_response": initial_response,
        "initial_reward": initial_reward,
        "model": model,
        "riddle_type": riddle_type,
    }

    rows = [{**base, "null_advice": False}]
    if include_null:
        rows.append({**base, "null_advice": True})
    return rows


def process_split(
    items: List[Dict[str, Any]],
    model: str,
    api_base: Optional[str],
    max_workers: int,
    dry_run: bool,
    desc: str,
    include_null: bool = True,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_item, item, model, api_base, dry_run, include_null): item
            for item in items
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            try:
                rows.extend(future.result())
            except Exception as e:
                print(f"[process_split] Row failed: {e}")
    return rows


def _write_parquet(rows: List[Dict[str, Any]], path: Path) -> None:
    """Write rows as parquet without huggingface metadata.

    datasets >=4.x writes a 'huggingface' metadata key with a 'List' feature type
    that datasets 3.x (used in the SkyRL venv) cannot parse. We strip that key so
    the file is readable by both versions, matching the format of existing repo parquets.
    """
    ds = datasets.Dataset.from_pandas(pd.DataFrame(rows))
    ds.to_parquet(str(path))
    # Strip the huggingface metadata key that causes version incompatibility
    table = pq.read_table(str(path))
    clean_meta = {k: v for k, v in (table.schema.metadata or {}).items() if k != b"huggingface"}
    pq.write_table(table.replace_schema_metadata(clean_meta), str(path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct RiddleBench dataset")
    parser.add_argument("--output_dir", type=str, default="data/riddlebench")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--max_workers", type=int, default=2,
                        help="Parallel API workers (keep low to avoid rate limits)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only this many riddles total (for testing)")
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for dataset shuffling")
    parser.add_argument("--dry_run", action="store_true",
                        help="Skip all API calls, use dummy responses (safe for testing)")
    args = parser.parse_args()

    import random
    random.seed(args.seed)
    api_base = os.environ.get("API_BASE", None)

    print("Loading ai4bharat/RiddleBench from HuggingFace...")
    ds = datasets.load_dataset("ai4bharat/RiddleBench")
    all_items = list(ds["train"])
    print(f"Loaded {len(all_items)} riddles")

    random.shuffle(all_items)
    if args.limit:
        all_items = all_items[: args.limit]
        print(f"Limiting to {args.limit} riddles (--limit flag)")

    split_idx = int(len(all_items) * args.train_ratio)
    train_items = all_items[:split_idx]
    val_items = all_items[split_idx:]
    n_total = len(train_items) + len(val_items)

    if args.dry_run:
        print(f"DRY RUN — no API calls will be made ({n_total} items).")
    else:
        print(f"This will make {n_total} API calls to {args.model}.")
    print(f"Train will have {len(train_items)*2} rows (each question → advised + null pair).")

    train_rows = process_split(
        train_items, args.model, api_base,
        args.max_workers, args.dry_run, "train", include_null=True,
    )
    val_rows = process_split(
        val_items, args.model, api_base,
        args.max_workers, args.dry_run, "val", include_null=False,
    )

    null_count = sum(1 for r in train_rows if r["null_advice"])
    # compute accuracy only over advised rows to avoid double-counting
    advised_train = [r for r in train_rows if not r["null_advice"]]
    advised_val = [r for r in val_rows if not r["null_advice"]]
    init_acc_train = sum(r["initial_reward"] for r in advised_train) / max(len(advised_train), 1)
    init_acc_val = sum(r["initial_reward"] for r in advised_val) / max(len(advised_val), 1)
    print(f"Null-advice rows: {null_count}/{len(train_rows)} ({null_count/max(len(train_rows),1):.1%})")
    print(f"Initial accuracy — train: {init_acc_train:.3f}  val: {init_acc_val:.3f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.suffix}" if args.suffix else ""
    train_path = output_dir / f"train{suffix}.parquet"
    val_path = output_dir / f"validation{suffix}.parquet"

    _write_parquet(train_rows, train_path)
    _write_parquet(val_rows, val_path)

    print(f"Wrote {len(train_rows)} train rows → {train_path}")
    print(f"Wrote {len(val_rows)} val rows   → {val_path}")
