"""
code-debugger: a multi-turn, tool-using RL environment.

The model is given a buggy Python function plus a natural-language
description of the bug. It must use the `run_python` tool to execute
code and iterate, then submit a final fixed implementation in a fenced
```python code block. The rollout is graded by actually executing the
model's final code against a hidden unit-test suite for that task.

This environment exercises two things at once:
  1. Agentic multi-turn tool use (ToolEnv: call a tool, observe the
     result, decide whether to call it again).
  2. Code generation + execution as the grading mechanism (the reward
     is not string-matching -- it's "does the code actually work").
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import verifiers as vf
from datasets import Dataset

# --------------------------------------------------------------------------
# 1. Task bank
# --------------------------------------------------------------------------
# Each task is a small, self-contained buggy function plus a hidden test
# script (a plain Python script that raises AssertionError on failure and
# prints "PASS" lines on success). Keeping this inline (rather than a
# separate data file) keeps the environment a single importable module,
# per the Verifiers convention.

TASKS: list[dict[str, Any]] = [
    {
        "id": "off_by_one",
        "prompt": (
            "The function below is supposed to return the sum of all "
            "integers from 1 to n (inclusive), but it has a bug.\n\n"
            "```python\n"
            "def sum_to_n(n):\n"
            "    total = 0\n"
            "    for i in range(1, n):\n"
            "        total += i\n"
            "    return total\n"
            "```\n\n"
            "Find and fix the bug. Use the run_python tool to test your "
            "fix, then reply with ONLY the corrected function in a single "
            "```python code block."
        ),
        "entry_point": "sum_to_n",
        "tests": [
            "assert sum_to_n(1) == 1",
            "assert sum_to_n(5) == 15",
            "assert sum_to_n(10) == 55",
            "assert sum_to_n(0) == 0",
        ],
    },
    {
        "id": "mutable_default",
        "prompt": (
            "This function is supposed to append `item` to a fresh list "
            "each time it's called with no `items` argument, but calls "
            "leak state into each other.\n\n"
            "```python\n"
            "def add_item(item, items=[]):\n"
            "    items.append(item)\n"
            "    return items\n"
            "```\n\n"
            "Fix the bug so each call without `items` starts from an "
            "empty list. Test with run_python, then reply with ONLY the "
            "corrected function in a ```python code block."
        ),
        "entry_point": "add_item",
        "tests": [
            "assert add_item(1) == [1]",
            "assert add_item(2) == [2]",
            "assert add_item(3, [9]) == [9, 3]",
        ],
    },
    {
        "id": "string_reverse_words",
        "prompt": (
            "This function should reverse the ORDER of words in a "
            "sentence (not the letters), but it's broken.\n\n"
            "```python\n"
            "def reverse_words(s):\n"
            "    return s[::-1]\n"
            "```\n\n"
            "Fix it. `reverse_words('the sky is blue')` should return "
            "`'blue is sky the'`. Test with run_python, then reply with "
            "ONLY the corrected function in a ```python code block."
        ),
        "entry_point": "reverse_words",
        "tests": [
            "assert reverse_words('the sky is blue') == 'blue is sky the'",
            "assert reverse_words('hello world') == 'world hello'",
            "assert reverse_words('one') == 'one'",
        ],
    },
    {
        "id": "recursive_factorial",
        "prompt": (
            "This recursive factorial function crashes for n=0 and gives "
            "wrong answers otherwise.\n\n"
            "```python\n"
            "def factorial(n):\n"
            "    if n == 1:\n"
            "        return 0\n"
            "    return n * factorial(n - 1)\n"
            "```\n\n"
            "Fix it. Test with run_python, then reply with ONLY the "
            "corrected function in a ```python code block."
        ),
        "entry_point": "factorial",
        "tests": [
            "assert factorial(0) == 1",
            "assert factorial(1) == 1",
            "assert factorial(5) == 120",
            "assert factorial(6) == 720",
        ],
    },
    {
        "id": "dedupe_preserve_order",
        "prompt": (
            "This function should remove duplicates from a list while "
            "preserving the original order of first occurrence, but it "
            "doesn't preserve order.\n\n"
            "```python\n"
            "def dedupe(items):\n"
            "    return list(set(items))\n"
            "```\n\n"
            "Fix it. Test with run_python, then reply with ONLY the "
            "corrected function in a ```python code block."
        ),
        "entry_point": "dedupe",
        "tests": [
            "assert dedupe([1, 2, 2, 3, 1, 4]) == [1, 2, 3, 4]",
            "assert dedupe([]) == []",
            "assert dedupe(['b', 'a', 'b', 'c']) == ['b', 'a', 'c']",
        ],
    },
    {
        "id": "binary_search",
        "prompt": (
            "This binary search should return the index of `target` in "
            "sorted list `arr`, or -1 if not found. It has an infinite "
            "loop / boundary bug.\n\n"
            "```python\n"
            "def binary_search(arr, target):\n"
            "    lo, hi = 0, len(arr)\n"
            "    while lo < hi:\n"
            "        mid = (lo + hi) // 2\n"
            "        if arr[mid] == target:\n"
            "            return mid\n"
            "        elif arr[mid] < target:\n"
            "            hi = mid\n"
            "        else:\n"
            "            lo = mid\n"
            "    return -1\n"
            "```\n\n"
            "Fix it. Test with run_python, then reply with ONLY the "
            "corrected function in a ```python code block."
        ),
        "entry_point": "binary_search",
        "tests": [
            "assert binary_search([1,3,5,7,9], 7) == 3",
            "assert binary_search([1,3,5,7,9], 1) == 0",
            "assert binary_search([1,3,5,7,9], 9) == 4",
            "assert binary_search([1,3,5,7,9], 4) == -1",
            "assert binary_search([], 4) == -1",
        ],
    },
]


def _build_dataset(tasks: list[dict[str, Any]]) -> Dataset:
    rows = []
    for t in tasks:
        rows.append(
            {
                "question": t["prompt"],
                "answer": t["entry_point"],
                "info": {"entry_point": t["entry_point"], "tests": t["tests"]},
                "task": "code-debugger",
            }
        )
    return Dataset.from_list(rows)


# --------------------------------------------------------------------------
# 2. The tool the model can call
# --------------------------------------------------------------------------

def run_python(code: str) -> str:
    """Execute a snippet of Python and return its stdout/stderr.

    Use this to test whether your fix works before submitting your final
    answer. The snippet runs in a fresh interpreter with a short timeout,
    so include any print() calls you want to see output from.

    Args:
        code: A full, self-contained Python snippet to execute.

    Returns:
        The combined stdout/stderr produced by running the snippet, or an
        error message if it crashed or timed out.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(code)
        path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        out = proc.stdout
        if proc.stderr:
            out += ("\n" if out else "") + "[stderr]\n" + proc.stderr
        return out.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: execution timed out after 10 seconds."
    except Exception as e:  # noqa: BLE001
        return f"Error: {e}"
    finally:
        Path(path).unlink(missing_ok=True)


# --------------------------------------------------------------------------
# 3. Extracting the submitted code from the final assistant message
# --------------------------------------------------------------------------

_CODE_BLOCK_RE = re.compile(r"```python\s*(.*?)```", re.DOTALL)


def _extract_final_code(completion: list[dict[str, Any]] | str) -> str | None:
    if isinstance(completion, str):
        text = completion
    else:
        # completion is the list of turns added by the model; take the
        # last assistant message with actual text content.
        text = ""
        for msg in reversed(completion):
            if msg.get("role") == "assistant" and msg.get("content"):
                text = msg["content"]
                break
    matches = _CODE_BLOCK_RE.findall(text)
    return matches[-1].strip() if matches else None


# --------------------------------------------------------------------------
# 4. Reward functions
# --------------------------------------------------------------------------

def correctness_reward(completion, info: dict, **kwargs) -> float:
    """Fraction of hidden unit tests that pass against the submitted code."""
    code = _extract_final_code(completion)
    if not code:
        return 0.0

    tests: list[str] = info["tests"]
    script = code + "\n\n" + "\n".join(
        f"try:\n    {t}\n    print('PASS')\nexcept Exception as e:\n    print('FAIL:', e)"
        for t in tests
    )
    result = run_python(script)
    passed = result.count("PASS")
    return passed / len(tests) if tests else 0.0


def submitted_code_reward(completion, **kwargs) -> float:
    """Small shaping reward: did the model submit a final code block at all."""
    return 1.0 if _extract_final_code(completion) else 0.0


# --------------------------------------------------------------------------
# 5. Environment factory
# --------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a careful software engineer fixing bugs in Python functions. "
    "You have access to a `run_python` tool that executes code and returns "
    "its output -- use it to test hypotheses before answering. When you are "
    "confident your fix is correct, respond with ONLY the corrected "
    "function in a single ```python code block and stop calling tools."
)


def load_environment(max_turns: int = 6, **kwargs) -> vf.Environment:
    """Entry point required by Verifiers / the Prime Environments Hub.

    Args:
        max_turns: max number of assistant turns (tool calls + final
            answer) allowed per rollout before the episode is cut off.
    """
    dataset = _build_dataset(TASKS)

    rubric = vf.Rubric(
        funcs=[correctness_reward, submitted_code_reward],
        weights=[1.0, 0.0],  # correctness is the real reward; the second is a logged metric
    )

    return vf.ToolEnv(
        dataset=dataset,
        tools=[run_python],
        system_prompt=SYSTEM_PROMPT,
        rubric=rubric,
        max_turns=max_turns,
        **kwargs,
    )
