# code-debugger

A multi-turn, tool-using RL environment built with [Verifiers](https://github.com/PrimeIntellect-ai/verifiers).

## Task

The model is given a Python function that contains a bug and a description
of what it's supposed to do. It has access to a `run_python` tool it can
call any number of times (up to `max_turns`) to test hypotheses before
submitting a final answer as a single fenced ` ```python ` code block.

This exercises two things at once:

- **Agentic multi-turn tool use** — the model decides when to call the
  tool, reads the result, and decides whether to call it again or answer.
- **Verifiable code execution as the reward signal** — grading is not
  string/edit-distance matching. The model's final submission is actually
  executed against a hidden unit-test suite for that task, and the reward
  is the fraction of tests that pass.

Six hand-written bug categories are included out of the box (off-by-one
loop bounds, mutable default arguments, wrong string operation, missing
base case in recursion, order-losing deduplication, and a binary-search
boundary bug). Add more by appending to the `TASKS` list in
`code_debugger.py`.

## Reward

| Function | Weight | What it measures |
|---|---|---|
| `correctness_reward` | 1.0 | Fraction of hidden tests the submitted fix passes (0.0–1.0) |
| `submitted_code_reward` | 0.0 (metric only) | Whether the model submitted a code block at all |

## Usage

```bash
uv pip install -e .
uv run vf-eval code-debugger
```

Or from Python:

```python
import verifiers as vf

env = vf.load_environment("code-debugger", max_turns=6)
```

## Caveats / next steps

- `run_python` executes code via a plain subprocess with a 10s timeout —
  this is fine for local development and this repo's own toy tasks, but
  it is **not a hardened sandbox**. For untrusted model-generated code at
  scale, swap `run_python` for Prime Intellect's `SandboxEnv` /
  `PythonEnv`, which run code in an isolated container.
- The task bank is intentionally small (6 tasks) as a template — a real
  submission to the Environments Hub should have a larger, more diverse
  dataset (or load one from Hugging Face via `vf.load_example_dataset`
  or `datasets.load_dataset`).
