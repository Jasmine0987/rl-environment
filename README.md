# RL Environments

Reinforcement learning environments built with [Verifiers](https://github.com/PrimeIntellect-ai/verifiers), the environment framework behind Prime Intellect's [Environments Hub](https://app.primeintellect.ai/dashboard/environments).

Each environment here is a self-contained, installable Python package that pairs a task with a **verifiable reward function** — rollouts are graded by actually running code or checking real outcomes, not string similarity.

## Environments

### [`code-debugger`](environments/code_debugger)

A multi-turn, tool-using environment where a model is given a buggy Python function and must fix it. The model has access to a `run_python` tool it can call to test hypotheses, then submits a final answer that's graded by running it against a hidden unit-test suite.

**What it exercises:**
- Agentic multi-turn tool use (call a tool → read the result → decide to call again or answer)
- Reward grounded in code execution, not pattern matching

**Verified working end-to-end** — evaluated locally against `qwen2.5:1.5b` (CPU-only, no GPU) via Ollama:

```
--- All ---
Rewards:
reward: avg - 0.894, std - 0.259
correctness_reward: avg - 0.894, std - 0.259
submitted_code_reward: avg - 1.000, std - 0.000
```

The model submitted valid code on 100% of rollouts, with correctness varying by bug difficulty — exactly the kind of signal a reward function should produce.

See [environments/code_debugger/README.md](environments/code_debugger/README.md) for task details, reward breakdown, and design notes.

## Setup

```bash
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# install the Prime CLI (for pushing to the Environments Hub)
uv tool install prime
prime config set-api-key <your-api-key>
```

## Running an environment locally

```bash
cd environments/code_debugger
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -e .
pip install verifiers

# using a local model via Ollama (no API cost, no GPU required)
vf-eval code-debugger -m qwen2.5:1.5b -b http://localhost:11434/v1 -k OPENAI_API_KEY
```

## Publishing to the Environments Hub

```bash
cd environments/code_debugger
prime env push
```

## Repo structure

```
environments/
  code_debugger/
    code_debugger.py    # environment logic: dataset, tool, reward functions
    pyproject.toml       # package metadata + dependencies
    README.md            # environment-specific docs
```

Each environment is its own installable package, following the Environments Hub convention — this keeps environments independently versioned and dependency-isolated rather than living in one shared monorepo namespace.