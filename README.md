# CapAI — Self-Expanding, Sandbox-Verified Capability Acquisition Layer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**CapAI** converts natural-language requests into sandbox-verified, permanently cached Python functions and classes, exposed as native [Model Context Protocol (MCP)](https://modelcontextprotocol.io) tools.

> Paper: *CapAI: A Self-Expanding, Sandbox-Verified Capability Acquisition Layer for Large and Small Language Models* — PES University, 2026

---

## Key Results

| Metric | Value |
|---|---|
| Cache speedup | **6.14×** (cold 6,731 ms → warm 1,095 ms) |
| Cached correctness | 83.0% (Wilson CI [81.4%, 84.4%]) |
| Build success rate | 85.9% (Wilson CI [77.0%, 91.9%]) |
| Confidence score Pearson r | **0.79** (p < 0.001) |
| HumanEval Pass@1 | 34.9% (149 tasks, verified) |
| Multi-model Pass@1 | 86.7% (8B, 20B, 70B — all equal) |

---

## Repository Structure

```
capai/
├── benchmark_tasks.yaml       # 190-task benchmark (150 /run + 40 /build)
├── run_benchmark.py           # Benchmark runner
├── analyze_results.py         # Statistical analysis → paper tables
├── generate_figures.py        # Reproduces all paper figures from CSV
├── run_ablation.py            # Ablation study runner (6 conditions)
├── analyze_ablation.py        # Ablation results analyzer
├── run_humaneval.py           # HumanEval evaluation
├── run_humaneval_v2.py        # HumanEval with test-case parsing
├── run_multimodel.py          # Multi-model evaluation
├── run_casestudy.py           # Small-model case study (50 prompts)
├── requirements.txt           # Python dependencies
├── results/                   # All experimental CSV files
│   ├── capai_benchmark_FINAL.csv
│   ├── capai_ALL_RESULTS_MERGED.csv
│   └── capai_ablation_*.csv
├── paper/                     # LaTeX source and figures
│   ├── CapAI_paper_FINAL.tex
│   ├── fig2_latency.pdf
│   ├── fig3_category.pdf
│   └── fig4_calibration.pdf
├── prompts/                   # LLM synthesis and repair prompt templates
└── example_usage/             # Example scripts
```

---

## Reproduce All Results

**One command to reproduce every table and figure:**

```bash
python analyze_results.py results/capai_benchmark_FINAL.csv
python generate_figures.py results/capai_benchmark_FINAL.csv
```

---

## Setup

```bash
# Clone
git clone https://github.com/jsanjayvarma06-oss/CapAI-Capability-Bootstrapping-AI
cd CapAI

# Create virtual environment
python3 -m venv capai_bench
source capai_bench/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Environment variables required:**

```bash
export CAPAI_URL=https://capai-capability-bootstrapping-ai-fu58.onrender.com
export GROQ_API_KEY=your_groq_api_key_here
```

---

## Run the Benchmark

```bash
# Dry run — validate tasks without API calls
python run_benchmark.py --dry-run

# Single task test
python run_benchmark.py --task-id is_prime

# Full /run benchmark (~4 hours)
python run_benchmark.py --workflow run --warm-trials 5 --delay 2

# Full /build benchmark (~1 hour)
python run_benchmark.py --workflow build --build-trials 3 --delay 3

# Analyze results
python analyze_results.py results/capai_benchmark_YYYYMMDD_HHMMSS.csv
```

---

## Run the Ablation Study

```bash
# All 6 conditions
python run_ablation.py

# Single condition
python run_ablation.py --condition A1   # No cache
python run_ablation.py --condition A3   # No repair loop
python run_ablation.py --condition A6   # No persistence

# Analyze
python analyze_ablation.py results/capai_ablation_*.csv
```

**Note:** A2 (no sandbox) requires `CAPAI_SKIP_SANDBOX=true` in Render env vars.
A4 (no heuristics) requires `CAPAI_SKIP_HEURISTICS=true` in Render env vars.

---

## Run HumanEval

```bash
pip install datasets
python run_humaneval_v2.py --convert-only   # Download 164 tasks
python run_humaneval_v2.py --limit 164      # Run evaluation
```

---

## Run Multi-Model Evaluation

```bash
# Change CAPAI_GROQ_MODEL in Render env vars to the target model, then:
python run_multimodel.py --model llama8b --trials 3
python run_multimodel.py --model gptoss20b --trials 3
python run_multimodel.py --analyze
```

---

## Run Case Study

```bash
export GROQ_API_KEY=your_key
python run_casestudy.py
```

---

## Live Server

The CapAI server is deployed at:
```
https://capai-capability-bootstrapping-ai-fu58.onrender.com
```

Check health:
```bash
curl https://capai-capability-bootstrapping-ai-fu58.onrender.com/health
```

Try a capability:
```bash
curl -X POST https://capai-capability-bootstrapping-ai-fu58.onrender.com/run \
  -H "Content-Type: application/json" \
  -d '{"name":"is_prime","description":"Return True if n is prime","args":[17]}'
```

---

## Benchmark Results Summary

The complete 3,337-trial dataset is in `results/capai_ALL_RESULTS_MERGED.csv`.

| Experiment | Tasks | Trials | Key Finding |
|---|---|---|---|
| Main benchmark | 190 | 3,337 | 6.14× cache speedup |
| Ablation A1 | 30 | 120 | No cache → 86.7% correct |
| Ablation A2 | 30 | 120 | No sandbox → 86.7%, 643ms |
| Ablation A3 | 10 | 43 | Repair: +27.4pp success |
| Ablation A4 | 30 | 120 | No heuristics → 90.0% |
| Ablation A6 | 15 | 45 | No persist → 40.0% post-restart |
| HumanEval | 149 | 149 | 34.9% verified Pass@1 |
| Multi-model | 30 | 270 | 86.7% across 8B/20B/70B |
| Case study | 50 | 50 | 87.5% tool invocation rate |

---

## Citation

```bibtex
@inproceedings{varma2026capai,
  title     = {CapAI: A Self-Expanding, Sandbox-Verified Capability
               Acquisition Layer for Large and Small Language Models},
  author    = {Varma, Sanjay and others},
  booktitle = {Proceedings of the IEEE International Conference},
  year      = {2026},
  institution = {PES University, Bengaluru, India}
}
```

---

## License

MIT License — see [LICENSE](LICENSE).