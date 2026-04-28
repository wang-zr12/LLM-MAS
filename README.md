# LLM-MAS: Multi-Agent Collaboration-Enhanced Role-Playing Dialogue Generation

> 面向角色扮演的基于多智能体协作增强的对话生成方法  
> Southeast University (东南大学) Undergraduate Thesis | Wang Zirui (王梓芮) | 2024.12 – 2025.5

---

## Overview

This project proposes and evaluates a **Multi-Agent System (MAS)** framework for Role-Playing Conversational Agents (RPCAs). Single-agent LLMs often struggle with role consistency and contextual awareness in complex multi-turn dialogues. This work addresses these limitations through task specialization and dynamic inter-agent interaction.

The framework consists of four collaborative modules:

| Module | Role |
|--------|------|
| **Analyzer** | Parses user input and decomposes the role-playing task |
| **Actor (Action Agent)** | Generates dialogue conditioned on role profile, conversation history, and memory |
| **Critic** | Evaluates the Actor's output and provides iterative feedback |
| **Memory & Log** | Maintains FIFO conversation history and cross-turn context |

**Pipeline**: `Input → Analyzer → Actor ⇌ Critic (feedback loop) → Output`

---

## Key Features

- Collaborative multi-agent architecture with iterative critique-and-refine
- Structured system prompt engineering (Role Play, Control Flow, Output Confine, Facilitate Automation, Grounding)
- Ablation study isolating contributions of the Specifier and Critic modules
- Batch inference support (up to 40 samples/iteration) for efficient evaluation
- Comprehensive 12-metric evaluation via CharacterRM reward model

---

## Models / Configurations

| ID | Description | Backbone |
|----|-------------|----------|
| `sa` | Single Agent baseline | DeepSeek-V3 |
| `cot` | Single Agent + Chain-of-Thought | DeepSeek-V3 |
| `sa_r1` | Single Agent with Reasoning | DeepSeek-R1 |
| `mas` | Full Multi-Agent System | DeepSeek-V3 |
| `mas_drop_sp` | MAS without Specifier (ablation) | DeepSeek-V3 |
| `mas_drop_cri` | MAS without Critic (ablation) | DeepSeek-V3 |

---

## Dataset

**CharacterEval** (Tu et al., 2024)
- 1,785 characters, 11,376 conversations, 77 evaluation dimensions
- 13 Personality Back-Testing questions, 12 evaluation metrics
- Scorer: **CharacterRM** (Baichuan-13B fine-tuned reward model)

Data files are located in `data/`:
- `character_profiles.json` / `c_p.json` — character profiles
- `id2metric.jsonl` — evaluation metric mappings

---

## Evaluation Metrics

The CharacterRM scorer evaluates across three groups:

**Character Consistency**: Accuracy (KA), Knowledge Exposure (KE), Knowledge Hallucination (KH), Personality Back-Testing (PB), Personality Utterance (PU)

**Conversational Ability**: Fluency, Coherence, Consistency, (Avg.)

**Role-playing Attractiveness**: Humanlikeness (HL), Communication Skills (CS), Emotional Depth (ED), Empathy (Emp.)

### Main Results (CharacterEval)

| Model | Char. Consistency Avg. | Conv. Ability Avg. | RP Attractiveness Avg. |
|-------|----------------------|-------------------|----------------------|
| Blank | 2.238 | 3.545 | 2.492 |
| Single Agent | 3.079 | 3.938 | 3.337 |
| CoT | 2.990 | 3.980 | 3.268 |
| Sa-r1 | **3.420** | 3.836 | 3.652 |
| **MAS** | 3.328 | **4.081** | 3.636 |
| Mas-drop-sp | 3.267 | 4.016 | 3.665 |
| Mas-drop-cri | 3.327 | 4.061 | 3.632 |

---

## Project Structure

```
LLM-MAS/
├── roleplay_generator/         # Generation pipelines
│   ├── sa_async.py             # Single Agent (async)
│   ├── sa_CoT_async.py         # Single Agent + CoT (async)
│   ├── sa_r1.py                # Single Agent with Reasoning
│   ├── mas0.py                 # MAS without Critic
│   ├── mas1.py                 # Full MAS
│   └── experiments/            # Experiment scripts
├── BaichuanCharRM/             # CharacterRM reward model
├── data/                       # CharacterEval dataset
├── my_results/                 # Raw model outputs
├── middle_results/             # Intermediate processing results
├── LLMInterface.py             # Unified LLM API interface
├── IO.py                       # I/O utilities
├── main.py                     # Entry point
├── eval.py                     # Evaluation script
├── run_characterRM.py          # CharacterRM scoring runner
├── transform.py                # Data format transformations
├── id_map.py                   # ID mapping utilities
├── visualization.py            # Result visualization
└── vilotation.py               # Volatility analysis
```

---

## Environment

| Item | Version |
|------|---------|
| Python | 3.9 |
| LLM API | DeepSeek (V3 / R1) |
| Evaluation Platform | Google Colab |
| GPU | NVIDIA A100 (40GB RAM) |

---

## Usage

### 1. Generate dialogue

```bash
# Single Agent
python roleplay_generator/sa_async.py

# Multi-Agent System (full)
python roleplay_generator/mas1.py

# MAS without Critic (ablation)
python roleplay_generator/mas0.py
```

### 2. Score with CharacterRM

```bash
python run_characterRM.py
```

### 3. Evaluate and visualize

```bash
python eval.py
python visualization.py
```

---

## Citation

If you use this codebase, please cite:

```
王梓芮. 面向角色扮演的基于多智能体协作增强的对话生成方法 [D].
东南大学本科毕业论文, 2025.
```

---

## Acknowledgements & Third-Party Attribution

> **Important**: This project **directly uses content and code from the CharacterEval paper and repository**, including but not limited to:
> - The **CharacterEval benchmark dataset** (character profiles, conversations, evaluation dimensions) — see `data/`
> - The **CharacterRM** reward model (Baichuan-13B based) — see `BaichuanCharRM/`
> - Portions of evaluation/scoring code adapted from the official CharacterEval repository — see `run_characterRM.py` and related scripts
> - The 13-metric evaluation taxonomy (Accuracy, Coherence, Consistency, Humanlikeness, etc.)
>
> All credit for the dataset, reward model, and original evaluation pipeline belongs to the CharacterEval authors. Please cite their work when using this codebase:
>
> ```bibtex
> @article{tu2024charactereval,
>   title={CharacterEval: A Chinese Benchmark for Role-Playing Conversational Agent Evaluation},
>   author={Tu, Quan and Fan, Shilong and Tian, Zihang and Yan, Rui},
>   journal={arXiv preprint arXiv:2401.01275},
>   year={2024}
> }
> ```
>
> Original repository: https://github.com/morecry/CharacterEval

### Other Acknowledgements

- DeepSeek — backbone LLMs (V3 / R1) used in the MAS pipeline
- Related multi-agent / prompting works referenced in this thesis: MetaGPT, ChatDev, CAMEL, AutoAgents, Generative Agents, Reflexion, Chain-of-Thought
