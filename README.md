<h1 align="center">Open-Endedness Bench (OEB)</h1>

<p align="center"><b>Measuring epistemic process from agent records</b></p>

<p align="center">
  <a href="https://www.agenticresearch.sh/blog/science-in-broad-daylight"><img alt="Blog" src="https://img.shields.io/badge/Blog-Science%20in%20Broad%20Daylight-b23a2e"></a>
  <a href="https://huggingface.co/datasets/AgentNativeResearchLab/oeb-scored-runs"><img alt="Hugging Face dataset" src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-oeb--scored--runs-ffd21e"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-3776ab">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-2e7d32"></a>
</p>

<p align="center">
  <img src="assets/overview.png" alt="OEB in four steps: raw logs from any harness are converted into one record format; an LLM extracts what the agent said and did with exact quotes and links them into a graph; code counts over the graph to give competence scores and persona traits." width="100%">
</p>

Code for the paper *Open-Endedness Bench: Measuring Epistemic Process from Agent
Records*. The blog post
[Science in Broad Daylight](https://www.agenticresearch.sh/blog/science-in-broad-daylight)
walks through one scored run, with a short video.

OEB scores how an agent does open-ended research, using only the record of what
it did. The scorer never reads a reference answer, the task's outcome score, or
the agent's own summary of its work. A converter may store the benchmark's verdict
for a run in `world.json` for later analysis; no score and no judge prompt reads it.

This repository holds the scoring pipeline (`oeb/`), the converters for the
paper's three benchmarks (`benches/`), and the scripts that rebuild the paper's
tables, figures, and numbers (`paper/`). The records, graphs, judge outputs,
judge-call logs, and scores of every run in the paper are a separate dataset:
[AgentNativeResearchLab/oeb-scored-runs](https://huggingface.co/datasets/AgentNativeResearchLab/oeb-scored-runs).

## What OEB measures

A claim counts as supported only when an action the agent ran returned evidence
for it. OEB turns the record into a graph with two kinds of
nodes: **acts** (what the agent ran) and **stances** (what it stated: beliefs,
conjectures, predictions, questions, reliances, retractions). Edges such as
*expects* or *contradicts* link a stance to the act that tested it. Each node
carries an exact quote from the record, and code checks that the quote is there.

Most measures are one ratio:

    score = opportunities taken / opportunities offered

An opportunity is a concrete situation in the graph (a launched experiment, a
refuted idea, a stated belief). It is taken when the agent did what the measure
asks for. A measure with no opportunity reports nothing. A measure with fewer than
five opportunities reports its counts but stays out of the axis mean. Two measures
are not ratios: E4 turns a count k into 1/(1+k), and T1 is a share of output tokens.

**Competence axes**, scored from 0 to 1. E1-E3 compare what the agent wrote with
what it did; E4 reads only what it did.

| axis | question | score |
|---|---|---|
| E1 Evidence | Are claims backed by what was seen? | mean of *belief grounded*, *assertion within evidence*, *reliance tested*, *unbacked belief held up* |
| E2 Experiment | Do experiments give readable results? | mean of *consequential check rate*, *experiment yield* |
| E3 Revision | Does the agent deal with ideas the evidence contradicted? | *contradicted resolved* |
| E4 No reward hacking | Does it avoid gaming the evaluation? | 1/(1+k), k = reward-hacking instances |

**Persona traits** describe a style and are never ranked. Each sits between two poles:

| trait | poles |
|---|---|
| T1 Budget | doing vs. thinking (share of output tokens spent on actions) |
| T2 Concurrency | one subject vs. several at once |
| T3 Search topology | refine an idea vs. test new ideas |
| T4 Failure style | pivot vs. persist after an adverse result |
| T5 Provenance | transferred vs. novel ideas |
| T6 Epistemic tempo | hypothesis-first vs. run-then-read |

No axis or trait is folded into an overall score.

## What we found

We scored 119 existing runs over 12 tasks from three benchmarks: PostTrainBench
(post-training a small language model), Chip-Bench (hardware design), and the
nanoGPT speedrun (a training-speed record).

- **Most claimed wins are noise.** Checked against the results the benchmarks
  logged, only 16% (speedrun) to 29% (Chip-Bench) of the improvements agents
  claim are real.
- **The best runs keep exploring.** On 9 of 10 tasks, the best run tests more
  new ideas in its second half than the worst run. On Chip-Bench, an idea's
  fourth and later tries are 36% of experiments and bring 10% of the real gain.
- **Research habits follow the model.** For every persona trait, the model that
  ran explains more of the variance across runs than the task (a median of 43%
  against 7%).

<p align="center"><img src="assets/winners.png" alt="Left: new ideas first tested in the second half of the best and the worst run of each task; the best run tests more on 9 of 10 tasks. Right: on Chip-Bench, an idea's fourth and later tries are 36% of experiments and 10% of the real gain." width="100%"></p>

<p align="center"><img src="assets/persona.png" alt="Six persona traits for six models on PostTrainBench; each model keeps a characteristic position across tasks, for example GPT-5.5 and GPT-5.6 spend about a tenth of their output on actions and Kimi K3 about a third." width="90%"></p>

## Pipeline

```
L0  convert    raw logs -> record.jsonl (steps tagged by scope) + world.json   [code, benches/]
L1  extract    acts and stances with checked quotes, three passes merged        [LLM]
L2  connect    closed questions about items and pairs, majority of three votes  [LLM]
L3  assemble   unified_graph.json                                               [code]
    juries     further closed questions on the finished graph                   [LLM]
    measure    E1-E4, T1-T6 -> measure.json                                     [code]
```

The judge only pulls items out of the record and answers closed questions with
fixed answer sets; code does all counting and ordering. No prompt asks a model to
grade a transcript.

## Install and run

```
pip install -e .
python -m oeb.chain <unit_dir> [<unit_dir> ...] --model <judge> [--jobs 4]
```

A unit directory holds `record.jsonl`, `world.json`, and optionally
`domain_note.txt` (background on the task that the judges receive). The chain runs
these stages in order; each can also run alone as
`python -m <module> <unit_dir>` (with `--model <judge>` for the judge stages):

| stage | module | writes |
|---|---|---|
| `extract` (L1) | `oeb.graph.extract` | `acts.json` (acts and stances) |
| `connect` (L2) | `oeb.graph.connect` | `deed_juries.json` |
| `assemble` (L3) | `oeb.graph.assemble` | `unified_graph.json` |
| `uptake` | `oeb.graph.juries.uptake` | `uptake.json` |
| `overreach` | `oeb.graph.juries.overreach` | `overreach.json` |
| `analogy` (T5) | `oeb.graph.analogy` | `analogy.json` |
| `reward_hacking` (E4) | `oeb.graph.juries.reward_hacking` | `aim.json` |
| `coverage` (E4) | `oeb.graph.juries.coverage` | `coverage.json` |
| `measure` | `oeb.measure` | `measure.json` |

`OEB_CHAIN_SKIP=<stage,...>` skips stages by the names in the first column; the
measures that need a skipped stage report nothing. Every judge question and answer
is logged in `<unit_dir>/calls.jsonl`, so rerunning a failed chain costs nothing for
the questions already answered.

### Rescoring a released unit without calling a judge

To rescore a unit from the dataset using only its logged answers:

```
OEB_LLM_REPLAY_ONLY=1 OEB_MAX_WINDOW=5 python -m oeb.chain <unit_dir> --model <the unit's judge>
```

`OEB_LLM_REPLAY_ONLY=1` turns any question missing from the log into an error
instead of a new, paid call. `OEB_MAX_WINDOW=5` is the extraction window the
paper's runs used. This works only for units whose log covers every question the
current code asks; that is not true of every released unit. For the others, drop
`OEB_LLM_REPLAY_ONLY` and the missing questions go to the judge.

### Judges

All model calls go through `oeb/llm.py`. The judge's model name picks the route:

| judge | default route | switch |
|---|---|---|
| Claude | `claude` CLI | `OEB_LLM_TRANSPORT=api`: Anthropic API with `ANTHROPIC_API_KEY` |
| `gpt*` | `codex` CLI | `GPT_TRANSPORT=http`: HTTP with `OPENAI_API_KEY` (`OPENAI_BASE_URL` for another endpoint) |
| GLM | HTTP to the Anthropic-compatible endpoint `GLM_BASE_URL` with `GLM_API_KEY` | `GLM_TRANSPORT=cli`: the `claude` CLI pointed at that endpoint |
| DeepSeek | its CLI | `DEEPSEEK_TRANSPORT=http`: HTTP with `DEEPSEEK_API_KEY` |
| Gemini | Antigravity CLI (`agy`) | |
| Kimi, Grok | their CLIs | |

Keys can also go in a `.env` file at the repository root. `OEB_LLM_CONCURRENCY`
caps parallel calls per process (default 6). When a judge refuses a request, the
same request goes to a fallback judge (`OEB_JUDGE_REFUSAL_FALLBACK`, default
`gpt-5.5`; set it empty to turn this off).

## Converting a benchmark's logs

The scoring code knows nothing about any benchmark; everything benchmark-specific
happens in the converter. A converter must:

1. split the raw log into steps, the same way every time;
2. tag each step as an environment step (acts on the system under study and
   returns evidence), a workspace step (works on files and tools), or the
   submission step;
3. copy action and observation text unchanged;
4. write into `world.json` what the log does not state: the task's rules, the size
   of the evaluation set, where the final result goes. E4 judges the agent's
   actions against these (fields listed in `oeb/record/world.py`).

The paper's converters:

| folder | benchmark | how to convert |
|---|---|---|
| `benches/posttrainbench/` | PostTrainBench (post-training gemma-3-4b for six held-out tasks) | `adapter.py` |
| `benches/chipbench/` | Chip-Bench (hardware design against a hidden grader) | `convert.py`, then `adapter.py` |
| `benches/speedrun/` | nanoGPT speedrun (training-speed record) | `convert.py`, then `adapter.py` |

A converted unit holds `record.jsonl`, `world.json`, `record_manifest.json`
(metadata no score reads), and `usage.json` when the log reports token counts.
`convert.py` turns one raw run into plain steps; `adapter.py` tags them through
`benches/base_adapter.py` and adds the benchmark's own rules. The PostTrainBench
adapter reads the raw harness log (`solve_out.txt`) directly and shares the
tool-kind table in `benches/base_adapter.py`.

Task background for the judges (`domain_note.txt`): Chip-Bench fills in
`benches/chipbench/domain_note.txt` per task; PostTrainBench copies one fixed note
(the paper's HumanEval and GPQA panels ran without it); speedrun units have none,
and the rulebook the agent was given (`benches/speedrun/program.md`) goes into
`world.json` as the task's rules.

## Reproducing the paper's numbers

Download the dataset (`huggingface-cli download AgentNativeResearchLab/oeb-scored-runs
--repo-type dataset --local-dir <dir>`), then point `OEB_DATA` at it:

```
pip install -e .[paper]
export OEB_DATA=/path/to/dataset
python paper/data/make_tables.py
python paper/data/make_consistency_numbers.py
python paper/data/make_findings_numbers.py
python paper/data/make_persona_numbers.py
python paper/figures/gen_fig_rescore.py
python paper/figures/gen_fig_winners.py
python paper/figures/gen_fig_persona.py
python paper/figures/gen_fig_hack_methods.py
```

They write `paper/tables/*.tex`, `paper/data/numbers*.tex`, and the figures. Every
number the paper cites in Sections 5 and 6 comes from one of these macros.

## Layout

```
oeb/
  record/     record format, step loading, world.json
  graph/      extract, connect, assemble; juries/ holds one module per question set
  measure/    E1-E4 and T1-T6 from the graph
  jury.py     shared jury code: quote checking, majority vote
  llm.py      the one logged entry point for model calls
  chain.py    the full chain for one or more units
benches/      converters for the paper's three benchmarks (base_adapter.py is shared)
paper/        scripts for the paper's tables, figures, and numbers
```

## Original names in the output files

The output files, the JSON fields inside them, and the judge prompts keep their
original names, so the released dataset still loads and replays. In the paper's
terms:

| name in the files | paper term |
|---|---|
| `deed` | act |
| `receipt` | verified quote |
| `wager`, `stake`, `bet` | claim |
| `hearing` | a jury's verdict |
| `settlers` | the checks that decided a proposition |
| `chances` (in `measure.json`) | opportunities |
| `deed_juries.json` | the connecting juries (L2) |
| `aim.json` | the reward-hacking jury (E4) |

## Citation

```bibtex
@article{shi2026oeb,
  title  = {Open-Endedness Bench: Measuring Epistemic Process from Agent Records},
  author = {Shi, Chengyang and Ji, Xianglin and Huang, Jintao and Wang, Jicheng and He, Yifeng and Liu, Jiachen},
  year   = {2026}
}
```

## License

MIT (see `LICENSE`).
