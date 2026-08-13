# Bug2Code Project Documentation

## 1. Problem

A developer opens a new bug report and has to guess which source files are relevant before
reading any code. Bug2Code turns that guess into a ranking problem:

```
bug report (title + description)  ->  ranked list of source-code files
```

There is no fixed set of classes: for each bug the model orders every source file that existed
in the project at that moment, and the goal is for the files the eventual fix touched (the
**gold files**) to appear near the top of that ranking, out of thousands of **candidate files**.

Five questions drove the final design:

- **RQ1** — how does Fine-tuned CodeBERT compare with a TF-IDF baseline?
- **RQ2** — does fine-tuning CodeBERT on this task improve ranking over Frozen CodeBERT?
- **RQ3** — does combining Fine-tuned CodeBERT with TF-IDF (Hybrid) improve ranking further?
- **RQ4** — can the Jira Component field, used only to filter the candidate set, reduce the
  search space and improve localization?
- **RQ5** — can a model trained on Cassandra extend to other code projects?

## 2. Dataset

**Sources.** Apache Jira (bug metadata: title, description, Component, dates) and local clones
of the Apache Spark, Cassandra and HBase repositories (fixing commits, historical file
contents). [GitBugs](https://github.com/) was used only as an independent cross-check on
collection completeness, never as a data source.

**Construction.** For each fixed Jira bug, its Jira key is matched against commit messages in
the project's own git history. The earliest matching commit on the default branch, after
rejecting merges, reverts, off-branch back-ports and commits touching either no or more than 30
source files, is taken as the fixing commit. Its **parent commit** is the code state immediately
before the fix — every in-scope source file present there is a candidate file, and the files the
fixing commit changed are the gold files. A file the fix only *creates* cannot be a candidate, so
it is dropped from the gold set; a bug left with no gold files is dropped entirely. This
guarantees the model never sees the fix itself (see `docs/methodology.md` for the full pipeline,
rejection-rule counts and an independent 89.8%-agreement validation of the linker).

**Split.** Temporal, per project, by issue creation date: oldest 70% train, next 10%
validation, newest 20% test. Test is touched exactly once, after every configuration choice is
final.

**Final scope: Cassandra only, 30% subset.** Full-split CodeBERT experimentation was not
feasible on local hardware (Apple Silicon / MPS memory pressure), so the final experiments use a
fixed, deterministic, seeded 30% subsample drawn independently within each of Cassandra's three
splits — Train 449 / Validation 64 / Test 128 bugs, out of the full 1,497 / 214 / 428. This is
not a dev-time convenience later replaced by full-split numbers: it **is** the final experiment
population, as the course allows for exactly this compute reason. HBase and Spark get their own
30%-subset samples, drawn the same way, used only for the generalization-to-other-projects
extension (§3) — never for training.

**Component → files map.** Built from Cassandra 30% Train bugs only, mapping each Jira Component
to the files that training bugs under it had fixed. Used solely to filter the candidate set
before ranking (never as model input), and only for the Component-filtering experiment (§3).

## 3. Final Experimental Setup

Four ranking methods, all evaluated on the same Cassandra30 Test population (128 bugs):

- **TF-IDF + cosine similarity** — identifier-aware tokenization (camelCase/snake_case split),
  fit on Train, no training beyond that.
- **Frozen CodeBERT** — `microsoft/codebert-base` used exactly as pretrained, no fine-tuning.
- **Fine-tuned CodeBERT dual encoder** — two independently weighted towers (bug text, code),
  trained from the pretrained weights with in-batch-negative MNR loss. Files longer than
  CodeBERT's input limit are split into overlapping chunks (256 tokens, stride 192, ≤12 chunks
  per file); a file's score is the max similarity over its chunks. Trained for 3 epochs on the
  Cassandra30 Train subset with epoch-by-epoch resumable checkpoints; the best epoch (epoch 3)
  is selected by Cassandra30 Validation MRR alone, before Test is ever touched.
- **Hybrid (RQ3)** — per-bug min-max normalized TF-IDF and Fine-tuned CodeBERT scores combined as
  `alpha * codebert + (1 - alpha) * tfidf`. Alpha is swept over `[0.0, 0.25, 0.5, 0.75, 1.0]` on
  Validation MRR (selected: alpha=0.5) and fixed before the single Test run.

Two additional experiments reuse the selected Fine-tuned CodeBERT checkpoint without retraining:

- **Component filtering (RQ4)** — the same model ranks the same Test bugs twice: once over the
  full candidate set, once restricted to the Component → files map. Paired comparison, over only
  the bugs that have a known Jira Component.
- **Generalization to other projects (RQ5)** — the Cassandra-trained checkpoint, unchanged,
  evaluated on the HBase30 and Spark30 Test subsets. No new training.

Final error analysis looks at two dimensions on Cassandra30 Test: bug reports containing code
identifiers/stack traces vs. plain prose, and bugs with one gold file vs. several.

## 4. Code Structure

| Path | Purpose |
|---|---|
| `src/bug2code/data/jira.py`, `collect_issues.py` | Jira REST ingestion |
| `src/bug2code/data/repos.py` | Local git clones of the Apache mirrors |
| `src/bug2code/data/linking.py`, `link_commits.py` | Jira key → fixing commit linking |
| `src/bug2code/data/gitbugs.py` | GitBugs cross-check (validation only) |
| `src/bug2code/data/validate_linking.py` | Independent linker-accuracy check |
| `src/bug2code/data/build_snapshots.py` | Per-bug candidate sets from the pre-fix snapshot |
| `src/bug2code/data/split.py` | Temporal train/validation/test split |
| `src/bug2code/data/dev_subset.py` | Deterministic 30% subsampling (final experiment population) |
| `src/bug2code/data/component_map.py` | Component → files map (Train bugs only) |
| `src/bug2code/data/dataset_report.py` | Dataset diagnostics and figures |
| `src/bug2code/localization/tokenize.py`, `tfidf.py` | TF-IDF baseline |
| `src/bug2code/localization/frozen_codebert.py` | Frozen CodeBERT |
| `src/bug2code/localization/train_codebert.py` | Fine-tuned CodeBERT dual-encoder training |
| `src/bug2code/localization/validate_finetuned.py` | Epoch selection on Validation |
| `src/bug2code/localization/save_tfidf_val_scores.py`, `save_finetuned_val_scores.py` | Cache raw Validation candidate scores |
| `src/bug2code/localization/save_tfidf_test_scores.py`, `save_finetuned_test_scores.py` | Cache raw Test candidate scores |
| `src/bug2code/localization/hybrid_experiment.py`, `hybrid_test_experiment.py` | Hybrid alpha selection / final Test run |
| `src/bug2code/localization/component_experiment.py`, `component_test_experiment.py` | Component filtering, Validation / final Test |
| `src/bug2code/localization/cross_project_scores.py` | Generalization to other projects: HBase/Spark evaluation |
| `src/bug2code/localization/candidates.py`, `candidate_scores.py` | Candidate listing and score caching |
| `src/bug2code/localization/metrics.py` | Hit@K, Recall@K, MRR, MAP |
| `src/bug2code/localization/error_analysis.py` | Qualitative Validation error analysis |
| `src/bug2code/localization/lexical_error_analysis.py` | Identifier-rich vs. plain-prose analysis |
| `src/bug2code/localization/gold_count_error_analysis.py` | Single- vs. multi-gold-file analysis |
| `configs/*.yaml` | Per-project and per-scope run configuration |
| `tests/` | Unit tests, one file per `src` module |

## 5. How To Run

```bash
# 1. Dataset construction
python -m bug2code.data.collect_issues
python -m bug2code.data.repos
python -m bug2code.data.link_commits
python -m bug2code.data.build_snapshots
python -m bug2code.data.split
python -m bug2code.data.component_map
python -m bug2code.data.dataset_report
python -m bug2code.data.validate_linking   # independent linker check

# 2. Training (Cassandra30)
python -m bug2code.localization.train_codebert --config configs/cassandra30.yaml

# 3. Validation / model selection
python -m bug2code.localization.validate_finetuned --config configs/cassandra30.yaml
python -m bug2code.localization.save_tfidf_val_scores --config configs/cassandra30.yaml
python -m bug2code.localization.save_finetuned_val_scores --config configs/cassandra30.yaml
python -m bug2code.localization.hybrid_experiment --config configs/cassandra30.yaml
python -m bug2code.localization.component_experiment --config configs/cassandra30.yaml

# 4. Final Cassandra30 Test
python -m bug2code.localization.save_tfidf_test_scores --config configs/cassandra30.yaml
python -m bug2code.localization.save_finetuned_test_scores --config configs/cassandra30.yaml
python -m bug2code.localization.hybrid_test_experiment --config configs/cassandra30.yaml
python -m bug2code.localization.component_test_experiment --config configs/cassandra30.yaml

# 5. Cross-project evaluation
python -m bug2code.localization.cross_project_scores --config configs/hbase30.yaml
python -m bug2code.localization.cross_project_scores --config configs/spark30.yaml

# 6. Error analysis
python -m bug2code.localization.lexical_error_analysis --config configs/cassandra30.yaml
python -m bug2code.localization.gold_count_error_analysis --config configs/cassandra30.yaml

# Tests
pytest
```

Fine-tuning and Frozen CodeBERT inference are GPU-bound and were run on Google Colab; the
Fine-tuned CodeBERT checkpoint and its raw score caches from that run are not committed to this
repository (see §6 provenance notes).

## 6. Final Results

Cassandra30 Test (128 bugs), full result table in `reports/final_results.csv`:

| Method | Hit@1 | Hit@5 | Hit@10 | Hit@20 | MRR | MAP |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF | 0.297 | 0.531 | 0.664 | 0.789 | 0.411 | 0.311 |
| Frozen CodeBERT | 0.008 | 0.016 | 0.023 | 0.063 | 0.019 | 0.015 |
| Fine-tuned CodeBERT (epoch 3) | 0.164 | 0.352 | 0.453 | 0.578 | 0.265 | 0.169 |
| Hybrid (TF-IDF + Fine-tuned, α=0.5) | 0.336 | 0.656 | 0.719 | 0.836 | 0.474 | 0.367 |

**RQ1/RQ2/RQ3** — fine-tuning improves CodeBERT drastically over frozen (MRR 0.019 → 0.265), but
TF-IDF alone still beats Fine-tuned CodeBERT alone; combining the two (Hybrid) gives the best
result overall, ~15% relative MRR gain over TF-IDF.

**Provenance.** TF-IDF, Fine-tuned CodeBERT and Hybrid rows are backed by raw per-bug candidate
scores cached locally and are recomputable from this repository. Frozen CodeBERT and the
cross-project rows below were executed on Colab; their aggregate results are final and reported
as-is, but the underlying model/raw-score artifacts are intentionally not committed (see §7).
`reports/final_results.csv` and `reports/final_analysis_results.csv` both carry a `provenance`
column recording this distinction per row.

**Component filtering (RQ4)** — restricting the candidate set to a bug's Jira Component is a
strong efficiency gain (mean candidates drop from ~1,700 to ~16), but 59 of 115 eligible Test
bugs get an *empty* filtered candidate set (the mapping never saw that Component during
training), so filtered performance is far worse than the full candidate set (Hit@10 0.426 →
0.139). RQ4's answer for Cassandra: hard Component filtering greatly reduces the search space,
but under an evolving codebase — where Components migrate and training coverage is sparse — the
loss of relevant-file coverage can outweigh that advantage.

**Generalization to other projects (RQ5)** — the Cassandra-trained checkpoint transfers to HBase and
Spark with Test performance close to its Cassandra numbers: HBase scores a bit higher on
Hit@10/Hit@20, Spark a bit higher on MRR/MAP (Cassandra MRR 0.265, HBase 0.231, Spark 0.232).
Despite Cassandra and HBase being more alike (both Java, wide-column storage) than Cassandra and
Spark, that similarity does not clearly translate into stronger transfer for HBase.

**Error analysis** — full tables in `reports/final_analysis_results.csv`. Identifier-rich bug
reports (containing stack traces, filenames or qualified names) rank noticeably better under
every method than plain-prose reports. Bugs needing several gold files are harder than
single-gold-file bugs for every method, most sharply for Fine-tuned CodeBERT alone; Hybrid
recovers most of that gap.

**Key takeaways**

- Fine-tuning improves CodeBERT, but TF-IDF remains a strong baseline.
- Hybrid TF-IDF + Fine-tuned CodeBERT performs best on Cassandra Test.
- TF-IDF works especially well when bug reports contain code identifiers.
- Multi-file bugs are harder to fully localize than single-file bugs.
- Component filtering greatly reduces the search space, but may lose relevant files.
- The model transfers reasonably well to both HBase and Spark.

## 7. Limitations / Future Work

**Limitations**

- **Compute constraints** — the main experiments use a deterministic Cassandra30 subsample, not
  the full temporal split, driven by local compute limits; absolute numbers may shift at full
  scale even if the ranking between methods likely would not.
- **Component filtering depends on historical coverage** — the training-only Component → files
  map is too sparse for Cassandra; when Train data never connects a Component to the right
  files, the filter removes relevant candidates instead of narrowing to them.
- **Cross-project comparison is limited** — HBase and Spark may not be different enough from
  each other to draw a strong conclusion about how project similarity affects transfer.

**Future work**

- Evaluate on the full dataset and additional projects.
- Further fine-tuning and broader hyperparameter exploration.
- Use richer bug information, such as stack traces and Jira metadata.
