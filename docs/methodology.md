# Methodology: Dataset Construction

Technical appendix to [`project_documentation.md`](project_documentation.md), covering the
measured detail of dataset construction, commit linking, leakage control, and candidate/gold
file generation. Final model results, error analysis, and conclusions live in
`project_documentation.md`; this file only records how the dataset itself was built and
validated.

## Final scope: what was actually used

The final experiments — TF-IDF, Frozen CodeBERT, Fine-tuned CodeBERT, Hybrid, and Component
filtering — run on **Cassandra only**, on a fixed, deterministic, seeded **30% subsample**:
**449 train / 64 validation / 128 test bugs (641 total)**, drawn from Cassandra's full temporal
split of 1,497 / 214 / 428 (2,139 usable bugs). This is the population behind every result in
`project_documentation.md` §6.

HBase and Spark are used only for RQ5 (generalization to other projects): their own 30%-subset
Test bugs are scored with the already-trained Cassandra checkpoint, never trained on —
**HBase30 Test = 136 bugs, Spark30 Test = 332 bugs**.

Everything below (§1-§2) documents how the *full* 3-project dataset — 14,949 issues collected
from Jira, all the way down to the full temporal train/val/test split — was collected and
validated, before the Cassandra scope above was drawn from it. None of the larger numbers below
are the final experiment population; they are the population the final scope was sampled from,
and the process that gives it its validation guarantees (commit-linking accuracy, leakage
control, Component-coverage ceilings).

## 1. Data collection

*Run on 2026-08-08. All numbers below were produced by the commands listed.*

### 1.1 Jira ingestion

`python -m bug2code.data.collect_issues` → `data/raw/issues.parquet`, `reports/tables/01_issue_collection.csv`.

One JQL per project against `https://issues.apache.org/jira/rest/api/2/search`, anonymously
(no token is needed, so no secret ever enters the pipeline), 100 issues per request, every
page cached on disk:

```
project = SPARK AND issuetype IN ("Bug") AND status IN ("Resolved", "Closed")
AND resolution IN ("Fixed") AND created >= "2014-01-01" AND created < "2021-01-01"
ORDER BY created ASC
```

Only fixed bugs are collected: an unfixed issue has no fixing commit and therefore no
ground-truth files. Collection returned **14,949 issues** across the three projects.

| project | issues | with component |
|---|---|---|
| cassandra | 3,087 | 2,032 |
| hbase | 4,259 | 2,593 |
| spark | 7,603 | 7,404 |

Descriptions are stripped of Jira wiki markers (`{code}`, `{noformat}`, `{quote}`) at
ingestion; the markers add tokens and carry no signal.

### 1.2 Why local clones rather than the GitHub API

`python -m bug2code.data.repos` → `data/repos/{project}`, local clones, history only.

Unauthenticated GitHub allows 60 core and 10 search requests per hour. Linking 14,949
issues through the search API is impossible within that budget, and §1.4 shows the linker
must be re-runnable whenever a policy changes. A local `git log --all --name-status` scan
of a whole project takes only seconds. Clones use `--no-checkout` (history only, no working
tree) and are *not* partial: `--filter=blob:none` would make every later snapshot file read
trigger a network fetch.

### 1.3 GitBugs is a cross-check, not a data source

`python -m bug2code.data.gitbugs` → `reports/tables/03_gitbugs_crosscheck.csv`, `04_gitbugs_missing_types.csv`.

Three measured reasons GitBugs cannot serve as the data source here:

1. **No Component field.** Its columns are `Summary, Issue id, Status, Priority, Resolution,
   Created, Resolved, Affects Version/s, Description`. RQ4 needs the Component.
2. **Cost.** It stores internal numeric issue ids, so recovering the missing fields costs one
   API call per issue, against a handful of JQL searches that already return Component.
3. **Wrong period.** GitBugs starts in 2020; our window is 2014–2021. They overlap on 2020
   alone.

On that 2020 intersection, restricted to the same status and resolution filters, the
comparison is:

| project | GitBugs comparable | ours | shared | ours only | GitBugs only |
|---|---|---|---|---|---|
| cassandra | 567 | 320 | 319 | 1 | 248 |
| hbase | 1,219 | 350 | 348 | 2 | 871 |
| spark | 2,210 | 663 | 659 | 4 | 1,551 |

**"ours only" is ~0**: we lose essentially nothing GitBugs has. The large "GitBugs only"
column is not a gap in our collection — it is a difference in what counts as a bug. A random
sample of 160 GitBugs-only issues (40 per collected project) was looked up by issue type in one
JQL call each: **zero were of type `Bug`.** They are Sub-tasks (67), Improvements (63),
Tasks (20), New Features (5), Tests (3), Documentation (1), Umbrella (1). GitBugs labels
every Jira issue a "bug"; our collection is the stricter and cleaner of the two.

### 1.4 Commit linking

`python -m bug2code.data.link_commits` → `data/interim/commit_candidates.parquet`,
`data/interim/fix_commits.parquet`, `reports/tables/02_commit_linking.csv`.

Each history is walked once. Every commit message is scanned for issue keys as whole tokens
(`\bSPARK-\d+\b`, case-insensitive), giving 32,711 key–commit candidate pairs. **Nothing is
discarded during the scan**: all candidates are persisted, and the policy is applied
afterwards by `select_fix_commits`, so a policy change is a re-filter rather than a re-scan,
and every rejection is counted.

Rejection rules, in order, with counts:

| rule | cassandra | hbase | spark |
|---|---|---|---|
| merge commit | 21 | 3 | 13 |
| revert commit | 18 | 357 | 253 |
| off default branch | 5 | 9,778 | 6,521 |
| no source files changed | 646 | 1,563 | 2,009 |
| >30 source files changed | 14 | 10 | 16 |
| duplicate, not earliest | 333 | 223 | 981 |

**Multiple commits per key** is real and common — the risk that makes a naive linker wrong: 795
Spark, 267 Cassandra and 189 HBase issues still have more than one surviving candidate after
filtering. Assuming the first search hit is the fix — as the original feasibility notebook did
— would have silently mislabelled these. The policy is `earliest_on_default_branch`: the
earliest commit reachable from the default branch, since later commits with the same key are
follow-ups and back-ports.

**Off-default-branch rejection** is by far the largest filter, and it is deliberate. Apache
projects back-port the same fix to every supported release branch; each copy has a different
parent tree, so its "code before the fix" is a different codebase from the one our snapshots
come from. Keeping only the default-branch commit keeps every gold set consistent with one
code state. The cost is visible: 1,760 HBase and 1,534 Spark issues have commits *only* off the
default branch and are therefore unlinked.

Resulting link rate — **9,947 / 14,949 = 66.5%**, close to the 64% obtained in the original
28-issue feasibility check. Per-project issue counts and link rate are in the funnel table in
§1.5 below.

### 1.5 Dataset description

`python -m bug2code.data.dataset_report` → `reports/tables/05..10_*.csv`, `reports/figures/fig01..04_*.png`.

Funnel from collected issues to bugs usable for localization. A bug is *usable* when it has a
fixing commit and at least one gold file that existed before the fix; the last column is the
subset that also carries a Component, which is what RQ4 can be run on:

| project | issues | with component | linked | link rate | usable | usable + component |
|---|---|---|---|---|---|---|
| cassandra | 3,087 | 2,032 | 2,141 | 69.4% | 2,139 | 1,333 |
| hbase | 4,259 | 2,593 | 2,277 | 53.5% | 2,275 | 1,324 |
| spark | 7,603 | 7,404 | 5,529 | 72.7% | **5,527** | 5,399 |

Gold files per linked bug are few and dominated by modifications, which is what bug
localization assumes:

| project | mean | median | p90 | single-file | >5 files | gold files new (`A`) |
|---|---|---|---|---|---|---|
| cassandra | 2.70 | 1 | 6 | 53.8% | 11.1% | 2.6% |
| hbase | 2.38 | 1 | 5 | 60.0% | 8.6% | 1.8% |
| spark | 2.14 | 1 | 4 | 58.4% | 6.4% | 1.5% |

Gold files are overwhelmingly modifications (`M`), with only a small share of additions (`A`),
deletions (`D`) or renames (`R*`).

**Newly-added-file policy, measured.** The `drop_from_gold` policy (a file created by
the fixing commit cannot be ranked, because it does not exist in the pre-fix snapshot) costs
almost nothing: by this file-status estimate, only 6 bugs — 2 per project — lose their entire
gold set. §2.1 measures the same policy against the real parent trees, which catches a few more
cases: renames whose *new* path is equally absent from the parent tree, a case the `A`-status
count does not see. Gold is Java-only for HBase and Cassandra; Spark is 86.9% Scala, 8.5%
Python, 4.5% Java.

Text lengths justify the 256-token cap: mean title 7.7–8.5 words, median description 43–83
words, p95 211–379 words. 655 issues have an empty description (476 of them Spark), so those
bugs are ranked from the title alone.

### 1.6 Independent validation of the linker

`python -m bug2code.data.validate_linking` → `reports/tables/11_linker_validation.csv`.

Commit linking is the component of the pipeline most able to fail silently: a wrong commit
produces a plausible-looking gold set that is simply wrong, and no downstream metric would
reveal it. Apache Jira has a rarely-used custom field, *Source Control Link*
(`customfield_12313924`), in which a human pasted the fixing commit URL. It is an independent,
human-curated answer key for exactly the decision `select_fix_commits` makes automatically.

| project | issues with the field | also linked by us | same commit | agreement |
|---|---|---|---|---|
| cassandra | 476 | 303 | 272 | **89.8%** |
| hbase | 0 | 0 | 0 | — |
| spark | 0 | 0 | 0 | — |

The field is far too sparse to link with — hence the git-history scanner — but on the 303
Cassandra bugs where both sources produced a commit, the automatic linker chose the same
commit 89.8% of the time. Shas are compared by prefix, because Jira URLs sometimes carry an
abbreviated sha. This field is used for validation only and never enters training or
evaluation.

The residual ~10% is expected rather than alarming: where a fix spans several commits, the
human may have recorded a follow-up while the `earliest_on_default_branch` policy takes the
first, and some pasted URLs point at back-port branches the policy deliberately rejects.

### 1.7 Component candidate coverage — the ceiling on RQ4, before any model exists

`python -m bug2code.data.dataset_report` → `reports/tables/10_component_candidate_coverage.csv`.

Condition B (the Component-filtering design) ranks only the files that training bugs of the same Component touched.
That restriction has an upper bound no ranker can exceed: if no gold file is inside the
candidate set, the bug is lost whatever the model does. It is measurable now, with no model.
Using the same temporal boundary as the real split, the Component → files map is built from
**training bugs only** and applied to the test bugs. (Boundary indices are rounded, not
truncated: `0.7 + 0.1` is `0.7999…` in binary floating point, so `int()` moves the test
boundary one bug early. The real split must use the same convention.)

| project | test bugs | files in universe | mean candidates | search space kept | candidate coverage | unseen component |
|---|---|---|---|---|---|---|
| cassandra | 228 | 1,044 | 28.3 | 2.7% | 26.8% | 42.1% |
| hbase | 206 | 769 | 26.8 | 3.5% | 39.3% | 12.1% |
| spark | 920 | 1,740 | 495.7 | 28.5% | **75.9%** | 0.8% |

This is exactly the trade-off RQ4 predicts. Spark's components are broad, so the filter keeps
28.5% of the search space and still covers 75.9% of bugs. The other two keep only 2.7–3.5% of
the space but lose 61–73% of the answers with it, because their components are narrow and
their training bugs never touched enough files.

The last column explains much of that loss: a test bug whose Component never appears in
training gets an **empty** candidate set and cannot be ranked at all under Condition B. For
Cassandra this is 42% of test bugs, because it migrated its component taxonomy mid-window
(76% of its 2014–2016 bugs carry a `Legacy/*` component against 9% in 2019–2020, and the two
vocabularies are disjoint). Spark is the project where RQ4 can be tested cleanly.

A negative RQ4 result is an acceptable outcome and will be reported as measured.

## 2. Ground truth and leakage control

### 2.1 Candidate sets from each bug's own pre-fix snapshot

`python -m bug2code.data.build_snapshots` → `data/processed/localization_bugs.parquet`,
`reports/tables/12_candidate_sets.csv`.

Every bug is ranked against the parent of its own fixing commit: the tree the fix was applied
to, which cannot contain the fix (the leakage-prevention rule). The candidate set is *every* in-scope source file
present in that tree — `git ls-tree -r --name-only <parent_sha>` filtered by extension, source
root and exclude pattern — not the files that commit changed. Paths are not stored; only the
count, since the set is reproducible from `parent_sha` at any time.

| project | linked bugs | usable bugs | mean candidates | median | mean gold | bugs losing gold | gold files dropped |
|---|---|---|---|---|---|---|---|
| cassandra | 2,141 | 2,139 | 1,285.8 | 1,275 | 2.62 | 92 | 161 |
| hbase | 2,277 | 2,274 | 1,507.0 | 1,505 | 2.33 | 78 | 106 |
| spark | 5,529 | 5,523 | 1,726.6 | 1,822 | 2.10 | 158 | 209 |

Candidate sets vary widely in size across bugs, so the task is a genuine needle-in-a-haystack
retrieval problem: on average ~2.3 correct files among roughly 1,580 candidates.

**Newly-added-file policy, measured against the real trees.** 476 gold files are dropped
because they do not exist in the parent tree, and a small number of bugs lose their entire gold
set and are dropped with them — the vast majority of gold files (over 98%) survive. The small
gap beyond the file-status estimate in §1.5 is renames, whose *new* path is equally absent from
the parent tree, a case the `A`-status count does not see.

### 2.2 Temporal split

`python -m bug2code.data.split` → `data/processed/localization_dataset.parquet`,
`reports/tables/13_splits.csv`.

Per project, by issue creation date: oldest 70% train, next 10% validation, newest 20% test
(the temporal split design). Boundaries are rounded, not truncated. The result is 6,955 train / 993 validation
/ 1,988 test bugs.

| project | train | val | test | train ends | val ends | test ends |
|---|---|---|---|---|---|---|
| cassandra | 1,497 | 214 | 428 | 2017-01-24 | 2017-12-13 | 2020-12-17 |
| hbase | 1,592 | 227 | 455 | 2017-12-14 | 2018-08-21 | 2020-12-31 |
| spark | 3,866 | 552 | 1,105 | 2017-09-15 | 2018-08-28 | 2020-12-30 |

Candidate sets grow with the split, as the codebase grows over time: Spark goes from 1,406
files per train bug to 2,608 per test bug. Test bugs are therefore harder than train bugs by
construction, which is the honest setting.

This is the full Cassandra split (1,497 / 214 / 428) that the final scope's 30% subsample
(449 / 64 / 128, see top of this file) is drawn from.

### 2.3 Component → files map

`python -m bug2code.data.component_map` → `data/processed/component_files.parquet`,
`reports/tables/14_component_map.csv`.

115 components are mapped (cassandra 25, hbase 63, spark 27) from the 6,955 training
bugs only. A training bug with several Components contributes its gold files to each of them,
mirroring the union rule used at prediction time. Validation coverage — the ceiling on
Condition B, now measured on the real split rather than the §1.7 preview:

| project | val bugs | eligible | no component | empty candidate set | mean candidates | coverage |
|---|---|---|---|---|---|---|
| cassandra | 214 | 136 | 78 | 8 | 104.1 | 65.4% |
| hbase | 227 | 122 | 105 | 7 | 58.2 | 47.5% |
| spark | 552 | 552 | 0 | 1 | 535.1 | **83.7%** |

The candidate-set sizes here are before intersecting with each bug's snapshot, so they are an
upper bound; coverage is unaffected, because a gold file always exists in its own snapshot. The
ordering matches §1.7 — Spark is where RQ4 can be tested cleanly — but the values are higher,
since validation sits closer in time to training than the test set does.
