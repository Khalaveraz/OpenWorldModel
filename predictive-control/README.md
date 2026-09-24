# Predictive Control — Weeks 1 and 2

Python 3.12, PyTorch 2.11, Gymnasium 1.3, Pymunk 7.3, NumPy, and Pillow.
The environment, collection, and storage checks run on CPU. Week 2 adds matching
torchvision 0.26 and uses the selected Torch CPU/CUDA build for neural models.

Run the following from the project directory in PowerShell. Setting PYTHONPATH
permits running directly from source without building or installing the project.

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
py -3.12 -B -m unittest discover -s tests -v
py -3.12 -B -m predictive_control diagnose
py -3.12 -B -m predictive_control collect --split pilot --transitions 8000
py -3.12 -B -m predictive_control audit --split pilot
```

Expected: all core/integration tests passing, six successful controlled solvability examples, a
completed collection, and an audit with `integrity: pass`. `coverage_status:
review_required` means inspect the recorded evidence; it is not an automatic
scientific admission. `deficient` blocks sealing. A zero success count alone
does not fail random exploration. The audit prints its full report path and hash.

Before sealing, inspect each task/reset stratum's action histogram, contact
episodes and frequency, object displacement/path length, target/position cells,
returns, successes, layout counts, and sampled physical parameters. The frozen
plan specifies no minimum contact percentage or other numeric coverage threshold.
Do not invent one after seeing results. Lack of contacts or object motion,
missing strata, and degenerate action support are flagged explicitly. These
descriptive position grids are not the later Stage 1 distractor-resistance gate.

If the evidence is adequate, use the **exact hash from the report you reviewed**:

```powershell
$reviewedAudit = Read-Host 'Paste the audit_hash from the pilot report you reviewed'
py -3.12 -B -m predictive_control seal --split pilot --reviewed-audit $reviewedAudit
py -3.12 -B -m predictive_control verify --split pilot
```

Week 1 is complete when the core/integration checks and controlled solvability
checks pass, the visual inspection passes, and a small representative dataset is
reviewed, sealed, and successfully reopened. Full comparison datasets do not
have to be collected before Week 2 feature/model implementation. Do not treat
these checks as evidence that a learned model can predict or control the world.

## Collection and recovery

The only collection policy is random exploration, holding each sampled force for
three decisions with a 10% probability of choosing zero force. Tasks alternate
until one reaches its requested quota; the remaining task then finishes its quota.
Push episodes alternate near/broad starts in equal episode counts. Physics and
action seeds use separate streams within nonoverlapping ranges for each configured
split root. Pilot data are separate from training, validation, and test data.

Counts are nominal minimum targets: the collector retains complete natural
episodes and finishes each task's half plus any pending push-stratum pair. The
JSON result reports requested and actual transitions by task; short terminal
episodes can cause a small overshoot. Do not silently trim final observations or
label a budget boundary as an environment termination. Use the reported actual
counts for all later matched-data comparisons and compute accounting.

Interrupt with Ctrl+C. Rerun the identical collection command to resume. Committed
episodes are verified and retained; unfinished episodes are repeated with the same
seed. `.pending-*` directories from a killed write are uncommitted and never used
as observations. They count toward disk use and are not silently deleted.
Changing configuration, source files, Python/library versions, or target size
requires a different `--data-root`; an existing collection is not overwritten.
Use a separate directory for diagnostic experiments rather than modifying a
sealed dataset. The `--episode-limit N` option stops after N new complete episodes
and returns exit code 3 if collection is not yet complete.

## Storage and provenance

Each committed episode directory contains `observed.npz`, `diagnostics.npz`, and
`record.json`. Numeric arrays are loaded with `allow_pickle=False`. Record files
contain SHA-256 checksums, resolved configuration and its hash, source hashes,
library versions, observation/action IDs, physical units, sensor metadata, action
status/reasons, and collection seeds. Arrays contain T+1 observations and T actions
with actual terminal/truncated observations, timing, validity, ages, and outcomes.

`EpisodeStore.load_episode(id)` exposes observed arrays and provenance. The
privileged state arrays require `include_diagnostics=True` and are physically
separate. Configuration/physical-parameter provenance must not be fed to learned
models; future replay should consume only the declared observation/action arrays.
Imagined records and non-random behavior policies are rejected. The stored action
sequence is checked against its declared random-policy seed.

Episode publication and JSON publication are atomic and refuse replacements.
The store supports **one collector per data root**. Checksums detect later changes;
filesystem permissions do not make files uneditable by their owner. Sealed
manifests are verified against every episode and the audit/source definition.
This is process-interruption recovery, not a guarantee against hardware/disk loss.

Writes stop before the 40 GiB logical campaign budget, reserving 1 GiB headroom,
and check filesystem free space. Filesystem allocation overhead is not measured
by this Week 1 logical-byte counter; full resource supervision belongs to Week 3.
Only one episode is buffered in memory. Collection results report wall-clock
collection/reopen time and logical campaign bytes. No observed data are deleted.

## Later full snapshots

Once pilot admission is settled, these commands collect the configured nominal
200,000/20,000/20,000 transitions. Each split needs its own audit, explicit review,
seal, and verification using the same commands as the pilot.

```powershell
py -3.12 -B -m predictive_control collect --split train
py -3.12 -B -m predictive_control collect --split validation
py -3.12 -B -m predictive_control collect --split test
```

The RGB reactive policy uses disclosed benchmark colors and current self velocity.
The privileged policy checks a few controlled obstacle-free cases; neither is an
obstacle-routing solution or evidence of broad task mastery. Neither policy is
available for collection. Conditional scripted collection remains unimplemented.
Recurrent replay, the campaign trainer, checkpoints, planners, and campaign
evaluation are not implemented in this Week 2 increment.

## Week 2 setup and verification

Use the same Python 3.12 interpreter that passed Week 1. For the existing
Torch 2.11.0+cu130 installation, install its matching torchvision build:

```powershell
Set-Location 'C:\Users\Emiliano Guzman\Zed\Python\World_Model_Ren\predictive-control'
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
py -3.12 -m pip install 'torchvision==0.26.0+cu130' --index-url https://download.pytorch.org/whl/cu130
py -3.12 -m pip check
py -3.12 -B -m unittest discover -s tests -v
py -3.12 -B -m predictive_control verify --split pilot
py -3.12 -B -m predictive_control encoder-check --device cuda
```

The first encoder use downloads official ResNet18 ImageNet weights (about 45 MB).
CPU-only environments must use matching CPU Torch/torchvision builds and
`--device cpu`; do not replace a working CUDA build with a CPU package.
Tests use temporary data and the real frozen encoder. They never edit the pilot.
Original Week 1 configurations remain readable and retain their original hash.
New configurations include the model section and therefore have a new hash.

Week 2 requires a **separate training snapshot** for feature normalization.
Use this disposable diagnostic collection, leaving the final campaign data root
and the pilot untouched. Stop if any command fails:

```powershell
py -3.12 -B -m predictive_control collect --data-root data/week2-smoke/data --split train --transitions 8000
py -3.12 -B -m predictive_control audit --data-root data/week2-smoke/data --split train
```

Read the printed audit report. If its integrity and coverage evidence are
acceptable, paste that report's `audit_hash` into the following prompt. The
implementation does not automatically approve coverage or introduce scripted
collection when coverage is deficient.

```powershell
$reviewedAudit = Read-Host 'Paste the train audit_hash you reviewed'
if ([string]::IsNullOrWhiteSpace($reviewedAudit)) { throw 'An audit hash is required' }
py -3.12 -B -m predictive_control seal --data-root data/week2-smoke/data --split train --reviewed-audit $reviewedAudit
if ($LASTEXITCODE -ne 0) { throw 'Sealing failed' }
py -3.12 -B -m predictive_control verify --data-root data/week2-smoke/data --split train
if ($LASTEXITCODE -ne 0) { throw 'Verification failed' }
py -3.12 -B -m predictive_control cache --data-root data/week2-smoke/data --split train --device cuda
```

The cache command prints its `cache` path and `normalizer` path. Copy the cache
directory path (not a JSON file) when prompted. Run smoke as a **new process**:

```powershell
$featureCache = Read-Host 'Paste the cache directory printed by the cache command'
py -3.12 -B -m predictive_control smoke --data-root data/week2-smoke/data --cache $featureCache --device cuda
```

Cache generation and model optimization have separate process lifetimes: the
training process never constructs ResNet18. `cache --split validation/test/pilot`
creates only clean features and cannot fit a normalizer; those features must use
the normalizer from their associated training snapshot. No mutable global
normalization is fitted on evaluation data. Cache files store FP16 unnormalized
visual coordinates; normalization and likelihoods use FP32. Normalizer fitting
uses every valid clean cached training observation, including final observations,
once. Missing input coordinates are zero placeholders with explicit masks/ages.

## Week 2 admission and limits

`encoder-check` tests visible displacement of either of two identical objects,
contact versus a gap, and target movement. Each feature separation must exceed
ten times FP16 rounding error and 1e-6. This is a necessary distinguishability
check; it does not establish spatial decoding accuracy, tracking, or control.
Failure stops cache admission and does not select a replacement encoder.

`smoke` uses the full **9,265,928-parameter Gaussian RSSM**, two observed training
episodes, both clean/corrupted views, and eight transitions per sequence. A
blackout must occur in each selected episode's initial window. This deliberately
fixed fixture is not the later replay sampler. Tests also exercise the full
32-transition/33-observation shape. Normalization uses the entire sealed training
snapshot, not merely the overfit fixture.

The default is 500 AdamW updates (3e-4 learning rate, 1e-4 weight decay, global
gradient norm cap 10). `--updates` permits 50–500; the diagnostic stops after a
15-minute budget. It uses fresh latent noise during optimization and identical
fixed noise for before/after measurements. These implementation acceptance
criteria are declared in code before the run:

- All losses and gradients stay finite, with nonzero global gradient norms.
- Total objective decreases; visual and proprioceptive MSE each fall by at least 50%.
- Reward MSE, cost BCE, and continuation BCE each decrease.
- Unit tests verify frozen encoder/BN, training-only normalization, immutable
  cache provenance, aligned corruptions/clean targets, missing-input/terminal/
  padding masks, ordinary KL gradients, and isolated observed/imagined updates.

Smoke prints a report and saves it under the diagnostic root's `runs/week2`.
It records the actual parameter count, source/configuration/cache/normalizer
hashes, losses, seed, updates, wall time, and measured CUDA peaks where applicable.
The new numeric overfit/rounding criteria are engineering diagnostics; they do
not revise the frozen plan's scientific gates or establish generalization.

Week 2 passes when the tests, encoder checks, and tiny-overfit report all pass on
the intended environment. The later full pipeline smoke still requires Week 3
checkpoint/recovery and Week 4 closed-loop planning; those features are absent.
No KL balancing/free nats, encoder replacement, scripted collection, architecture
switch, planner, or per-objective gradient audit is activated here.

## Week 2 public model inputs

`RSSM(config, encoder_version=..., normalizer_version=...)` binds the fixed
representation. `observe` accepts an existing `BeliefState`, executed action
`[B,2]` in newtons, elapsed seconds `[B]`, reset flags `[B]`, and a feature mapping
with `sensory [B,4610]`, `validity [B,2]`, `sample_ages [B,2]`, `source_ids`,
`encoder_version`, and `normalizer_version`. All numeric model inputs are FP32
Torch tensors on the model device; masks are boolean. At reset use `dt=0`;
otherwise elapsed time is positive. `ObservationUpdate` returns the belief,
differentiable prior parameters, and per-row posterior-update mask. The belief's
single distribution label summarizes the batch; use that mask for mixed rows.

`forward(SequenceBatch, root=...)` consumes reset-root windows or an explicitly
reconstructed root. It does not implement prefix warm-up. `model_loss` exposes
component sums and counts, excluding padding from numerator and denominator.
Future logical-batch accumulation must combine these sums/counts before reduction.

`imagine(state, action_sequences, noise, goal)` accepts `[B,K,H,2]` forces and
explicit `[B,K,H,64]` noise. It advances 0.1 seconds per decision, permits at most
8,192 total latent steps per call, consumes no RNG, and returns an
`ImaginedTrajectory` with observed-root references. It does not search, choose an
action, update weights, or write replay. `uncheckpointed` is an explicit provisional
model label; checkpoint identity and parameter-version lifecycle belong to Week 3.
