# CandidateSocial

Candidate-conditioned hierarchical refinement for multi-agent trajectory prediction.

This release contains the complete prediction model, a portable evaluation interface,
and seven pretrained Full-model checkpoints for ETH/UCY, SDD, and NBA.
Each checkpoint includes the initial generator, refinement module, residual basis,
and coordinate scale. No additional generator checkpoint is needed.

## Model

The model encodes observed motion and generates 20 complete candidate futures.
For each target candidate, it summarizes relevant futures within each neighbor,
then aggregates messages across neighbors. A gated residual update refines the
trajectory coefficients over two shared rounds.

| Benchmark | Observed / predicted steps | Candidates | Maximum neighbors | Coordinates |
|---|---:|---:|---:|---|
| ETH/UCY | 8 / 12 | 20 | 8 | meters |
| SDD | 8 / 12 | 20 | 8 | pixels |
| NBA | 10 / 20 | 20 | 10 | benchmark scale: source coordinates divided by 94/28 |

## Installation

Create a Python 3.10 environment, install PyTorch for your hardware using its
[official installation instructions](https://pytorch.org/get-started/locally/), then:

```bash
python -m pip install -r requirements.txt
python -m candidate_social --help
```

The preparation environment uses Python 3.10.19, PyTorch 2.9.1 (CUDA 12.8), and
NumPy 2.2.6. CPU inference is supported with `--device cpu`.
The conda environment file is an alternative starting point.

## Test data

Download the benchmark test trajectories using [the data guide](docs/DATA.md).
Only test files are needed for this release. Place them as follows:

```text
data/
  ethucy/
    eth/test/biwi_eth.txt
    hotel/test/biwi_hotel.txt
    univ/test/students001.txt
    univ/test/students003.txt
    zara1/test/crowds_zara01.txt
    zara2/test/crowds_zara02.txt
  sdd/sdd_test.pkl
  nba/nba_test.npy
```

## Evaluation

Run these commands from the repository root. The checkpoint supplies its model
configuration; `--data-root` points to your own data directory.

```bash
python -m candidate_social evaluate --checkpoint checkpoints/eth.pt --data-root data/ethucy --output outputs/eth.json
python -m candidate_social evaluate --checkpoint checkpoints/hotel.pt --data-root data/ethucy --output outputs/hotel.json
python -m candidate_social evaluate --checkpoint checkpoints/univ.pt --data-root data/ethucy --output outputs/univ.json
python -m candidate_social evaluate --checkpoint checkpoints/zara1.pt --data-root data/ethucy --output outputs/zara1.json
python -m candidate_social evaluate --checkpoint checkpoints/zara2.pt --data-root data/ethucy --output outputs/zara2.json
python -m candidate_social evaluate --checkpoint checkpoints/sdd.pt --data-root data/sdd --output outputs/sdd.json
python -m candidate_social evaluate --checkpoint checkpoints/nba.pt --data-root data/nba --output outputs/nba.json
```

The default device is CUDA when available, otherwise CPU. Use `--batch-size` to
adjust memory use. For a quick installation check, add `--max-windows 8` and use a
different output filename. A prefix check does not produce the full test score.
The evaluator refuses to overwrite an existing result file.

### Released checkpoints

Within each setting, the seed with the lowest validation `minADE + 0.5 * minFDE`
was selected from seeds 42, 43, and 44. Its epoch had already been selected using
the same validation criterion. The following are the recorded test results of
these individual checkpoints; the manuscript reports averages over three seeds.

| Setting | Seed | Selected epoch | minADE20 | minFDE20 |
|---|---:|---:|---:|---:|
| ETH | 43 | 87 | 0.397959 | 0.594605 |
| HOTEL | 43 | 22 | 0.133836 | 0.189710 |
| UNIV | 44 | 50 | 0.218974 | 0.394644 |
| ZARA1 | 44 | 66 | 0.169631 | 0.305257 |
| ZARA2 | 42 | 30 | 0.120047 | 0.216739 |
| **ETH/UCY macro average** | mixed, as above | | **0.208089** | **0.340191** |
| SDD | 44 | 64 | 7.440269 | 11.955278 |
| NBA | 43 | 97 | 0.792626 | 1.041991 |

Within a fold, errors are averaged over valid actor-windows. The ETH/UCY aggregate
is the unweighted mean of the five fold scores. NBA evaluates the official first
12,500 test windows, including all ten players and the ball. minADE and minFDE
each take their minimum over the 20 candidates independently.

See [the checkpoint manifest](checkpoints/manifest.json) for validation selection
values, precise metrics, and SHA256 checksums.

## Prediction on trajectory inputs

Save an NPZ containing `obs` (float32, `[B,N,H,2]`) and `agent_mask` (boolean,
`[B,N]`). Here `B` is the number of windows, `N` the padded agent count, and `H`
the observation length. Coordinates must already use the checkpoint's units.

```bash
python -m candidate_social predict --checkpoint checkpoints/eth.pt --input example.npz --output outputs/prediction.npz
```

The output contains `pred` and `initial_pred` with shape `[B,N,20,F,2]`, plus
`agent_mask` and JSON metadata. `F` is the forecast length. NBA prediction inputs
must additionally declare their coordinate scaling; see [the data guide](docs/DATA.md).

## Contents

- `candidate_social/`: model, geometry, test loaders, metrics, and inference CLI.
- `checkpoints/`: seven self-contained Full-model checkpoints.
- `configs/`: readable copies of the configurations embedded in the checkpoints.
- `docs/DATA.md`: dataset sources, exact preprocessing, and expected test sizes.
- `tests/`: lightweight interface and metric checks.

This initial release provides prediction and evaluation. Training runners,
ablation checkpoints, and external-generator plug-in experiments are outside
this release.
