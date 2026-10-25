# Test data

The datasets are obtained from their original distributors and are not included
in this repository. Follow the respective dataset terms.

## ETH/UCY

Use the Social GAN preprocessed ETH/UCY release:

- [Official download instructions](https://github.com/agrimgupta92/sgan/blob/691231e1adb6a344c7bcea9ebf2534518b226ead/TRAINING.md).
- [Dataset archive linked in those instructions](https://www.dropbox.com/s/8n02xqv3l9q18r1/datasets.zip?dl=1).

Retain the five fold directories under `data/ethucy`. Only each fold's `test`
directory is required. Files contain four columns: frame ID, agent ID, x, y.
Coordinates are in meters. The loader keeps agents present throughout each
20-frame window, using 8 observed and 12 future frames with stride 1.

| Fold | Test files | Windows | Actor-windows |
|---|---|---:|---:|
| ETH | `biwi_eth.txt` | 253 | 364 |
| HOTEL | `biwi_hotel.txt` | 445 | 1,197 |
| UNIV | `students001.txt`, `students003.txt` | 947 | 24,334 |
| ZARA1 | `crowds_zara01.txt` | 705 | 2,356 |
| ZARA2 | `crowds_zara02.txt` | 998 | 5,910 |

## SDD

Use the prebuilt SDD test trajectories from the official
[MART repository](https://github.com/gist-ailab/MART), which credits the
PECNet/Human-Path-Prediction preprocessing:

- [Pinned `sdd_test.pkl`](https://github.com/gist-ailab/MART/blob/fa043648aad41f12f9c15109d53679fcb0113e60/datasets/stanford/sdd_test.pkl).

Download the raw file and place it at `data/sdd/sdd_test.pkl`. It is a list of
1,951 NumPy arrays `[N,20,2]`, totaling 2,829 actor-windows. Convert coordinates
to float32 and split 8 observed / 12 future steps. Coordinates are pixels;
do not apply an external model's additional scale multiplier.

The raw test coordinates match the metadata-bearing version used in the research
experiments exactly after float32 conversion. Evaluation does not require scene
images, metadata restoration, or train/validation files.

Expected SHA256:

```text
011a487c9566f7f47cd9d5ffd507255174a85f5097e19152082731b501ba4de4
```

## NBA

Use the NBA array from the official MART release, whose preprocessing follows LED:

- [Pinned `nba_test.npy`](https://github.com/gist-ailab/MART/blob/fa043648aad41f12f9c15109d53679fcb0113e60/datasets/nba/nba_test.npy).
- [Official MART NBA loader](https://github.com/gist-ailab/MART/blob/fa043648aad41f12f9c15109d53679fcb0113e60/loaders/dataloader_nba.py).

Download the actual Git LFS payload rather than the small text pointer. The
file has 126,561,728 bytes and shape `[47940,30,11,2]`, with float32 coordinates.
Place it at `data/nba/nba_test.npy`.

The primary evaluation uses the first 12,500 windows and all 11 entities
(ten players and the ball), totaling 137,500 entity-windows. Observe 10 steps
and predict 20 at 5 Hz. Preserve the benchmark conversion exactly:

```python
trajectory = raw_window.copy()             # float32, [30,11,2]
trajectory /= 94.0 / 28.0
trajectory = trajectory.transpose(1, 0, 2)  # [11,30,2]
```

The unit is called `nba_m` internally: it uses the benchmark's 28/94 scaling,
not a separate physical feet-to-meters conversion. Evaluation applies it
automatically; do not pre-scale `nba_test.npy`.

Expected SHA256:

```text
a1e190bd197511eaae4bb2d8dfb9016ad2b404e068ae46c928b9cd2fc2a9b205
```

For custom NBA NPZ prediction inputs, coordinates must already be scaled.
Include a scalar string array named `metadata` containing this JSON object:

```json
{"unit":"nba_m","coordinates_are_in_model_units":true,"coordinate_scale_applied":0.2978723404255319}
```

## Source acknowledgments

- Social GAN: Gupta et al., *Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks*, CVPR 2018.
- PECNet: Mangalam et al., *It Is Not the Journey but the Destination: Endpoint Conditioned Trajectory Prediction*, ECCV 2020.
- LED: Mao et al., *Leapfrog Diffusion Model for Stochastic Trajectory Prediction*, CVPR 2023.
- MART: Lee et al., *MART: MultiscAle Relational Transformer Networks for Multi-agent Trajectory Prediction*, ECCV 2024.

The release imports no third-party predictor code and distributes no benchmark
data. These references identify the data releases and their preprocessing lineage.
