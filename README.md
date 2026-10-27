# Certified Neural Networks: from Verification to Synthesis — HorizontalCAS experiments

This repository contains the code and data for the HorizontalCAS experiments in:

> Matteo Zavatteri, Davide Bresolin, Nicolò Navarin.
> **Certified Neural Networks: from Verification to Synthesis.**
> *Journal of Artificial Intelligence Research*, 87, 2026.
> DOI: [10.1613/jair.1.22825](https://doi.org/10.1613/jair.1.22825)

```bibtex
@article{ZavatteriBresolinNavarin2026,
  author  = {Zavatteri, Matteo and Bresolin, Davide and Navarin, Nicol{\`o}},
  title   = {Certified Neural Networks: from Verification to Synthesis},
  journal = {Journal of Artificial Intelligence Research},
  volume  = {87},
  year    = {2026},
  doi     = {10.1613/jair.1.22825},
  url     = {https://jair.org/index.php/jair/article/view/22825}
}
```

The paper was published in the JAIR special track [*Integration of Logical Constraints in Deep Learning*](https://www.jair.org/index.php/jair/SpecialTrack-LogicDL).

This repository includes:

- the training data (`HCAS_polar_networks/`),
- the synthesis tool (`HCASCertifier.py` and `properties/`),
- the raw results of every run reported in the paper (`experiments/`),
- the scripts that turn those results into the paper's tables and figures, and the figures themselves (`img/`).

## What the code does

`HCASCertifier.py` trains a feed-forward ReLU network that imitates a HorizontalCAS advisory table. It then uses a counterexample-guided inductive synthesis (CEGIS) loop to make the network satisfy a given safety property. Each CEGIS iteration does the following:

1. **Train** a network on the current training set: Adam, batch size 512, early stopping on the validation loss, and an asymmetric MSE loss that penalises underestimating the optimal advisory (`training.py`).
2. **Opportunistic search**: evaluate the network on 1000 random inputs that satisfy the property's precondition and collect any that violate the postcondition.
3. **Formal verification**: if step 2 finds nothing, export the network to ONNX and ask [Marabou](https://github.com/NeuralNetworkVerification/Marabou) for a counterexample.
4. **Repair**: for each counterexample (at most `--counterexamples`), solve a small MIQP with Gurobi. It finds the smallest change to the network's output scores that makes the property hold.
5. **Sampling**: add `--sampling` noisy copies of each repaired point that still satisfy the property.
6. Add the repaired and sampled pairs to the training set and go back to step 1. By default each iteration trains a new network from scratch (use `--keep` to keep training the same one).

The loop stops when Marabou proves that no counterexample exists, so the last network saved is certified for the property.

With `--soft-acceleration` (`-SA` in result names), training also adds a penalty for property violations on random inputs from the precondition region. This usually reduces the number of CEGIS iterations.

The two tolerance parameters are:

- `--delta` (Δ): the margin by which the postcondition must hold. The property is strengthened by Δ.
- `--epsilon` (ε): how far a point must violate the Δ-strengthened postcondition to count as a counterexample. ε must be smaller than Δ.

## Dataset

The networks are trained on the HorizontalCAS dataset from:

> Kyle D. Julian, Mykel J. Kochenderfer.
> *Guaranteeing safety for neural network-based aircraft collision avoidance systems.*
> Digital Avionics Systems Conference (DASC), 2019.

The `.h5` files in `HCAS_polar_networks/` were generated with the generator and original data from <https://github.com/sisl/HorizontalCAS>. There is one file per combination of:

- `pra`, the previous advisory: `0` COC, `1` WL, `2` WR, `3` SL, `4` SR;
- `tau`, the time to loss of vertical separation in seconds: `00 05 10 15 20 30 40 60`.

That makes 5 × 8 = 40 datasets. Each file holds normalised inputs `X` and target scores `y`:

| Inputs (`conf.py`) | Meaning | Denormalisation |
|---|---|---|
| ρ (`rho`) | distance to the intruder | 56000 · x + 11423.75 ft |
| θ (`theta`) | bearing angle to the intruder | 2π · x rad |
| ψ (`psi`) | relative heading of the intruder | 2π · x rad |

The outputs are scores for the five advisories: **COC** (clear of conflict), **WL**/**WR** (weak left/right) and **SL**/**SR** (strong left/right). The advisory with the highest score is the one issued.

After a seeded shuffle, the samples are split into 43,000 for training, 5,396 for validation and the rest for test.

## Properties

Each property is in `properties/P<k>.py`. Each file defines the precondition, the postcondition, the soft loss, the opportunistic search, the Marabou query, the repair constraints and the sampling. `fm.py` selects the module that matches the property number.

| ID | Property | Datasets |
|---|---|---|
| $P_{1}$ | If the intruder is distant, the COC score is always above a threshold. | all |
| $P_{2}$ | If the intruder is distant, the COC score is never the minimum. | all |
| $P_{3}$ | If the intruder is directly ahead and moving towards the ownship, COC is not the maximum. | all |
| $P_{4}$ | If the intruder is directly ahead and moving away from the ownship, COC is not the maximum. | all |
| $P_{5}$ | If the intruder is near and approaching from the left, strong right is advised. | all |
| $P_{6}$ | If the intruder is far enough away, COC is advised. | all |
| $P_{7}$ | If vertical separation is large, a strong turn is never advised. | `tau = 60` |
| $P_{8}$ | With large vertical separation and previous advisory WL, the network advises COC or WL. | `pra = 1`, `tau = 60` |
| $P_{9}$ | Even if the previous advisory was WR, a nearby intruder makes the network advise SL. | `pra = 2` |
| $P_{10}$ | For a far-away intruder, COC is advised. | all |
| $\mathcal{P}_{1}$ = `12345610` | $P_{1} \land P_{2} \land P_{3} \land P_{4} \land P_{5} \land P_{6} \land P_{10}$ | all |
| $\mathcal{P}_{2}$ = `123456810` | $P_{1} \land \dots \land P_{6} \land P_{8} \land P_{10}$ | `pra = 1`, `tau = 60` |
| $\mathcal{P}_{3}$ = `123456910` | $P_{1} \land \dots \land P_{6} \land P_{9} \land P_{10}$ | `pra = 2`, `tau = 60` |

For the combined properties, verification checks each property separately. Repair solves a single MIQP containing the constraints of every property whose precondition holds at the counterexample. `properties/P123467810.py` holds one more combination; it is not among the paper's main experiments.

## Requirements

- Python 3 with `torch`, `numpy`, `pandas`, `h5py`, `matplotlib` and `onnx`
- [`maraboupy`](https://github.com/NeuralNetworkVerification/Marabou) for formal verification
- [`gurobipy`](https://www.gurobi.com/) with a valid Gurobi licence, for repair and for `realizability-analysis.py`
- A LaTeX installation, for the plotting scripts (they set `text.usetex = True`)

For example:

```bash
pip install torch numpy pandas h5py matplotlib onnx maraboupy gurobipy
```

Every property module imports Marabou and Gurobi, so both must be installed even to run training alone. A CUDA GPU is used automatically if one is available.

Run every script from the repository root, because the scripts use relative paths.

## Running a single experiment

```bash
python3 HCASCertifier.py 1 --pra 0 --tau 00 \
    --epsilon 0.0001 --delta 0.01 \
    --learning-rate 0.001 --epochs 1000 --early-stopping 50 \
    --hidden-layers 6 --neurons-per-hidden-layer 50 \
    --counterexamples 50 --sampling 20 --seed 100 \
    --soft-acceleration
```

This example uses the settings from the paper. Run `python3 HCASCertifier.py -h` for the full list of options:

| Option | Default | Description |
|---|---|---|
| `property` (positional) | — | Property ID: `1`–`10`, `12345610`, `123456810`, `123456910` or `123467810` |
| `--pra` | `0` | Previous advisory (selects the dataset) |
| `--tau` | `00` | Time to loss of vertical separation (selects the dataset) |
| `--epsilon` | `0.0001` | ε, the counterexample tolerance |
| `--delta` | `0.01` | Δ, the postcondition margin |
| `--learning-rate` | `0.001` | Adam learning rate |
| `--epochs` | `1000` | Maximum epochs per training phase |
| `--early-stopping` | `50` | Patience, in epochs, on the validation loss |
| `--hidden-layers` | `6` | Number of hidden layers |
| `--neurons-per-hidden-layer` | `50` | Width of each hidden layer |
| `--counterexamples` | `1` | Maximum counterexamples repaired per iteration |
| `--sampling` | `20` | Noisy samples generated around each repaired counterexample |
| `--seed` | `1` | Seed for shuffling and splitting the dataset |
| `--soft-acceleration` | off | Add the soft property loss during training |
| `--keep` | off | Keep training the same network across iterations instead of starting fresh |
| `--cores` | `4` | Number of Marabou workers |
| `--force` | off | Overwrite an existing result directory |

### Output

Each run writes to:

```
experiments/pra<pra>-tau<tau>/<property>-<epsilon>-<delta>-<lr>-<layers>x<neurons>-<counterexamples>-<sampling>-<seed>-<L|K>[-SA]/
```

`L` means a new network was trained at each iteration and `K` means the same network was kept (`--keep`). `-SA` marks soft acceleration. The directory contains:

- `network-<i>.pt`: the PyTorch `state_dict` of the network after CEGIS iteration `i`. The highest `i` is the certified network.
- `stats.json`: per-iteration lists `TrainingTIME`, `OpportunisticSeachTIME`, `FormalVerificationTIME`, `RepairTIME`, `SamplingTIME`, `LossOnValidation`, `LossOnTest`, `OpportunisticCEX`, `FormalVerificationCEX` and `NewPairs`. It also holds `CEGISiterations`, `OriginalInputs` and `TotalTIME` (in seconds). `TotalTIME` is `null` if the run did not finish.

If the result directory already exists, the run stops unless you pass `--force`.

## Reproducing the paper

### All experiments

```bash
python3 Run.py
```

`Run.py` runs, one after another, every combination of property, dataset, (ε, Δ) ∈ {10⁻², 10⁻³, 10⁻⁴}² with Δ > ε, and soft acceleration on or off. It skips properties on datasets where they do not apply. The grid is defined in `experiments.py` and comes to 2016 runs, each with a 24-hour timeout.

> **Note:** `experiments/` already contains the paper's results, and `HCASCertifier.py` skips runs whose directory exists. To recompute the results, move `experiments/` somewhere else first.

### Tables and figures

These scripts read `experiments/` and write to the current directory or `img/`:

| Script | Output |
|---|---|
| `python3 experiments.py` | Prints how many runs are present, which are missing and which timed out |
| `python3 latex-epsilon-delta.py` | `table-epsilon-delta.tex`: the LaTeX table of per-property averages (± std) of iterations, times, losses, counterexamples and new pairs for each (ε, Δ) |
| `python3 convergence.py --epsilon-delta` | `img/convergence-HCAS-epsilon-delta-P<k>.pdf`: the average number of counterexamples during the CEGIS iterations (without `--epsilon-delta`, it uses only ε = 10⁻⁴, Δ = 10⁻² and writes `…-baseline-…`) |
| `python3 hystogram.py` | `img/fv-vs-rep-HCAS.pdf`: average formal verification time compared with repair time, per property |
| `python3 plot-polar.py` | `img/policy/*-{initial,final}.pdf`: polar plots of the advisories of the first and the certified network, with the property's region outlined |
| `python3 realizability-analysis.py` | Uses Gurobi to check every pair and triple of P1–P10 for conflicting postconditions on overlapping preconditions |

`plot.py` is an older Cartesian version of `plot-polar.py`, and both write to the same file names.

## Repository layout

```
HCASCertifier.py          CEGIS loop (entry point for one experiment)
training.py               Training loop and asymmetric MSE loss
fm.py                     Selects the property module for each step
conf.py                   Input/output indices, input bounds, constants
properties/P*.py          One module per property (and per combination)
experiments.py            Experiment grid and result loader
Run.py                    Runs the whole grid
convergence.py, hystogram.py, latex-epsilon-delta.py,
plot-polar.py, plot.py    Scripts that build the tables and figures
realizability-analysis.py Checks whether combinations of properties can be satisfied together
HCAS_polar_networks/      HorizontalCAS training data (40 .h5 files, about 130 MB)
experiments/              Raw results of the paper (about 650 MB)
img/                      Figures used in the paper
```
