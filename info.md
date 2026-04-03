# Semantic Campus Locations

Task: "Given a room name, return the corresponding room address."

The code repository is structured as follows.
```
./code/
  runs/
    1a2b3c/ # Example of the format I want:
      checkpoints/ # to store weights
        001.pt # 3-digit
        ...
        100.pt
      config.yaml
      data.md
      src.py
  config.yaml
  data.md
  main.py
  scoreboard.csv
  src.py # contains variables, classes, and functions for datasets, architectures, metrics
```

Let $\operatorname{NLS}(p, g) = 1 - \frac{\operatorname{lev}(p, g)}{\max(|p|, |g|)}$.

`config.yaml` contains something like
```
dataset: "pretrain"
architecture: "GPT2"
metrics: ["NLS", "TTLT"]
seed: 37
```
where the value for:
1. `dataset` is a dataset name found in `data.md`.
2. `architecture`, `metrics` are class names found in `src.py`.
3. `seed` is a number.

`scoreboard.csv` contains a table of the form `id, config, scores, duration`
where `id` is ID of the run,
and `config` is the `config.yaml` of the run spread out compactly like
e.g. `(dataset=..., architecture=..., metrics=..., seed=37)`,
and `scores` looks like e.g. `(val_mean_nls=..., test_mean_nls=..., val_mean_ttlt=..., test_mean_ttlt=...)`,
and `duration` is the time taken from the start to the end of the run e.g. `1600`.

`main.py` is the entry point that
- sets a run ID by taking the first 6 digits of the SHA-1 of the UNIX timestamp at the current time.
- enforces reproducibility using the seed from `config.yaml`.
- runs training (pre-training -> fine-tuning) with:
  - AdamW optimizer on default settings.
  - checkpointing and logging of train_loss, val_loss, grad_norm at every epoch, and early stopping, with the convention that the last epoch minimises val loss.
  - a `tqdm` progress bar of length 80.

## Data

Suppose `<` is BOS, `>` is EOS, `n` denotes a name and `A` a list of addresses.

`data.md` contains datasets named:
1. `edges`: contains raw mappings from names to addresses; names are immutable and canonically written (with British spelling).
2. `variants`: contains name abbreviations (variations on names used to generate all possible names).
3. `pretrain`: each row is either a name (e.g. `think tank 1`) or an address (e.g. `1.201`).
4. `finetune`: each row is of the form `n<A>` (e.g. `library<1.101, 1.201, 1.301>`).
5. `val`: each row is of the form `n<A>` (e.g. `library<1.101, 1.201, 1.301>`): an American or drop-plural or American + drop re-write of a row in `finetune`.*
6. `test`: each row is of the form `n<A>` (e.g. `library<1.101, 1.201, 1.301>`): an American or drop-plural or American + drop re-write of a row in `finetune`.*

*: Assume every row in `finetune` can have at most 1 American/drop-plural/American + drop-plural re-write; split all possible re-writes 50/50 so that |`val`| = |`test`|, and `val`, `test` disjoint.
