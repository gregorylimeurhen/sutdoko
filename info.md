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
  src.py # contains variables, classes, and functions for datasets, architectures, metrics
```

Let $\operatorname{NLS}(p, g) = 1 - \frac{\operatorname{lev}(p, g)}{\max(|p|, |g|)}$.

`config.yaml` contains something like
```
architecture: "GPT2"
epochs: 100
seed: 37
```
where the value for:
2. `architecture` is a class name found in `src.py`.
3. `epochs`, `seed` are natural numbers.

We log to W&B. We log a table of the form `id, config, scores, duration`
where `id` is ID of the run,
and `config` is the `config.yaml` of the run spread out compactly like
e.g. `(architecture=..., epochs=..., seed=...)`,
and `scores` looks like e.g. `(val_mean_nls=..., test_mean_nls=...)`,
and `duration` is the time taken from the start to the end of the run e.g. `1600`.

`main.py` is the entry point that:
- sets a run ID by taking the first 6 hex chars of SHA-1(current UNIX timestamp).
- enforces reproducibility using the seed from `config.yaml`.
- runs training (pre-training -> fine-tuning) with:
  - AdamW optimizer on default settings.
  - checkpointing and logging of train_loss, val_loss, grad_norm at every epoch.
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

*: There are 2 levers: American and drop-plural. Mathematically, this gives rise to possibilities: American, drop-plural, or both (American + drop-plural), or neither. Assume every row in `finetune` can have <=1 American re-write and <=1 drop-plural re-write and <=1 American + drop-plural re-write (mathematically by enumeration, a row can have (1) 0 or 1 American re-write, (2) 0 or 1 drop-plural re-write, and 0 or 1 American + drop-plural re-write); split all possible re-writes 50/50 so that |`val`| = |`test`|, and `val`, `test` disjoint.
