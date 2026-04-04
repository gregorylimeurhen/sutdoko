# Semantic Campus Locations

Task: "Given a room name, return the corresponding room address."

The code repository is structured as follows. Hereafter, let `./` represent the `code/` directory.
```
./
  train/
    runs/
      <train run ID>/
        checkpoints/ # to store weights
          001.pt # padded 3-digit
          ...
          100.pt
        data/
        src/
        config.yaml
        main.py
    data/
    src/
    config.yaml
    main.py
  test/
    runs/
      <test run ID>/
        results/
          answers.csv # use "input, gold, output" format
          scores.json # contain mean_nls (mean NLS score) somewhere
        src/
        data.txt
        config.yaml
        main.py
    config.yaml
    main.py
    data.txt # contain test set
```

Let $\operatorname{NLS}(p, g) = 1 - \frac{\operatorname{lev}(p, g)}{\max(|p|, |g|)}$.

`train/config.yaml` contains something like
```
architecture: "GPT2"
epochs: 100
seed: 37
```
where the value for:
1. `architecture` is a class name found in `train/src/`.
2. `epochs`, `seed` are natural numbers.

`test/config.yaml` contains something like
```
run: ...
checkpoint: ...
```
where the value for:
1. `run` is a training run ID.
2. `checkpoint` is a checkpoint number.

`train/main.py` should:
- set a run ID.
- enforce reproducibility using the seed from `config.yaml`.
- run training (pre-training -> fine-tuning) with:
  - AdamW optimizer on default settings.
  - a `tqdm` progress bar of length 80.
  - file copying and saving (see file tree above), and logging to Weights & Biases (W&B):
    - global statistic `id, config, seconds` where `config` is spread out as before, and `seconds` is time taken in seconds for run to complete.
    - per-epoch stuff `id, epoch, stage, loss, grad_norm` (in this order).

`test/main.py` should:
- locally log the answer(s) of a model (checkpoint) on the test set using `input, gold, output` (see above file tree) and save some other stuff too.

Suppose `<` is BOS, `>` is EOS, `n` denotes a name and `A` a list of addresses.
1. `train/data/edges.tsv`: contains raw mappings from names to addresses; names are immutable and canonically written (with British spelling).
2. `train/data/variants.csv`: contains name abbreviations (variations on names used to generate all possible names).
3. `train/data/pretrain.txt`: contains pre-training data.
4. `train/data/finetune.txt`: contains fine-tuning data.

We shall also have a test dataset `test/data.txt`: each row is of the form `n<A>` (e.g. `library<1.101, 1.201, 1.301>`): an American or drop-plural or American + drop re-write of a row in `finetune`.*

*: There are 2 levers: American and drop-plural. Mathematically, this gives rise to possibilities: American, drop-plural, or both (American + drop-plural), or neither. Assume every row in `finetune` can have <=1 American re-write and <=1 drop-plural re-write and <=1 American + drop-plural re-write (mathematically by enumeration, a row can have (1) 0 or 1 American re-write, (2) 0 or 1 drop-plural re-write, and 0 or 1 American + drop-plural re-write).
