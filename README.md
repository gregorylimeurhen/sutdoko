# sutd404

Room-to-address lookup made easy!

- **Web application**: [sutd404.vercel.app](https://sutd404.vercel.app).
- **Training artefacts**: [hf.co/gregorylimeurhen/sutd404](https://hf.co/gregorylimeurhen/sutd404).

## Setup

Follow these steps after cloning our repository if you want to run our code locally.

### Web Application

Follow these steps if you want to run our web application locally.

1. Optionally, download and extract latest weights from latest training artefact into `./experiments/runs/`.
2. Optionally, run `./app/build.py`.
3. Open `./app/index.html`. For example:
```bash
# sutd404 $
            open ./app/index.html # MacOS
            xdg-open ./app/index.html # Linux
```

### Experiments

Follow these steps if you want to run our experiments locally.

1. Rename `./experiments/.env.example` to `./experiments/.env`. For example:
```bash
# sutd404 $
            mv ./experiments/.env.example ./experiments/.env
```
2. Set `WANDB_API_KEY` in `./experiments/.env` to working [Weights & Biases (W&B)](http://wandb.ai) API key.
3. Install packages in `./experiments/requirements.txt`. For example:
```bash
# sutd404 $
            pip install -r ./experiments/requirements.txt
```
4. Run
```bash
# sutd404 $
            ./experiments/run.sh
```

## Structure

```
.
├── app
│   ├── assets.json
│   ├── build.py                 # build script, entry
│   ├── deploy.py                # deployment script, entry
│   ├── index.css
│   ├── index.html               # app markup, entry
│   ├── index.js
│   ├── weights.bin
│   └── worker.js
├── experiments
│   ├── .env
│   ├── .env.example
│   ├── config.toml              # pipeline configuration, entry
│   ├── data
│   │   ├── aliases.tsv          # room aliases
│   │   ├── boundaries.txt       # keyboard boundary list
│   │   ├── edges.tsv            # room-to-address relations
│   │   ├── layout.txt           # keyboard layout map
│   │   ├── n2a.tsv              # room-to-address function
│   │   ├── neighbors.json       # keyboard neighbour map
│   │   ├── test.tsv             # testing dataset
│   │   ├── train.tsv            # training dataset
│   │   └── val.tsv              # validation dataset
│   ├── preprocess.py            # preprocessing script
│   ├── requirements.txt
│   ├── runs
│   │   └── <MMSS>
│   │       ├── test
│   │       │   ├── results
│   │       │   └── snapshot.zip
│   │       └── train
│   │           ├── latest.pt    # latest weights
│   │           ├── model.pt     # best weights
│   │           ├── snapshot.zip
│   │           └── wandb        # training logs
│   ├── test.py                  # testing script, entry
│   ├── train.py                 # training script, entry
│   └── utils.py
├── paper
│   ├── bibliography.bib
│   ├── main.pdf
│   └── main.tex
├── slides
│   ├── bibliography.bib
│   ├── main.pdf
│   └── main.tex
└── README.md
```
