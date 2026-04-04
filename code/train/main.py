import shutil
import time
import torch
import wandb
import yaml

from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
from src import *


def compact_items(data):
	return "(" + ", ".join(f"{key}={data[key]}" for key in data) + ")"


def device_for():
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_run_dir(root, identifier):
	run_dir = Path(root) / "runs" / identifier
	(run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
	return run_dir


def run_id():
	return datetime.now().strftime("%Y%m%d%H%M%S")


def snapshot_run_files(root, run_dir):
	shutil.copytree(root / "data", run_dir / "data", dirs_exist_ok=True)
	shutil.copytree(root / "src", run_dir / "src", dirs_exist_ok=True)
	for name in ["config.yaml", "main.py"]:
		shutil.copy2(root / name, run_dir / name)


def main():
	start = time.time()
	load_dotenv()
	root = Path(__file__).resolve().parent
	config = yaml.safe_load((root / "config.yaml").read_text())
	datasets = load_training_rows(root)
	device = device_for()
	epochs = int(config["epochs"])
	identifier = run_id()
	original_config = dict(config)
	rows = datasets["pretrain"] + datasets["finetune"]
	run_dir = ensure_run_dir(root, identifier)
	snapshot_run_files(root, run_dir)
	set_seed(int(config["seed"]))
	tokenizer = build_tokenizer(rows)
	model = build_model(config["architecture"], tokenizer, rows_block_size(rows)).to(device)
	batch_size = infer_batch_size(model, rows, tokenizer, device)
	with wandb.init(dir=str(run_dir), name=identifier, project="mlops") as run:
		epoch = 0
		epoch = train_stage(model, "pretrain", datasets["pretrain"], tokenizer, device, run_dir, epochs, batch_size, epoch, run, identifier)
		train_stage(model, "finetune", datasets["finetune"], tokenizer, device, run_dir, epochs, batch_size, epoch, run, identifier)
		run.log({
			"summary": wandb.Table(
				columns=["id", "config", "seconds"],
				data=[[identifier, compact_items(original_config), time.time() - start]],
			)
		})


if __name__ == "__main__":
	main()
