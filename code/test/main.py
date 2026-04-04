import csv
import importlib
import json
import shutil
import sys
import torch
import yaml
from datetime import datetime
from pathlib import Path


def device_for():
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_run_dir(root, identifier):
	run_dir = Path(root) / "runs" / identifier
	(run_dir / "results").mkdir(parents=True, exist_ok=True)
	return run_dir


def run_id():
	return datetime.now().strftime("%Y%m%d%H%M%S")


def load_training_src(train_run_dir):
	for name in list(sys.modules):
		if name == "src" or name.startswith("src."):
			del sys.modules[name]
	sys.path.insert(0, str(train_run_dir))
	return importlib.import_module("src")


def snapshot_run_files(root, train_run_dir, run_dir):
	shutil.copytree(train_run_dir / "src", run_dir / "src", dirs_exist_ok=True)
	for name in ["config.yaml", "data.txt", "main.py"]:
		shutil.copy2(root / name, run_dir / name)


def write_answers(path, rows):
	with path.open("w", newline="") as file:
		writer = csv.writer(file)
		writer.writerow(["input", "gold", "output"])
		for row in rows:
			writer.writerow([row["input"], row["gold"], row["output"]])


def write_scores(path, scores):
	path.write_text(json.dumps(scores, indent=2) + "\n")


def main():
	root = Path(__file__).resolve().parent
	config = yaml.safe_load((root / "config.yaml").read_text())
	device = device_for()
	identifier = run_id()
	run_dir = ensure_run_dir(root, identifier)
	train_run_dir = root.parent / "train" / "runs" / str(config["run"])
	src = load_training_src(train_run_dir)
	snapshot_run_files(root, train_run_dir, run_dir)
	train_config = yaml.safe_load((train_run_dir / "config.yaml").read_text())
	checkpoint = train_run_dir / "checkpoints" / f"{str(config['checkpoint']).zfill(4)}.pt"
	model, tokenizer = src.load_checkpoint(checkpoint, train_config["architecture"], device)
	rows = src.load_test_rows(root / "data.txt")
	scores, details = src.evaluate_rows(model, rows, tokenizer, src.instantiate_metric("NLS"), device, src.infer_max_tokens(rows, tokenizer))
	write_answers(run_dir / "results" / "answers.csv", details)
	write_scores(run_dir / "results" / "scores.json", scores)


if __name__ == "__main__":
	main()
