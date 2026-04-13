import csv, datetime, importlib, json, pathlib, shutil, sys, tomllib
import torch


def device_for():
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def checkpoint_name(value):
	name = str(value).strip()
	return name if name.endswith(".pt") else f"{name.zfill(4)}.pt"


def ensure_run_dir(root):
	run_dir = pathlib.Path(root) / "runs" / ("E" + datetime.datetime.now().strftime("%M%S"))
	(run_dir / "results").mkdir(parents=True, exist_ok=True)
	return run_dir


def latest_train_run_dir(root):
	run_dirs = [path for path in (pathlib.Path(root) / "runs").iterdir() if path.is_dir() and path.name.startswith("T")]
	if not run_dirs:
		raise FileNotFoundError("No train runs found")
	return max(run_dirs, key=lambda path: path.stat().st_mtime)


def load_training_src(train_run_dir):
	for name in list(sys.modules):
		if name == "src" or name.startswith("src."):
			del sys.modules[name]
	sys.path = [path for path in sys.path if path != str(train_run_dir)]
	sys.path.insert(0, str(train_run_dir))
	return importlib.import_module("src")


def snapshot_run_files(root, train_run_dir, run_dir):
	shutil.copytree(train_run_dir / "data", run_dir / "data", dirs_exist_ok=True)
	shutil.copytree(train_run_dir / "src", run_dir / "src", dirs_exist_ok=True)
	for name in ["n2a.tsv", "test.tsv"]:
		shutil.copy2(root / "data" / name, run_dir / "data" / name)
	for name in ["config.toml", "preprocess.py", "requirements.txt", "test.py", "train.py"]:
		shutil.copy2(root / name, run_dir / name)


def write_answers(path, rows):
	with path.open("w", newline="") as file:
		writer = csv.DictWriter(file, fieldnames=["input", "gold_room", "gold", "identity", "levenshtein", "ours"], quoting=csv.QUOTE_ALL)
		writer.writeheader()
		writer.writerows(rows)


def write_scores(path, scores):
	path.write_text(json.dumps(scores, indent=2) + "\n")


def main():
	root = pathlib.Path(__file__).resolve().parent
	config = tomllib.loads((root / "config.toml").read_text())["test"]
	device = device_for()
	run_dir = ensure_run_dir(root)
	train_run_dir = latest_train_run_dir(root)
	src = load_training_src(train_run_dir)
	snapshot_run_files(root, train_run_dir, run_dir)
	checkpoint = train_run_dir / "checkpoints" / checkpoint_name(config["checkpoint"])
	model, tokenizer, rooms = src.models.load_checkpoint(checkpoint, device)
	rows = src.utils.load_test_rows(run_dir)
	room_lookup = src.utils.load_room_lookup(run_dir)
	scores, details = src.metrics.evaluate_rows(model, rows, tokenizer, device, room_lookup, rooms)
	write_answers(run_dir / "results" / "answers.csv", details)
	write_scores(run_dir / "results" / "scores.json", scores)
	print(json.dumps(scores))


if __name__ == "__main__":
	main()
