import csv, json, pathlib
import src.utils


def write_answers(path, rows):
	with path.open("w", newline="") as file:
		writer = csv.DictWriter(file, fieldnames=["input", "gold_room", "gold", "identity", "levenshtein", "ours"], quoting=csv.QUOTE_ALL)
		writer.writeheader()
		writer.writerows(rows)


def write_scores(path, scores):
	path.write_text(json.dumps(scores, indent=2) + "\n")


def main():
	root = pathlib.Path(__file__).resolve().parent
	config = src.utils.load_config(root, "test")
	device = src.utils.device_for()
	print(src.utils.device_for().type)
	run_dir = src.utils.ensure_run_dir(root, "E")
	(run_dir / "results").mkdir(parents=True, exist_ok=True)
	train_run_dir = root / "runs" / str(config["run"]).strip()
	with src.utils.extracted_snapshot(train_run_dir / "snapshot.zip") as train_snapshot_root:
		src.utils.write_snapshot(run_dir / "snapshot.zip", train_snapshot_root, root, {"n2a.tsv": root / "data" / "n2a.tsv", "test.tsv": root / "data" / "test.tsv"})
	with src.utils.loaded_snapshot(run_dir / "snapshot.zip", "src") as (eval_snapshot_root, eval_src):
		model, tokenizer, rooms = eval_src.models.load_checkpoint(train_run_dir / "model.pt", device)
		rows = eval_src.utils.load_test_rows(eval_snapshot_root)
		room_lookup = eval_src.utils.load_room_lookup(eval_snapshot_root)
		scores, details = eval_src.metrics.evaluate_rows(model, rows, tokenizer, device, room_lookup, rooms)
	write_answers(run_dir / "results" / "answers.csv", details)
	write_scores(run_dir / "results" / "scores.json", scores)
	print(json.dumps(scores))


if __name__ == "__main__":
	main()
