import csv
import json
import pathlib
import utils


def main():
	dir_ = pathlib.Path(__file__).resolve().parent
	cfg = utils.load_config(dir_, "test")
	dev = utils.device_for()
	print(dev.type)
	train_run_dir = dir_ / "runs" / str(cfg["run"]).strip() / "train"
	run_dir = train_run_dir.parent / "test"
	results_dir = run_dir / "results"
	results_dir.mkdir(parents=True, exist_ok=True)
	utils.write_snapshot(run_dir / "snapshot.zip", dir_)
	answers_path = results_dir / "answers.csv"
	fieldnames = ["input", "gold_room", "gold", *utils.BASELINE_NAMES]
	snap_path = run_dir / "snapshot.zip"
	model_path = train_run_dir / "model.pt"
	with answers_path.open("w", newline="") as file:
		writer = csv.DictWriter(file, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
		writer.writeheader()
		with utils.loaded_snapshot(snap_path) as (root2, ev):
			model, tok, rooms = ev.load_checkpoint(model_path, dev)
			rows = ev.load_rows(root2, "test")
			room_map = ev.load_room_lookup(root2)
			seed = ev.load_seed(root2)
			score = ev.evaluate_rows_into
			write = writer.writerow
			scores = score(model, rows, tok, dev, room_map, rooms, write, seed)
	score_text = json.dumps(scores)
	(results_dir / "scores.json").write_text(json.dumps(scores, indent=2) + "\n")
	print(score_text)


if __name__ == "__main__":
	main()
