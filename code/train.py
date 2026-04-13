import datetime, pathlib, shutil, time, tomllib
import dotenv, torch, wandb
import src.models, src.utils


def device_for():
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_run_dir(root):
	run_dir = pathlib.Path(root) / "runs" / ("T" + datetime.datetime.now().strftime("%M%S"))
	(run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
	return run_dir


def snapshot_run_files(root, run_dir):
	shutil.copytree(root / "data", run_dir / "data", dirs_exist_ok=True)
	shutil.copytree(root / "src", run_dir / "src", dirs_exist_ok=True)
	for name in ["config.toml", "preprocess.py", "requirements.txt", "test.py", "train.py"]:
		shutil.copy2(root / name, run_dir / name)


def remove_wandb_dir(run_dir):
	shutil.rmtree(run_dir / "wandb", ignore_errors=True)


def main():
	start = time.time()
	root = pathlib.Path(__file__).resolve().parent
	dotenv.load_dotenv(root / ".env")
	config = tomllib.loads((root / "config.toml").read_text())["train"]
	rows = src.utils.load_training_rows(root)
	device = device_for()
	run_dir = ensure_run_dir(root)
	snapshot_run_files(root, run_dir)
	src.utils.set_seed(0)
	tokenizer = src.utils.build_tokenizer(root)
	model = src.models.build_model(config["depth"], tokenizer, src.utils.rows_block_size(rows)).to(device)
	try:
		with wandb.init(
			config=config,
			dir=str(run_dir),
			name=run_dir.name,
			project="mlops",
			settings=wandb.Settings(quiet=True, show_info=False, show_warnings=False, console="off"),
		) as run:
			batch_size = src.models.train(model, rows, tokenizer, device, run_dir, config["epochs"], run)
			run.log({"batch_size": batch_size, "seconds": time.time() - start})
	finally:
		remove_wandb_dir(run_dir)


if __name__ == "__main__":
	main()
