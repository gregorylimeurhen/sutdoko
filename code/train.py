import pathlib, shutil, time
import dotenv, wandb
import src.models, src.utils


def remove_wandb_dir(run_dir):
	shutil.rmtree(run_dir / "wandb", ignore_errors=True)


def main():
	start = time.time()
	root = pathlib.Path(__file__).resolve().parent
	dotenv.load_dotenv(root / ".env")
	config = src.utils.load_config(root, "train")
	rows = src.utils.load_training_rows(root)
	device = src.utils.device_for()
	print(src.utils.device_for().type)
	run_dir = src.utils.ensure_run_dir(root, "T")
	src.utils.write_snapshot(run_dir / "snapshot.zip", root)
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
			batch_size = src.models.train(model, rows, tokenizer, device, run_dir / "model.pt", config["epochs"], run)
			run.log({"batch_size": batch_size, "seconds": time.time() - start})
	finally:
		remove_wandb_dir(run_dir)


if __name__ == "__main__":
	main()
