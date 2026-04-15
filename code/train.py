import dotenv
import pathlib
import utils
import wandb


def main():
	dir_ = pathlib.Path(__file__).resolve().parent
	dotenv.load_dotenv(dir_.parent / ".env")
	cfg = utils.load_config(dir_, "train")
	seed = utils.load_seed(dir_)
	train = utils.load_rows(dir_, "train")
	val = utils.load_rows(dir_, "val")
	dev = utils.device_for()
	print(dev.type)
	run_dir = utils.ensure_run_dir(dir_, "train")
	utils.write_snapshot(run_dir / "snapshot.zip", dir_)
	utils.set_seed(seed)
	tok = utils.build_tokenizer(dir_)
	train_size = utils.rows_block_size(train)
	val_size = utils.rows_block_size(val)
	block_size = max(train_size, val_size)
	model = utils.build_model(cfg["depth"], tok, block_size).to(dev)
	n = run_dir.parent.name
	rd = str(run_dir)
	sets = wandb.Settings(quiet=True, show_info=False, show_warnings=False)
	run = wandb.init(config=cfg, dir=rd, name=n, project="mlops", settings=sets)
	with run:
		batch = cfg.get("batch_size")
		tol = cfg["tolerance"]
		path = run_dir / "model.pt"
		utils.train(model, train, val, tok, dev, path, tol, run, seed, batch)


if __name__ == "__main__":
	main()
