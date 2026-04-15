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
	sets = wandb.Settings(quiet=True, show_info=False, show_warnings=False)
	name = run_dir.parent.name
	path = run_dir / "model.pt"
	init = wandb.init
	dir2 = str(run_dir)
	run = init(config=cfg, dir=dir2, name=name, project="mlops", settings=sets)
	with run:
		utils.train(model, train, val, tok, dev, path, cfg["tolerance"], run, seed)


if __name__ == "__main__":
	main()
