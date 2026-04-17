import argparse
import importlib.util
import json
import pathlib
import sys
import torch


def project_root():
	return pathlib.Path(__file__).resolve().parent.parent


def app_root():
	return pathlib.Path(__file__).resolve().parent


def experiments_root():
	return project_root() / "experiments"


def load_utils():
	path = experiments_root() / "utils.py"
	name = "_experiments_utils"
	spec = importlib.util.spec_from_file_location(name, path)
	module = importlib.util.module_from_spec(spec)
	sys.modules.pop(name, None)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def latest_model(root=None):
	root = experiments_root() if root is None else pathlib.Path(root)
	pat = root / "runs"
	key = lambda path: path.stat().st_mtime
	paths = sorted(pat.glob("*/train/model.pt"), key=key)
	if paths:
		return paths[-1]
	raise RuntimeError("no experiments/runs/*/train/model.pt found")


def configured_model():
	root = experiments_root()
	utils = load_utils()
	run = int(utils.load_config(root, "build")["run"])
	path = root / "runs" / f"{run}" / "train" / "model.pt"
	if path.exists():
		return path
	raise RuntimeError(f"missing {path}")


def dump_tensor(file, tensor, offset):
	data = tensor.detach().cpu().float().contiguous()
	blob = bytes(data.untyped_storage())
	file.write(blob)
	return {
		"offset": offset,
		"shape": list(data.shape),
		"size": data.numel(),
	}


def export_model(model_path, out_dir):
	dev = torch.device("cpu")
	exp = experiments_root()
	ev0 = load_utils()
	model_path = pathlib.Path(model_path).resolve()
	snap_path = model_path.with_name("snapshot.zip")
	print(f"export model {model_path}")
	print(f"snapshot {snap_path}")
	if not snap_path.exists():
		raise FileNotFoundError(str(snap_path))
	with ev0.loaded_snapshot(snap_path) as (snap_root, ev):
		print("load checkpoint")
		model, tok, rooms = ev.load_checkpoint(model_path, dev)
		if hasattr(ev, "load_aliases"):
			print("load aliases from snapshot")
			aliases = ev.load_aliases(snap_root)
		else:
			print("load aliases from repo")
			aliases = ev0.load_aliases(exp)
		print("load room lookup")
		room_map = ev.load_room_lookup(snap_root)
		seed = ev.load_seed(snap_root)
		trie = ev.build_room_trie(rooms, tok)
	out_dir = pathlib.Path(out_dir).resolve()
	out_dir.mkdir(parents=True, exist_ok=True)
	offset = 0
	meta = {}
	path = out_dir / "weights.bin"
	print(f"write weights {path}")
	with path.open("wb") as file:
		for name, tensor in model.state_dict().items():
			print(f"tensor {name}")
			info = dump_tensor(file, tensor, offset)
			meta[name] = info
			offset += info["size"] * 4
	config = {
		"depth": model.config.depth,
		"sequence_len": model.config.sequence_len,
		"vocab_size": model.config.vocab_size,
		"n_head": model.config.n_head,
		"n_embd": model.config.n_embd,
	}
	items = sorted(room_map.items())
	assets = {
		"config": config,
		"aliases": aliases,
		"rooms": sorted(rooms),
		"room_lookup": {key: value for key, value in items},
		"seed": seed,
		"tensors": meta,
		"trie": trie,
		"tokenizer": tok.to_dict(),
	}
	text = json.dumps(assets, indent=2) + "\n"
	print(f"write assets {out_dir / 'assets.json'}")
	print(f"rooms {len(rooms)} aliases {len(aliases)}")
	(out_dir / "assets.json").write_text(text)
	return out_dir


def parse_args():
	parser = argparse.ArgumentParser()
	model = configured_model()
	parser.add_argument("model", nargs="?", default=str(model))
	return parser.parse_args()


def main():
	args = parse_args()
	app = app_root()
	model = pathlib.Path(args.model).resolve()
	print(f"app {app}")
	print(f"experiments {experiments_root()}")
	print(f"model {model}")
	if not model.exists():
		raise FileNotFoundError(str(model))
	export_model(model, app)


if __name__ == "__main__":
	main()
