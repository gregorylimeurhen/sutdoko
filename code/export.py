import argparse
import importlib.util
import json
import pathlib
import torch


def load_utils(path):
	spec = importlib.util.spec_from_file_location("_app_utils", path)
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


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
	code_dir = pathlib.Path(__file__).resolve().parent
	utils = load_utils(code_dir / "utils.py")
	device = torch.device("cpu")
	model_path = pathlib.Path(model_path).resolve()
	snap_path = model_path.with_name("snapshot.zip")
	if not snap_path.exists():
		raise FileNotFoundError(str(snap_path))
	with utils.loaded_snapshot(snap_path) as (snap_root, ev):
		model, tok, rooms = ev.load_checkpoint(model_path, device)
		room_map = ev.load_room_lookup(snap_root)
		seed = ev.load_seed(snap_root)
		trie = ev.build_room_trie(rooms, tok)
	out_dir = pathlib.Path(out_dir).resolve()
	out_dir.mkdir(parents=True, exist_ok=True)
	state = model.state_dict()
	meta = {}
	offset = 0
	path = out_dir / "weights.bin"
	with path.open("wb") as file:
		for name, tensor in state.items():
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
		"rooms": sorted(rooms),
		"room_lookup": {key: value for key, value in items},
		"seed": seed,
		"tensors": meta,
		"trie": trie,
		"tokenizer": tok.to_dict(),
	}
	text = json.dumps(assets, indent=2) + "\n"
	(out_dir / "assets.json").write_text(text)
	return out_dir


def parse_args():
	parser = argparse.ArgumentParser()
	path = pathlib.Path(__file__).resolve().parent
	cfg = load_utils(path / "utils.py").load_config(path, "export")
	run = str(cfg["run"]).strip()
	model = path / "runs" / run / "train" / "model.pt"
	out = path / "app"
	parser.add_argument("model", nargs="?", default=str(model))
	parser.add_argument("--out", default=str(out))
	return parser.parse_args()


def main():
	args = parse_args()
	out_dir = export_model(args.model, args.out)
	print(out_dir)


if __name__ == "__main__":
	main()
