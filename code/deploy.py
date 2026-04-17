import argparse
import hashlib
import json
import pathlib
import shutil
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import torch
import utils


class ApiError(RuntimeError):
	def __init__(self, code, method, url, text):
		super().__init__(f"{method} {url} failed: {text}")
		self.code = code


def load_token(root):
	path = root / ".env"
	if not path.exists():
		raise RuntimeError("missing VERCEL_ACCESS_TOKEN in code/.env")
	for line in path.read_text().splitlines():
		key, sep, value = line.partition("=")
		if key != "VERCEL_ACCESS_TOKEN" or not sep:
			continue
		value = value.strip().strip("\"'")
		if value:
			return value
	raise RuntimeError("missing VERCEL_ACCESS_TOKEN in code/.env")


def load_deploy(root):
	cfg = utils.load_config(root, "deploy")
	api = str(cfg["api"]).rstrip("/")
	project = str(cfg["project"]).strip()
	team = str(cfg["team"]).strip()
	return {"api": api, "project": project, "team": team}


def err_text(err):
	data = err.read()
	if not data:
		return f"{err.code} {err.reason}"
	try:
		obj = json.loads(data)
	except json.JSONDecodeError:
		text = data.decode("utf-8", "replace").strip()
		return text or f"{err.code} {err.reason}"
	if not isinstance(obj, dict):
		return json.dumps(obj)
	msg = obj.get("message")
	if msg:
		return msg
	code = obj.get("code")
	if code is not None:
		return f"{code}"
	return json.dumps(obj)


def latest_model(root):
	pat = root / "runs"
	key = lambda path: path.stat().st_mtime
	paths = sorted(pat.glob("*/train/model.pt"), key=key)
	if paths:
		return paths[-1]
	raise RuntimeError("no code/runs/*/train/model.pt found")


def api_url(cfg, path, query=None):
	url = f"{cfg['api']}{path}"
	if not query:
		return url
	qs = urllib.parse.urlencode(query)
	return f"{url}?{qs}"


def scope(cfg):
	team = cfg["team"]
	if team:
		return {"slug": team}
	return {}


def get_project(cfg, tok):
	name = urllib.parse.quote(cfg["project"])
	url = api_url(cfg, f"/v9/projects/{name}", scope(cfg))
	return req("GET", url, tok)


def make_project(cfg, tok):
	url = api_url(cfg, "/v11/projects", scope(cfg))
	body = json.dumps({"name": cfg["project"]}).encode()
	return req("POST", url, tok, body, "application/json")


def ensure_project(cfg, tok):
	name = cfg["project"]
	print(f"project {name}")
	try:
		print("check project")
		return get_project(cfg, tok)
	except ApiError as err:
		if err.code != 404:
			raise
	print("create project")
	try:
		return make_project(cfg, tok)
	except ApiError as err:
		if err.code != 409:
			raise
	print("project already exists")
	return get_project(cfg, tok)


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
	root = pathlib.Path(__file__).resolve().parent
	model_path = pathlib.Path(model_path).resolve()
	snap_path = model_path.with_name("snapshot.zip")
	print(f"export model {model_path}")
	print(f"snapshot {snap_path}")
	if not snap_path.exists():
		raise FileNotFoundError(str(snap_path))
	with utils.loaded_snapshot(snap_path) as (snap_root, ev):
		print("load checkpoint")
		model, tok, rooms = ev.load_checkpoint(model_path, dev)
		if hasattr(ev, "load_aliases"):
			print("load aliases from snapshot")
			aliases = ev.load_aliases(snap_root)
		else:
			print("load aliases from repo")
			aliases = utils.load_aliases(root)
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


def build_dir(root, model):
	app = root / "app"
	tmp = tempfile.TemporaryDirectory()
	out = pathlib.Path(tmp.name)
	print(f"build dir {out}")
	print(f"copy app {app}")
	shutil.copytree(app, out, dirs_exist_ok=True)
	export_model(model, out)
	return tmp, out


def upload_file(cfg, tok, data):
	sha = hashlib.sha1(data).hexdigest()
	url = api_url(cfg, "/v2/files", scope(cfg))
	headers = {"x-vercel-digest": sha}
	req("POST", url, tok, data, "application/octet-stream", headers)
	return sha


def req(method, url, tok, body=None, ctype=None, extra=None):
	headers = {"Authorization": f"Bearer {tok}"}
	if ctype:
		headers["Content-Type"] = ctype
	if extra:
		headers.update(extra)
	req_ = urllib.request.Request(url, body, headers, method=method)
	try:
		with urllib.request.urlopen(req_, timeout=60) as res:
			data = res.read()
			if not data:
				return None
			return json.loads(data)
	except urllib.error.HTTPError as err:
		text = err_text(err)
		raise ApiError(err.code, method, url, text) from None


def create_deploy(cfg, tok, root):
	files = []
	print(f"upload dir {root}")
	for path in sorted(root.rglob("*")):
		if path.is_dir():
			continue
		data = path.read_bytes()
		name = path.relative_to(root).as_posix()
		size = len(data)
		print(f"upload {name} {size}")
		item = {
			"file": name,
			"sha": upload_file(cfg, tok, data),
			"size": size,
		}
		files.append(item)
	body = {
		"files": files,
		"name": cfg["project"],
		"project": cfg["project"],
		"projectSettings": {"framework": None},
		"target": "production",
	}
	query = scope(cfg)
	query["skipAutoDetectionConfirmation"] = "1"
	url = api_url(cfg, "/v13/deployments", query)
	body = json.dumps(body).encode()
	print(f"create deploy files {len(files)}")
	return req("POST", url, tok, body, "application/json")


def get_deploy(cfg, dep_id, tok):
	path = urllib.parse.quote(dep_id)
	url = api_url(cfg, f"/v13/deployments/{path}", scope(cfg))
	return req("GET", url, tok)


def wait_ready(cfg, dep_id, tok):
	last = None
	print(f"wait deploy {dep_id}")
	while True:
		try:
			row = get_deploy(cfg, dep_id, tok)
		except urllib.error.URLError as err:
			print(f"poll retry {err.reason}")
			time.sleep(1)
			continue
		state = row.get("readyState") or row.get("status") or ""
		if state != last:
			print(f"state {state}")
			last = state
		if state == "READY":
			print("deploy ready")
			return row
		if state in {"ERROR", "CANCELED"}:
			msg = row.get("errorMessage") or "deploy failed"
			raise RuntimeError(msg)
		time.sleep(1)


def full_url(host):
	if host.startswith("http://") or host.startswith("https://"):
		return host
	return f"https://{host}"


def parse_args(root):
	parser = argparse.ArgumentParser()
	model = latest_model(root)
	parser.add_argument("model", nargs="?", default=str(model))
	return parser.parse_args()


def main():
	root = pathlib.Path(__file__).resolve().parent
	args = parse_args(root)
	cfg = load_deploy(root)
	tok = load_token(root)
	model = pathlib.Path(args.model).resolve()
	print(f"root {root}")
	print(f"api {cfg['api']}")
	print(f"model {model}")
	if not model.exists():
		raise FileNotFoundError(str(model))
	project = ensure_project(cfg, tok)
	run = model.parent.parent.name
	print(f"run {run}")
	tmp, out = build_dir(root, model)
	with tmp:
		dep = create_deploy(cfg, tok, out)
		print(f"deploy id {dep['id']}")
		dep = wait_ready(cfg, dep["id"], tok)
	url = dep.get("aliasFinal") or dep.get("url")
	url = full_url(url)
	print(json.dumps({
		"deploy_id": dep["id"],
		"model": str(model),
		"model_bytes": model.stat().st_size,
		"project_id": project["id"],
		"project_name": project["name"],
		"run": run,
		"url": url,
	}, indent=2))


if __name__ == "__main__":
	main()
