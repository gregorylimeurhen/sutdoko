import contextlib, dataclasses, datetime, importlib, json, pathlib, random, sys, tempfile, tomllib, zipfile
import torch

DATA_FILES = ["dev.tsv", "edges.tsv", "neighbors.json", "n2a.tsv", "test.tsv", "train.tsv"]
OPTIONAL_DATA_FILES = {"dev.tsv"}
PAD_TOKEN = "<pad>"
PROJECT_FILES = ["config.toml", "preprocess.py", "requirements.txt", "test.py", "train.py"]
SEP_TOKEN = "<sep>"
EOS_TOKEN = "<eos>"


@dataclasses.dataclass
class Tokenizer:
	vocab: list
	stoi: dict
	pad_id: int
	sep_id: int
	eos_id: int

	def encode_text(self, text):
		return [self.stoi[char] for char in text]

	def decode_text(self, token_ids):
		return "".join(self.vocab[token_id] for token_id in token_ids if token_id > self.eos_id)

	def to_dict(self):
		return {
			"vocab": self.vocab,
			"pad_id": self.pad_id,
			"sep_id": self.sep_id,
			"eos_id": self.eos_id,
		}

	@staticmethod
	def from_dict(data):
		vocab = list(data["vocab"])
		return Tokenizer(vocab, {token: index for index, token in enumerate(vocab)}, int(data["pad_id"]), int(data["sep_id"]), int(data["eos_id"]))


def normalize(text):
	return text.strip().lower()


def device_for():
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(root, section):
	return tomllib.loads((pathlib.Path(root) / "config.toml").read_text())[section]


def ensure_run_dir(root, prefix):
	run_dir = pathlib.Path(root) / "runs" / (prefix + datetime.datetime.now().strftime("%M%S"))
	run_dir.mkdir(parents=True, exist_ok=True)
	return run_dir


def load_package(root, name):
	root = str(pathlib.Path(root))
	for module_name in list(sys.modules):
		if module_name == name or module_name.startswith(name + "."):
			del sys.modules[module_name]
	sys.path[:] = [path for path in sys.path if path != root]
	sys.path.insert(0, root)
	return importlib.import_module(name)


def write_snapshot(path, source_root, project_root=None, data_overrides=None):
	data_overrides = {} if data_overrides is None else {name: pathlib.Path(value) for name, value in data_overrides.items()}
	path = pathlib.Path(path)
	project_root = pathlib.Path(source_root if project_root is None else project_root)
	source_root = pathlib.Path(source_root)
	with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
		for name in DATA_FILES:
			source = data_overrides.get(name, source_root / "data" / name)
			if source.exists():
				archive.write(source, f"data/{name}")
				continue
			if name in OPTIONAL_DATA_FILES:
				continue
			raise FileNotFoundError(source)
		for source in sorted((source_root / "src").rglob("*")):
			if source.is_file():
				archive.write(source, source.relative_to(source_root))
		for name in PROJECT_FILES:
			archive.write(project_root / name, name)


@contextlib.contextmanager
def extracted_snapshot(path):
	with tempfile.TemporaryDirectory() as temp_dir:
		with zipfile.ZipFile(path) as archive:
			archive.extractall(temp_dir)
		yield pathlib.Path(temp_dir)


@contextlib.contextmanager
def loaded_snapshot(path, name):
	with extracted_snapshot(path) as root:
		yield root, load_package(root, name)


def set_seed(seed):
	random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def load_neighbors(root):
	return json.loads((pathlib.Path(root) / "data" / "neighbors.json").read_text())


def load_tsv(path):
	return [tuple(line.split("\t")) for line in pathlib.Path(path).read_text().splitlines() if line]


def load_edges(root):
	return [(normalize(room), address.strip()) for room, address in load_tsv(pathlib.Path(root) / "data" / "edges.tsv")]


def load_pairs(path):
	return [{"input": normalize(left), "gold": normalize(right)} for left, right in load_tsv(path)]


def load_split_rows(root, name):
	return load_pairs(pathlib.Path(root) / "data" / f"{name}.tsv")


def load_training_rows(root):
	return load_split_rows(root, "train")


def load_dev_rows(root):
	path = pathlib.Path(root) / "data" / "dev.tsv"
	return [] if not path.exists() else load_pairs(path)


def load_test_rows(root):
	return load_split_rows(root, "test")


def load_room_lookup(root):
	return {normalize(room): address.strip() for room, address in load_tsv(pathlib.Path(root) / "data" / "n2a.tsv")}


def load_rooms(root):
	return sorted(load_room_lookup(root))


def build_room_trie(rooms, tokenizer):
	root = {}
	for room in sorted(rooms):
		node = root
		for token_id in tokenizer.encode_text(room):
			node = node.setdefault(token_id, {})
		node[tokenizer.eos_id] = {}
	return root


def build_tokenizer(root):
	chars = set()
	for room in load_rooms(root):
		chars.update(room)
	for key, values in load_neighbors(root).items():
		chars.update(key)
		for value in values:
			chars.update(value)
	vocab = [PAD_TOKEN, SEP_TOKEN, EOS_TOKEN] + sorted(chars)
	return Tokenizer(vocab, {token: index for index, token in enumerate(vocab)}, 0, 1, 2)


def rows_block_size(rows):
	return max(len(row["input"]) + len(row["gold"]) + 1 for row in rows)


def encode(prompt_ids, output_text, tokenizer):
	target = tokenizer.encode_text(normalize(output_text))
	tokens = list(prompt_ids) + [tokenizer.sep_id] + target + [tokenizer.eos_id]
	labels = list(tokens[1:])
	if prompt_ids:
		labels[:len(prompt_ids)] = [-100] * len(prompt_ids)
	return {"input_ids": tokens[:-1], "labels": labels}


def encode_pair(input_text, output_text, tokenizer):
	return encode(tokenizer.encode_text(normalize(input_text)), output_text, tokenizer)


def show_progress(label, current, total, width=20):
	total = max(1, total)
	current = min(max(0, current), total)
	filled = current * width // total
	sys.stdout.write(f"\r{label} [{'#' * filled}{'-' * (width - filled)}] {current}/{total}")
	sys.stdout.flush()


def end_progress():
	sys.stdout.write("\n")
	sys.stdout.flush()


def collate_examples(examples, tokenizer, device):
	sequence_len = max(len(example["input_ids"]) for example in examples)
	input_ids = torch.full((len(examples), sequence_len), tokenizer.pad_id, dtype=torch.long, device=device)
	labels = torch.full((len(examples), sequence_len), -100, dtype=torch.long, device=device)
	for row_index, example in enumerate(examples):
		length = len(example["input_ids"])
		input_ids[row_index, :length] = torch.tensor(example["input_ids"], dtype=torch.long, device=device)
		labels[row_index, :length] = torch.tensor(example["labels"], dtype=torch.long, device=device)
	return input_ids, labels
