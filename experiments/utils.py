import contextlib
import datetime
import importlib.util
import json
import math
import pathlib
import random
import sys
import tempfile
import time
import tomllib
import torch
import torch.nn as nn
import torch.nn.functional as F
import zipfile


PAD_TOKEN = "<pad>"
SEP_TOKEN = "<sep>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
BASELINE_NAMES = [
	"identity",
	# "levenshtein",
	"longest_common_prefix_length",
	"longest_common_substring_length",
	"whitespace_segment_histogram_intersection",
	"character_histogram_intersection",
	"damerau_levenshtein",
	"ours"
]


class Tokenizer:
	def __init__(self, vocab, stoi, pad_id, sep_id, eos_id, unk_id):
		self.vocab = vocab
		self.stoi = stoi
		self.pad_id = pad_id
		self.sep_id = sep_id
		self.eos_id = eos_id
		self.unk_id = unk_id

	def encode_text(self, text):
		unk_id = self.unk_id
		return [self.stoi.get(char, unk_id) for char in text]

	def decode_text(self, token_ids):
		unk_id = self.unk_id
		chars = [self.vocab[token_id] for token_id in token_ids if token_id > unk_id]
		return "".join(chars)

	def to_dict(self):
		return {
			"vocab": self.vocab,
			"pad_id": self.pad_id,
			"sep_id": self.sep_id,
			"eos_id": self.eos_id,
			"unk_id": self.unk_id,
		}

	@staticmethod
	def from_dict(data):
		vocab = list(data["vocab"])
		stoi = {token: index for index, token in enumerate(vocab)}
		pad_id = int(data["pad_id"])
		sep_id = int(data["sep_id"])
		eos_id = int(data["eos_id"])
		unk_id = int(data["unk_id"])
		return Tokenizer(vocab, stoi, pad_id, sep_id, eos_id, unk_id)


class ModelConfig:
	def __init__(self, depth, sequence_len, vocab_size, n_head, n_embd):
		self.depth = depth
		self.sequence_len = sequence_len
		self.vocab_size = vocab_size
		self.n_head = n_head
		self.n_embd = n_embd


class RMSNorm(nn.Module):
	def __init__(self, dim):
		super().__init__()
		self.weight = nn.Parameter(torch.ones(dim))

	def forward(self, x):
		return F.rms_norm(x, (x.size(-1),), self.weight)


def apply_rotary(x, cos, sin):
	left = x[..., ::2]
	right = x[..., 1::2]
	left_out = left * cos - right * sin
	right_out = left * sin + right * cos
	pairs = torch.stack((left_out, right_out), dim=-1)
	return pairs.flatten(-2)


class RotaryEmbedding(nn.Module):
	def __init__(self, sequence_len, head_dim):
		super().__init__()
		steps = torch.arange(sequence_len, dtype=torch.float32)
		steps2 = torch.arange(0, head_dim, 2, dtype=torch.float32)
		rates = 10000 ** (-steps2 / head_dim)
		angles = torch.outer(steps, rates)
		cos = angles.cos().unsqueeze(0).unsqueeze(0)
		sin = angles.sin().unsqueeze(0).unsqueeze(0)
		self.register_buffer("cos", cos, persistent=False)
		self.register_buffer("sin", sin, persistent=False)

	def forward(self, q, k, start=0):
		stop = start + q.size(2)
		cos = self.cos[:, :, start:stop]
		sin = self.sin[:, :, start:stop]
		return apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)


class SelfAttention(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.n_head = config.n_head
		self.head_dim = config.n_embd // config.n_head
		self.qkv = nn.Linear(config.n_embd, config.n_embd * 3, bias=False)
		self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
		self.rotary = RotaryEmbedding(config.sequence_len, self.head_dim)

	def qkv_heads(self, x):
		batch_size, sequence_len, _ = x.shape
		q, k, v = self.qkv(x).chunk(3, dim=-1)
		shape = batch_size, sequence_len, self.n_head, self.head_dim
		q = q.view(shape).transpose(1, 2)
		k = k.view(shape).transpose(1, 2)
		v = v.view(shape).transpose(1, 2)
		return batch_size, sequence_len, q, k, v

	def forward(self, x):
		batch_size, sequence_len, q, k, v = self.qkv_heads(x)
		q, k = self.rotary(q, k)
		q = F.rms_norm(q, (self.head_dim,))
		k = F.rms_norm(k, (self.head_dim,))
		y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
		y = y.transpose(1, 2).contiguous().view(batch_size, sequence_len, -1)
		return self.proj(y)

	def forward_cached(self, x, cache=None):
		batch_size, sequence_len, q, k, v = self.qkv_heads(x)
		start = 0 if cache is None else cache[0].size(2)
		q, k = self.rotary(q, k, start)
		q = F.rms_norm(q, (self.head_dim,))
		k = F.rms_norm(k, (self.head_dim,))
		if cache is None:
			full_k = k
			full_v = v
			y = F.scaled_dot_product_attention(q, full_k, full_v, is_causal=True)
		else:
			if sequence_len != 1:
				raise ValueError("cached decoding expects one token at a time")
			full_k = torch.cat((cache[0], k), dim=2)
			full_v = torch.cat((cache[1], v), dim=2)
			y = F.scaled_dot_product_attention(q, full_k, full_v)
		y = y.transpose(1, 2).contiguous().view(batch_size, sequence_len, -1)
		return self.proj(y), (full_k, full_v)


class MLP(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.fc = nn.Linear(config.n_embd, config.n_embd * 4, bias=False)
		self.proj = nn.Linear(config.n_embd * 4, config.n_embd, bias=False)

	def forward(self, x):
		return self.proj(F.relu(self.fc(x)).square())


class Block(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.attn_norm = RMSNorm(config.n_embd)
		self.attn = SelfAttention(config)
		self.mlp_norm = RMSNorm(config.n_embd)
		self.mlp = MLP(config)

	def forward(self, x):
		x = x + self.attn(self.attn_norm(x))
		return x + self.mlp(self.mlp_norm(x))

	def forward_cached(self, x, cache=None):
		attn, cache = self.attn.forward_cached(self.attn_norm(x), cache)
		x = x + attn
		return x + self.mlp(self.mlp_norm(x)), cache


class GPT(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config
		self.wte = nn.Embedding(config.vocab_size, config.n_embd)
		self.blocks = nn.ModuleList(Block(config) for _ in range(config.depth))
		self.norm = RMSNorm(config.n_embd)
		self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
		self.reset_parameters()

	def reset_parameters(self):
		nn.init.normal_(self.wte.weight, mean=0.0, std=0.02)
		nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
		for module in self.modules():
			if isinstance(module, nn.Linear) and module is not self.lm_head:
				nn.init.xavier_uniform_(module.weight)

	def forward(self, input_ids, labels=None):
		x = self.wte(input_ids)
		for block in self.blocks:
			x = block(x)
		logits = self.lm_head(self.norm(x))
		if labels is None:
			return logits
		logits = logits.reshape(-1, logits.size(-1))
		labels = labels.reshape(-1)
		return F.cross_entropy(logits, labels, ignore_index=-100)

	def forward_cached(self, input_ids, cache=None):
		if cache is not None and len(cache) != len(self.blocks):
			raise ValueError("cache depth mismatch")
		x = self.wte(input_ids)
		next_cache = []
		for index, block in enumerate(self.blocks):
			block_cache = None if cache is None else cache[index]
			x, block_cache = block.forward_cached(x, block_cache)
			next_cache.append(block_cache)
		return self.lm_head(self.norm(x)), next_cache


class Rng:
	def __init__(self, seed):
		state = int(seed) & 4294967295
		self.state = state or 1

	def next_u32(self):
		mask = 4294967295
		x = self.state
		x ^= (x << 13) & mask
		x ^= x >> 17
		x ^= (x << 5) & mask
		x &= mask
		self.state = x
		return x

	def random(self):
		return self.next_u32() / 4294967296.0

	def randrange(self, stop):
		if stop <= 0:
			raise ValueError("stop must be positive")
		return int(self.random() * stop)

	def shuffle(self, xs):
		for index in range(len(xs) - 1, 0, -1):
			swap = self.randrange(index + 1)
			xs[index], xs[swap] = xs[swap], xs[index]

	def sample(self, xs, count):
		if not 0 <= count <= len(xs):
			raise ValueError("sample count out of range")
		xs = list(xs)
		self.shuffle(xs)
		return xs[:count]


def normalize(text):
	return text.strip().lower()


def device_for():
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_toml(root):
	path = pathlib.Path(root) / "config.toml"
	return tomllib.loads(path.read_text())


def load_config(root, section):
	return load_toml(root)[section]


def load_seed(root):
	return int(load_toml(root)["seed"])


def ensure_run_dir(root, name):
	stamp = datetime.datetime.now().strftime("%M%S")
	run_dir = pathlib.Path(root) / "runs" / stamp / str(name)
	run_dir.mkdir(parents=True, exist_ok=True)
	return run_dir


def load_module(root):
	path = pathlib.Path(root) / "utils.py"
	spec = importlib.util.spec_from_file_location("_snapshot_utils", path)
	module = importlib.util.module_from_spec(spec)
	sys.modules.pop(spec.name, None)
	sys.modules[spec.name] = module
	spec.loader.exec_module(module)
	return module


def write_snapshot(path, source_root):
	path = pathlib.Path(path)
	source_root = pathlib.Path(source_root)
	with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
		for source in sorted(source_root.rglob("*")):
			relative = source.relative_to(source_root)
			parts = relative.parts
			if parts and parts[0] in {"app", "runs"}:
				continue
			archive.write(source, relative)


@contextlib.contextmanager
def extracted_snapshot(path):
	with tempfile.TemporaryDirectory() as temp_dir:
		with zipfile.ZipFile(path) as archive:
			archive.extractall(temp_dir)
		yield pathlib.Path(temp_dir)


@contextlib.contextmanager
def loaded_snapshot(path):
	with extracted_snapshot(path) as root:
		yield root, load_module(root)


def set_seed(seed):
	random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


def load_tsv(path):
	path = pathlib.Path(path)
	rows = path.read_text().splitlines()
	return [tuple(row.split("\t")) for row in rows if row]


def load_boundaries(root):
	path = pathlib.Path(root) / "data" / "boundaries.txt"
	lines = path.read_text().splitlines()
	return {line.rstrip("\n") for line in lines}


def load_edges(root):
	path = pathlib.Path(root) / "data" / "edges.tsv"
	rows = []
	for room, address in load_tsv(path):
		rows.append((normalize(room), address.strip()))
	return rows


def load_neighbors(root):
	path = pathlib.Path(root) / "data" / "neighbors.json"
	return json.loads(path.read_text())


def load_pairs(path):
	rows = []
	for left, right in load_tsv(path):
		rows.append({"input": normalize(left), "gold": normalize(right)})
	return rows


def load_rows(root, name):
	return load_pairs(pathlib.Path(root) / "data" / f"{name}.tsv")


def load_room_lookup(root):
	path = pathlib.Path(root) / "data" / "n2a.tsv"
	return {
		normalize(room): address.strip()
		for room, address in load_tsv(path)
	}


def load_aliases(root):
	path = pathlib.Path(root) / "data" / "aliases.tsv"
	rows = path.read_text(encoding="utf-8-sig").splitlines()
	pairs = []
	for index, row in enumerate(rows):
		if not row:
			continue
		if not index and row == "source\ttarget":
			continue
		source, target = row.split("\t")
		pairs.append((normalize(source), normalize(target)))
	return pairs


def build_room_trie(rooms, tokenizer):
	root = {"allowed": (), "children": {}}
	for room in sorted(rooms):
		node = root
		for token_id in tokenizer.encode_text(room):
			kids = node["children"]
			node = kids.setdefault(token_id, {"allowed": (), "children": {}})
		node["children"][tokenizer.eos_id] = {"allowed": (), "children": {}}
	stack = [root]
	while stack:
		node = stack.pop()
		kids = node["children"]
		key = lambda token_id: tokenizer.vocab[token_id]
		node["allowed"] = tuple(sorted(kids, key=key))
		stack.extend(node["children"].values())
	return root


def build_tokenizer(root):
	chars = set()
	room_lookup = load_room_lookup(root)
	for room in sorted(room_lookup):
		chars.update(room)
	neighbors = load_neighbors(root)
	for key, values in neighbors.items():
		chars.add(key)
		chars.update(values)
	vocab = [PAD_TOKEN, SEP_TOKEN, EOS_TOKEN, UNK_TOKEN]
	vocab.extend(sorted(chars))
	stoi = {token: index for index, token in enumerate(vocab)}
	return Tokenizer(vocab, stoi, 0, 1, 2, 3)


def rows_block_size(rows):
	sizes = (len(row["input"]) + len(row["gold"]) + 1 for row in rows)
	return max(sizes, default=1)


def encode(prompt_ids, output_text, tokenizer):
	target_ids = tokenizer.encode_text(normalize(output_text))
	input_ids = list(prompt_ids)
	tokens = input_ids + [tokenizer.sep_id] + target_ids + [tokenizer.eos_id]
	labels = list(tokens[1:])
	if input_ids:
		labels[:len(input_ids)] = [-100] * len(input_ids)
	return {"input_ids": tokens[:-1], "labels": labels}


def show_progress(label, current, total, width=20):
	total = max(1, total)
	current = min(max(0, current), total)
	sys.stdout.write(f"\r{label} {current}/{total}")
	sys.stdout.flush()


def end_progress():
	sys.stdout.write("\n")
	sys.stdout.flush()


def collate_examples(examples, tok, dev):
	sequence_len = max(len(example["input_ids"]) for example in examples)
	shape = len(examples), sequence_len
	kw = {"dtype": torch.long, "device": dev}
	input_ids = torch.full(shape, tok.pad_id, **kw)
	labels = torch.full(shape, -100, **kw)
	for row_index, example in enumerate(examples):
		length = len(example["input_ids"])
		input_ids[row_index, :length] = torch.tensor(example["input_ids"], **kw)
		labels[row_index, :length] = torch.tensor(example["labels"], **kw)
	return input_ids, labels


def build_config(depth, tokenizer, sequence_len):
	head_dim = 128
	n_embd = max(head_dim, math.ceil(depth * 64 / head_dim) * head_dim)
	vocab_size = len(tokenizer.vocab)
	n_head = n_embd // head_dim
	return ModelConfig(depth, sequence_len, vocab_size, n_head, n_embd)


def build_model(depth, tokenizer, sequence_len):
	return GPT(build_config(depth, tokenizer, sequence_len))


def save_checkpoint(path, model, tokenizer, rooms):
	config = {
		"depth": model.config.depth,
		"sequence_len": model.config.sequence_len,
		"vocab_size": model.config.vocab_size,
		"n_head": model.config.n_head,
		"n_embd": model.config.n_embd,
	}
	state = {
		"config": config,
		"model": model.state_dict(),
		"rooms": list(rooms),
		"tokenizer": tokenizer.to_dict(),
	}
	torch.save(state, path)


def load_checkpoint(path, device):
	state = torch.load(path, map_location=device)
	tokenizer = Tokenizer.from_dict(state["tokenizer"])
	config = ModelConfig(**state["config"])
	model = GPT(config).to(device)
	model.load_state_dict(state["model"])
	model.eval()
	return model, tokenizer, list(state["rooms"])


def is_cuda_oom(error, device):
	return device.type == "cuda" and "out of memory" in str(error).lower()


def can_fit_batch(model, examples, tokenizer, device):
	input_ids = None
	labels = None
	loss = None
	try:
		model.zero_grad(set_to_none=True)
		input_ids, labels = collate_examples(examples, tokenizer, device)
		loss = model(input_ids, labels)
		loss.backward()
		return True
	except RuntimeError as error:
		if not is_cuda_oom(error, device):
			raise
		return False
	finally:
		model.zero_grad(set_to_none=True)
		if device.type == "cuda":
			torch.cuda.empty_cache()
		del input_ids, labels, loss


def largest_batch_size(model, examples, tokenizer, device):
	if device.type != "cuda":
		best = min(32, len(examples))
		print(f"batch_size {best}")
		return best
	key = lambda example: len(example["input_ids"])
	longest = sorted(examples, key=key, reverse=True)
	best = 1
	candidate = 1
	while candidate <= len(longest):
		if not can_fit_batch(model, longest[:candidate], tokenizer, device):
			break
		best = candidate
		candidate *= 2
	if candidate > len(longest):
		print(f"batch_size {best}")
		return best
	low = best + 1
	high = min(candidate - 1, len(longest))
	while low <= high:
		middle = (low + high) // 2
		if can_fit_batch(model, longest[:middle], tokenizer, device):
			best = middle
			low = middle + 1
			continue
		high = middle - 1
	print(f"batch_size {best}")
	return best


def loss_tokens(labels):
	return int(labels.ne(-100).sum().item())


def grad_norm(model):
	total = 0.0
	for param in model.parameters():
		grad = param.grad
		if grad is None:
			continue
		total += float(grad.detach().pow(2).sum().item())
	return total ** 0.5


def train_epoch(model, examples, tok, dev, opt, batch, epoch, seed):
	order = list(range(len(examples)))
	Rng(seed + epoch).shuffle(order)
	model.train()
	total_loss = 0.0
	total_grad_norm = 0.0
	total_tokens = 0
	total_steps = 0
	show_progress(f"train {epoch}", 0, len(order))
	start = 0
	while start < len(order):
		rows = [examples[index] for index in order[start:start + batch]]
		input_ids = None
		labels = None
		loss = None
		loss_value = None
		token_count = None
		try:
			input_ids, labels = collate_examples(rows, tok, dev)
			opt.zero_grad(set_to_none=True)
			loss = model(input_ids, labels)
			loss.backward()
			grad_value = grad_norm(model)
			opt.step()
			loss_value = float(loss.item())
			token_count = loss_tokens(labels)
		except RuntimeError as error:
			if not is_cuda_oom(error, dev) or batch == 1:
				raise
			opt.zero_grad(set_to_none=True)
			model.zero_grad(set_to_none=True)
			torch.cuda.empty_cache()
			batch //= 2
			print(f"batch_size {batch}")
			continue
		finally:
			del input_ids, labels, loss
		total_grad_norm += grad_value
		total_loss += loss_value * token_count
		total_tokens += token_count
		total_steps += 1
		start += len(rows)
		show_progress(f"train {epoch}", start, len(order))
	return total_loss / total_tokens, batch, total_grad_norm / total_steps


def val_loss(model, examples, tok, dev, batch):
	model.eval()
	total_loss = 0.0
	total_tokens = 0
	start = 0
	while start < len(examples):
		rows = examples[start:start + batch]
		input_ids = None
		labels = None
		loss = None
		loss_value = None
		token_count = None
		try:
			with torch.inference_mode():
				input_ids, labels = collate_examples(rows, tok, dev)
				loss = model(input_ids, labels)
			loss_value = float(loss.item())
			token_count = loss_tokens(labels)
		except RuntimeError as error:
			if not is_cuda_oom(error, dev) or batch == 1:
				raise
			torch.cuda.empty_cache()
			batch //= 2
			print(f"batch_size {batch}")
			continue
		finally:
			del input_ids, labels, loss
		total_loss += loss_value * token_count
		total_tokens += token_count
		start += len(rows)
	return total_loss / total_tokens, batch


def train(model, tr, va, tok, dev, path, tol, run, seed, batch=None):
	train_xs = []
	for row in tr:
		prompt = tok.encode_text(row["input"])
		train_xs.append(encode(prompt, row["gold"], tok))
	val_xs = []
	for row in va:
		prompt = tok.encode_text(row["input"])
		val_xs.append(encode(prompt, row["gold"], tok))
	if batch is None:
		batch = largest_batch_size(model, train_xs, tok, dev)
	else:
		print(f"batch_size {batch}")
	opt = torch.optim.AdamW(model.parameters())
	rooms = sorted({row["gold"] for row in tr})
	path = pathlib.Path(path)
	latest = path.with_name("latest.pt")
	best_val = None
	best_epoch = 0
	epoch = 0
	step = train_epoch
	tx = train_xs
	vx = val_xs
	try:
		while True:
			epoch += 1
			loss, batch, norm = step(model, tx, tok, dev, opt, batch, epoch, seed)
			current_val_loss, batch = val_loss(model, vx, tok, dev, batch)
			save_checkpoint(latest, model, tok, rooms)
			if best_val is None or current_val_loss < best_val:
				best_val = current_val_loss
				best_epoch = epoch
				save_checkpoint(path, model, tok, rooms)
			run.log({
				"grad_norm": norm,
				"train_loss": loss,
				"val_loss": current_val_loss,
			})
			if epoch - best_epoch >= tol:
				break
	except KeyboardInterrupt:
		end_progress()
		raise
	end_progress()


def predict_room(model, tok, dev, text, trie, rng):
	prefix = tok.encode_text(normalize(text)) + [tok.sep_id]
	room = []
	node = trie
	device_key = str(dev)
	allowed_tensors = {}
	model.eval()
	with torch.inference_mode():
		kw = {"device": dev, "dtype": torch.long}
		logits, cache = model.forward_cached(torch.tensor([prefix], **kw))
		while True:
			allowed = node["allowed"]
			key = (id(node), device_key)
			if key not in allowed_tensors:
				allowed_tensors[key] = torch.tensor(allowed, **kw)
			allowed_logits = logits[0, -1].index_select(0, allowed_tensors[key])
			mask = allowed_logits == allowed_logits.max()
			best_indices = mask.nonzero().flatten().tolist()
			token_id = allowed[best_indices[rng.randrange(len(best_indices))]]
			if token_id == tok.eos_id:
				return tok.decode_text(room)
			room.append(token_id)
			node = node["children"][token_id]
			next_ids = torch.tensor([[token_id]], **kw)
			logits, cache = model.forward_cached(next_ids, cache)


def levenshtein_distance(left, right, max_distance=None):
	if max_distance is not None and abs(len(left) - len(right)) > max_distance:
		return max_distance + 1
	previous = list(range(len(right) + 1))
	for left_index, left_char in enumerate(left, start=1):
		current = [left_index]
		for right_index, right_char in enumerate(right, start=1):
			insert = current[-1] + 1
			delete = previous[right_index] + 1
			replace = previous[right_index - 1] + (left_char != right_char)
			value = min(insert, delete, replace)
			current.append(value)
		previous = current
	distance = previous[-1]
	if max_distance is not None and distance > max_distance:
		return max_distance + 1
	return distance


def damerau_levenshtein_distance(left, right, max_distance=None):
	if max_distance is not None and abs(len(left) - len(right)) > max_distance:
		return max_distance + 1
	limit = len(left) + len(right)
	last_seen = {}
	table = [[limit] * (len(right) + 2) for _ in range(len(left) + 2)]
	table[0][0] = limit
	for left_index in range(len(left) + 1):
		table[left_index + 1][0] = limit
		table[left_index + 1][1] = left_index
	for right_index in range(len(right) + 1):
		table[0][right_index + 1] = limit
		table[1][right_index + 1] = right_index
	for left_index, left_char in enumerate(left, start=1):
		last_match = 0
		for right_index, right_char in enumerate(right, start=1):
			match_index = last_seen.get(right_char, 0)
			cost = int(left_char != right_char)
			if not cost:
				last_match = right_index
			a = table[left_index][right_index] + cost
			b = table[left_index + 1][right_index] + 1
			c = table[left_index][right_index + 1] + 1
			d = table[match_index][last_match]
			d += left_index - match_index + right_index - last_match - 1
			value = min(a, b, c, d)
			table[left_index + 1][right_index + 1] = value
		last_seen[left_char] = left_index
	distance = table[-1][-1]
	if max_distance is not None and distance > max_distance:
		return max_distance + 1
	return distance


def longest_common_prefix_length(left, right, min_score=None):
	score = 0
	limit = min(len(left), len(right))
	if min_score is not None and limit < min_score:
		return -1
	while score < limit and left[score] == right[score]:
		score += 1
	return score


def longest_common_substring_length(left, right, min_score=None):
	if len(left) < len(right):
		left, right = right, left
	if min_score is not None and len(right) < min_score:
		return -1
	best = 0
	previous = [0] * (len(right) + 1)
	for left_char in left:
		current = [0]
		for right_index, right_char in enumerate(right, start=1):
			value = previous[right_index - 1] + 1
			if left_char != right_char:
				value = 0
			if value > best:
				best = value
			current.append(value)
		previous = current
	return best


def lcs_length(left, right, min_score=None):
	if len(left) < len(right):
		left, right = right, left
	if min_score is not None and len(right) < min_score:
		return -1
	previous = [0] * (len(right) + 1)
	for left_char in left:
		current = [0]
		for right_index, right_char in enumerate(right, start=1):
			value = previous[right_index]
			if current[-1] > value:
				value = current[-1]
			if left_char == right_char:
				match = previous[right_index - 1] + 1
				if match > value:
					value = match
			current.append(value)
		previous = current
	return previous[-1]


def seg_hist(text):
	hist = {}
	for seg in text.split():
		hist[seg] = hist.get(seg, 0) + 1
	return hist


def char_hist(text):
	hist = {}
	for char in text:
		hist[char] = hist.get(char, 0) + 1
	return hist


def hist_score(left, right, min_score=None):
	score = 0
	for char, count in left.items():
		score += min(count, right.get(char, 0))
	if min_score is not None and score < min_score:
		return -1
	return score


def nearest_room(text, rooms, rng, distance_fn):
	best_distance = None
	best_rooms = []
	for room in rooms:
		current_distance = distance_fn(text, room, best_distance)
		if best_distance is None or current_distance < best_distance:
			best_distance = current_distance
			best_rooms = [room]
			continue
		if current_distance == best_distance:
			best_rooms.append(room)
	return best_rooms[rng.randrange(len(best_rooms))]


def best_room(text, rooms, rng, score_fn):
	best_score = None
	best_rooms = []
	for room in rooms:
		current_score = score_fn(text, room, best_score)
		if best_score is None or current_score > best_score:
			best_score = current_score
			best_rooms = [room]
			continue
		if current_score == best_score:
			best_rooms.append(room)
	return best_rooms[rng.randrange(len(best_rooms))]


def nearest_room_address(text, room_lookup, rooms, rng, distance_fn):
	return room_lookup[nearest_room(text, rooms, rng, distance_fn)]


def best_room_address(text, room_lookup, rooms, rng, score_fn):
	return room_lookup[best_room(text, rooms, rng, score_fn)]


def levenshtein_address(text, room_lookup, rooms, rng):
	fn = levenshtein_distance
	return nearest_room_address(text, room_lookup, rooms, rng, fn)


def damerau_levenshtein_address(text, room_lookup, rooms, rng):
	fn = damerau_levenshtein_distance
	return nearest_room_address(text, room_lookup, rooms, rng, fn)


def longest_common_prefix_address(text, room_lookup, rooms, rng):
	fn = longest_common_prefix_length
	return best_room_address(text, room_lookup, rooms, rng, fn)


def longest_common_substring_address(text, room_lookup, rooms, rng):
	fn = longest_common_substring_length
	return best_room_address(text, room_lookup, rooms, rng, fn)


def lcs_address(text, room_lookup, rooms, rng):
	fn = lcs_length
	return best_room_address(text, room_lookup, rooms, rng, fn)


def hist_address(text, room_lookup, room_hists, rng):
	left = char_hist(text)
	best_score = None
	best_rooms = []
	for room, right in room_hists:
		score = hist_score(left, right, best_score)
		if best_score is None or score > best_score:
			best_score = score
			best_rooms = [room]
			continue
		if score == best_score:
			best_rooms.append(room)
	room = best_rooms[rng.randrange(len(best_rooms))]
	return room_lookup[room]


def hist_room(text, room_hists, rng):
	left = char_hist(text)
	best_score = None
	best_rooms = []
	for room, right in room_hists:
		score = hist_score(left, right, best_score)
		if best_score is None or score > best_score:
			best_score = score
			best_rooms = [room]
			continue
		if score == best_score:
			best_rooms.append(room)
	return best_rooms[rng.randrange(len(best_rooms))]


def seg_room(text, room_segs, rng):
	left = seg_hist(text)
	best_score = None
	best_rooms = []
	for room, right in room_segs:
		score = hist_score(left, right, best_score)
		if best_score is None or score > best_score:
			best_score = score
			best_rooms = [room]
			continue
		if score == best_score:
			best_rooms.append(room)
	return best_rooms[rng.randrange(len(best_rooms))]


def evaluate_rows_into(model, rows, tok, dev, rm, rooms, write, seed):
	room_set = set(rooms)
	trie = build_room_trie(rooms, tok)
	# lev_rng = Rng(seed)
	pre_rng = Rng(seed)
	sub_rng = Rng(seed)
	seg_rng = Rng(seed)
	hist_rng = Rng(seed)
	dam_rng = Rng(seed)
	ours_rng = Rng(seed)
	damf = damerau_levenshtein_distance
	pref = longest_common_prefix_length
	subf = longest_common_substring_length
	room_segs = [(room, seg_hist(room)) for room in rooms]
	room_hists = [(room, char_hist(room)) for room in rooms]
	preds = {}
	preds["identity"] = lambda text: text if text in room_set else ""
	# lev = lambda text: nearest_room(text, rooms, lev_rng, levenshtein_distance)
	pre = lambda text: best_room(text, rooms, pre_rng, pref)
	sub = lambda text: best_room(text, rooms, sub_rng, subf)
	seg = lambda text: seg_room(text, room_segs, seg_rng)
	hist = lambda text: hist_room(text, room_hists, hist_rng)
	dam = lambda text: nearest_room(text, rooms, dam_rng, damf)
	pick = lambda text: predict_room(model, tok, dev, text, trie, ours_rng)
	# preds["levenshtein"] = lambda text: lev(text)
	name = "longest_common_prefix_length"
	preds[name] = lambda text: pre(text)
	name = "longest_common_substring_length"
	preds[name] = lambda text: sub(text)
	name = "whitespace_segment_histogram_intersection"
	preds[name] = lambda text: seg(text)
	name = "character_histogram_intersection"
	preds[name] = lambda text: hist(text)
	name = "damerau_levenshtein"
	preds[name] = lambda text: dam(text)
	preds["ours"] = lambda text: text if text in room_set else pick(text)
	stats = {name: {"correct": 0, "latency": 0.0} for name in BASELINE_NAMES}
	show_progress("test", 0, len(rows))
	for row_index, row in enumerate(rows, start=1):
		text = row["input"]
		gold_address = rm[row["gold"]]
		detail = {"input": text, "gold_room": row["gold"], "gold": gold_address}
		for name in BASELINE_NAMES:
			start = time.perf_counter()
			pred_room = preds[name](text)
			stats[name]["latency"] += time.perf_counter() - start
			prediction = rm.get(pred_room, "")
			stats[name]["correct"] += int(prediction == gold_address)
			detail[name] = prediction
		write(detail)
		show_progress("test", row_index, len(rows))
	total = len(rows)
	end_progress()
	scores = {}
	for name, values in stats.items():
		scores[name] = {
			"accuracy": values["correct"] / total,
			"mean_latency": values["latency"] / total,
		}
	return scores
