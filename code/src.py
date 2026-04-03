import csv
import io
import math
import random
import re
import time
import torch
import torch.nn.functional as F

from collections import defaultdict, deque
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


AMERICAN_MAP = {
	"behavioural": "behavioral",
	"centre": "center",
	"characterisation": "characterization",
	"organisation": "organization",
	"theatre": "theater",
}

DROP_PLURAL_MAP = {
	"admissions": "admission",
	"arts": "art",
	"brothers": "brother",
	"cities": "city",
	"communications": "communication",
	"devices": "device",
	"facilities": "facility",
	"laboratories": "laboratory",
	"materials": "material",
	"relations": "relation",
	"resources": "resource",
	"sciences": "science",
	"services": "service",
	"studies": "study",
	"systems": "system",
}


class NLS:
	name = "nls"

	def __call__(self, predictions, targets, durations=None):
		if not targets:
			return 0.0
		return sum(self.score(prediction, target) for prediction, target in zip(predictions, targets)) / len(targets)

	def score(self, prediction, target):
		if not prediction and not target:
			return 1.0
		return 1.0 - self.levenshtein(prediction, target) / max(len(prediction), len(target), 1)

	def levenshtein(self, left, right):
		if len(left) < len(right):
			left, right = right, left
		previous = list(range(len(right) + 1))
		for i, left_char in enumerate(left, start=1):
			current = [i]
			for j, right_char in enumerate(right, start=1):
				insert_cost = current[j - 1] + 1
				delete_cost = previous[j] + 1
				substitute_cost = previous[j - 1] + (left_char != right_char)
				current.append(min(insert_cost, delete_cost, substitute_cost))
			previous = current
		return previous[-1]


class TTLT:
	name = "ttlt"

	def __call__(self, predictions, targets, durations=None):
		if not durations:
			return 0.0
		return sum(durations) / len(durations)


class TextDataset(Dataset):
	def __init__(self, rows, tokenizer):
		self.rows = [torch.tensor(tokenizer.encode(row), dtype=torch.long) for row in rows if len(row) > 1]

	def __len__(self):
		return len(self.rows)

	def __getitem__(self, index):
		return self.rows[index]


class CharTokenizer:
	def __init__(self, rows):
		characters = sorted({character for row in rows for character in row})
		self.pad_token = "\0"
		self.itos = [self.pad_token] + characters
		self.stoi = {token: index for index, token in enumerate(self.itos)}

	def encode(self, text):
		return [self.stoi[character] for character in text]

	def decode(self, ids):
		return "".join(self.itos[index] for index in ids if index and index < len(self.itos))

	@property
	def pad_id(self):
		return 0

	@property
	def vocab_size(self):
		return len(self.itos)


class CausalSelfAttention(nn.Module):
	def __init__(self, width, heads, dropout):
		super().__init__()
		self.dropout = dropout
		self.heads = heads
		self.width = width
		self.qkv = nn.Linear(width, width * 3)
		self.proj = nn.Linear(width, width)
		self.resid_dropout = nn.Dropout(dropout)

	def forward(self, x):
		batch, steps, channels = x.shape
		head_width = channels // self.heads
		qkv = self.qkv(x)
		query, key, value = qkv.chunk(3, dim=-1)
		query = query.view(batch, steps, self.heads, head_width).transpose(1, 2)
		key = key.view(batch, steps, self.heads, head_width).transpose(1, 2)
		value = value.view(batch, steps, self.heads, head_width).transpose(1, 2)
		if hasattr(F, "scaled_dot_product_attention"):
			output = F.scaled_dot_product_attention(query, key, value, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
		else:
			weights = query @ key.transpose(-2, -1)
			weights = weights / math.sqrt(head_width)
			mask = torch.triu(torch.ones(steps, steps, device=x.device, dtype=torch.bool), diagonal=1)
			weights = weights.masked_fill(mask, float("-inf"))
			weights = F.softmax(weights, dim=-1)
			weights = F.dropout(weights, p=self.dropout, training=self.training)
			output = weights @ value
		output = output.transpose(1, 2).contiguous().view(batch, steps, channels)
		return self.resid_dropout(self.proj(output))


class MLP(nn.Module):
	def __init__(self, width, dropout):
		super().__init__()
		self.fc = nn.Linear(width, width * 4)
		self.proj = nn.Linear(width * 4, width)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
	def __init__(self, width, heads, dropout):
		super().__init__()
		self.attn = CausalSelfAttention(width, heads, dropout)
		self.ln_1 = nn.LayerNorm(width)
		self.ln_2 = nn.LayerNorm(width)
		self.mlp = MLP(width, dropout)

	def forward(self, x):
		x = x + self.attn(self.ln_1(x))
		x = x + self.mlp(self.ln_2(x))
		return x


class GPT2(nn.Module):
	def __init__(self, vocab_size, block_size, width=128, heads=4, layers=4, dropout=0.1):
		super().__init__()
		self.block_size = block_size
		self.token_embedding = nn.Embedding(vocab_size, width)
		self.position_embedding = nn.Embedding(block_size, width)
		self.dropout = nn.Dropout(dropout)
		self.blocks = nn.ModuleList(Block(width, heads, dropout) for _ in range(layers))
		self.ln_f = nn.LayerNorm(width)
		self.head = nn.Linear(width, vocab_size, bias=False)
		self.head.weight = self.token_embedding.weight
		self.apply(self._init_weights)

	def _init_weights(self, module):
		if isinstance(module, nn.Embedding):
			nn.init.normal_(module.weight, mean=0.0, std=0.02)
			return
		if isinstance(module, nn.Linear):
			nn.init.normal_(module.weight, mean=0.0, std=0.02)
			if module.bias is not None:
				nn.init.zeros_(module.bias)

	def forward(self, input_ids, targets=None):
		batch, steps = input_ids.shape
		if steps > self.block_size:
			input_ids = input_ids[:, -self.block_size:]
			if targets is not None:
				targets = targets[:, -self.block_size:]
			steps = input_ids.shape[1]
		positions = torch.arange(steps, device=input_ids.device)
		x = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
		x = self.dropout(x)
		for block in self.blocks:
			x = block(x)
		logits = self.head(self.ln_f(x))
		if targets is None:
			return logits, None
		loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)
		return logits, loss


def address_sort_key(address):
	parts = re.findall(r"\d+|[A-Za-z]+|[^A-Za-z\d]+", address)
	return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def collate_rows(batch, pad_id):
	max_length = max(item.numel() for item in batch) - 1
	inputs = torch.full((len(batch), max_length), pad_id, dtype=torch.long)
	targets = torch.full((len(batch), max_length), -100, dtype=torch.long)
	for index, item in enumerate(batch):
		input_ids = item[:-1]
		target_ids = item[1:]
		inputs[index, : input_ids.numel()] = input_ids
		targets[index, : target_ids.numel()] = target_ids
	return inputs, targets


def dataset_rows(rows, tokenizer, batch_size, shuffle):
	dataset = TextDataset(rows, tokenizer)
	return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda batch: collate_rows(batch, tokenizer.pad_id))


def extract_section(text, name):
	match = re.search(rf"`{re.escape(name)}`\n\n```[^\n]*\n(.*?)\n```", text, re.S)
	if not match:
		raise ValueError(f"Missing section {name}")
	return match.group(1).strip("\n")


def replace_section(text, name, language, rows):
	body = "\n".join(rows)
	pattern = re.compile(rf"(`{re.escape(name)}`\n\n```{re.escape(language)}\n)(.*?)(\n```)", re.S)
	match = pattern.search(text)
	if not match:
		raise ValueError(f"Missing section {name}")
	return text[: match.start()] + match.group(1) + body + match.group(3) + text[match.end() :]


def normalize_name(name):
	return re.sub(r"\s+", " ", name).strip()


def variant_pattern(source):
	if re.fullmatch(r"[0-9a-z ]+", source):
		return re.compile(rf"(?<![0-9a-z]){re.escape(source)}(?![0-9a-z])")
	return re.compile(re.escape(source))


def replace_variant(text, start, end, source, target):
	replacement = target
	if source and all(not character.isalnum() for character in source) and target and target[0].isalnum():
		if start and text[start - 1].isalnum() and not replacement.startswith(" "):
			replacement = " " + replacement
		if end < len(text) and text[end].isalnum() and not replacement.endswith(" "):
			replacement = replacement + " "
	return normalize_name(text[:start] + replacement + text[end:])


def parse_edges(section):
	rows = []
	for line in section.splitlines():
		if line.strip():
			name, address = line.split("\t")
			rows.append((name, address))
	return rows


def parse_variants(section):
	rows = []
	reader = csv.DictReader(io.StringIO(section), skipinitialspace=True)
	for row in reader:
		source = row["source"].strip().strip('"')
		target = row["target"].strip().strip('"')
		rows.append((source, target, variant_pattern(source)))
	return rows


def expand_name(name, variants):
	seen = {name}
	queue = deque([name])
	while queue:
		current = queue.popleft()
		for source, target, pattern in variants:
			for match in pattern.finditer(current):
				candidate = replace_variant(current, match.start(), match.end(), source, target)
				if candidate in seen:
					continue
				seen.add(candidate)
				queue.append(candidate)
	return seen


def replace_words(text, mapping):
	pattern = re.compile("|".join(rf"(?<![0-9a-z]){re.escape(source)}(?![0-9a-z])" for source in sorted(mapping, key=len, reverse=True)))
	return pattern.sub(lambda match: mapping[match.group(0)], text)


def american_rewrite(name):
	return replace_words(name, AMERICAN_MAP)


def drop_plural_rewrite(name):
	return replace_words(name, DROP_PLURAL_MAP)


def row_text(name, addresses):
	return f'{name}<{", ".join(sorted(addresses, key=address_sort_key))}>'


def build_datasets(data_path):
	text = Path(data_path).read_text()
	edges = parse_edges(extract_section(text, "edges"))
	variants = parse_variants(extract_section(text, "variants"))
	name_to_addresses = defaultdict(set)
	for name, address in edges:
		for variant in expand_name(name, variants):
			name_to_addresses[variant].add(address)
	names = sorted(name_to_addresses)
	addresses = sorted({address for address_set in name_to_addresses.values() for address in address_set}, key=address_sort_key)
	finetune = [row_text(name, name_to_addresses[name]) for name in names]
	rewrites = set()
	for name in names:
		american_name = american_rewrite(name)
		drop_name = drop_plural_rewrite(name)
		both_name = drop_plural_rewrite(american_name)
		if american_name != name:
			rewrites.add(row_text(american_name, name_to_addresses[name]))
		if drop_name != name:
			rewrites.add(row_text(drop_name, name_to_addresses[name]))
		if both_name not in {name, american_name, drop_name}:
			rewrites.add(row_text(both_name, name_to_addresses[name]))
	rewrites = sorted(rewrites)
	if len(rewrites) % 2:
		rewrites = rewrites[:-1]
	val = rewrites[::2]
	test = rewrites[1::2]
	return {
		"addresses": addresses,
		"finetune": finetune,
		"name_to_addresses": {name: sorted(address_set, key=address_sort_key) for name, address_set in name_to_addresses.items()},
		"names": names,
		"pretrain": names + addresses,
		"test": test,
		"val": val,
		"variants": variants,
	}


def materialize_data_file(data_path):
	path = Path(data_path)
	text = path.read_text()
	datasets = build_datasets(path)
	updated = text
	for name in ["pretrain", "finetune", "val", "test"]:
		updated = replace_section(updated, name, "txt", datasets[name])
	if updated != text:
		path.write_text(updated)
	return datasets


def read_materialized_rows(data_path):
	text = Path(data_path).read_text()
	return {name: [line for line in extract_section(text, name).splitlines() if line.strip()] for name in ["pretrain", "finetune", "val", "test"]}


def split_rows(rows, seed, ratio=0.9):
	rows = list(rows)
	random.Random(seed).shuffle(rows)
	if len(rows) < 2:
		return rows, rows
	cutoff = max(1, min(len(rows) - 1, int(len(rows) * ratio)))
	return rows[:cutoff], rows[cutoff:]


def set_seed(seed):
	random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
	if hasattr(torch.backends, "cudnn"):
		torch.backends.cudnn.benchmark = False
		torch.backends.cudnn.deterministic = True


def build_tokenizer(rows):
	return CharTokenizer(rows)


def build_model(config, tokenizer, rows):
	block_size = max(len(row) for row in rows)
	block_size = max(block_size, int(config.get("block_size", block_size)))
	return GPT2(
		vocab_size=tokenizer.vocab_size,
		block_size=block_size,
		width=int(config.get("width", 128)),
		heads=int(config.get("heads", 4)),
		layers=int(config.get("layers", 4)),
		dropout=float(config.get("dropout", 0.1)),
	)


def batch_grad_norm(model):
	total = 0.0
	for parameter in model.parameters():
		if parameter.grad is None:
			continue
		value = parameter.grad.detach().norm(2).item()
		total += value * value
	return total ** 0.5


def evaluate_loss(model, rows, tokenizer, device, batch_size):
	if not rows:
		return 0.0
	loader = dataset_rows(rows, tokenizer, batch_size, False)
	model.eval()
	total = 0.0
	count = 0
	with torch.no_grad():
		for inputs, targets in loader:
			inputs = inputs.to(device)
			targets = targets.to(device)
			_, loss = model(inputs, targets)
			total += loss.item() * inputs.shape[0]
			count += inputs.shape[0]
	return total / max(count, 1)


def train_epoch(model, rows, tokenizer, optimizer, device, batch_size, description):
	loader = dataset_rows(rows, tokenizer, batch_size, True)
	model.train()
	total = 0.0
	count = 0
	grad_norm = 0.0
	for inputs, targets in tqdm(loader, desc=description, leave=False, ncols=80):
		inputs = inputs.to(device)
		targets = targets.to(device)
		optimizer.zero_grad(set_to_none=True)
		_, loss = model(inputs, targets)
		loss.backward()
		grad_norm = batch_grad_norm(model)
		optimizer.step()
		total += loss.item() * inputs.shape[0]
		count += inputs.shape[0]
	return total / max(count, 1), grad_norm


def clone_state_dict(model):
	return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def save_checkpoint(path, model, tokenizer, metadata):
	torch.save({"metadata": metadata, "model": model.state_dict(), "tokenizer": tokenizer.itos}, path)


def train_stage(model, stage, train_rows, val_rows, tokenizer, device, run_dir, config, epoch_offset):
	if not train_rows:
		return epoch_offset, []
	checkpoint_dir = Path(run_dir) / "checkpoints"
	batch_size = int(config.get("batch_size", 32))
	epochs = int(config.get(f"{stage}_epochs", config.get("epochs", 100)))
	patience = int(config.get(f"{stage}_patience", config.get("patience", 10)))
	optimizer = torch.optim.AdamW(model.parameters())
	best_epoch = 0
	best_state = clone_state_dict(model)
	best_val_loss = float("inf")
	records = []
	stale = 0
	for stage_epoch in range(1, epochs + 1):
		global_epoch = epoch_offset + stage_epoch
		train_loss, grad_norm = train_epoch(model, train_rows, tokenizer, optimizer, device, batch_size, f"{stage}:{global_epoch:03d}")
		val_loss = evaluate_loss(model, val_rows, tokenizer, device, batch_size)
		record = {
			"epoch": global_epoch,
			"grad_norm": grad_norm,
			"stage": stage,
			"stage_epoch": stage_epoch,
			"train_loss": train_loss,
			"val_loss": val_loss,
		}
		records.append(record)
		save_checkpoint(checkpoint_dir / f"{global_epoch:03d}.pt", model, tokenizer, record)
		if val_loss <= best_val_loss:
			best_epoch = stage_epoch
			best_state = clone_state_dict(model)
			best_val_loss = val_loss
			stale = 0
			continue
		stale += 1
		if stale >= patience:
			break
	model.load_state_dict(best_state)
	best_global_epoch = epoch_offset + best_epoch
	for checkpoint in sorted(checkpoint_dir.glob("*.pt")):
		if int(checkpoint.stem) > best_global_epoch:
			checkpoint.unlink()
	return best_global_epoch, records[:best_epoch]


def prompt_and_target(row):
	name, rest = row.split("<", 1)
	return name + "<", rest[:-1] if rest.endswith(">") else rest


def generate_until_eos(model, tokenizer, prompt, device, max_new_tokens):
	model.eval()
	input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
	start = time.time()
	generated = []
	with torch.no_grad():
		for _ in range(max_new_tokens):
			window = input_ids[:, -model.block_size :]
			logits, _ = model(window)
			next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
			input_ids = torch.cat([input_ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
			character = tokenizer.decode([next_id])
			generated.append(character)
			if character == ">":
				break
	return "".join(generated), time.time() - start


def evaluate_rows(model, rows, tokenizer, metrics, device, max_new_tokens):
	predictions = []
	targets = []
	durations = []
	details = []
	for row in rows:
		prompt, target = prompt_and_target(row)
		completion, duration = generate_until_eos(model, tokenizer, prompt, device, max_new_tokens)
		prediction = completion.split(">", 1)[0]
		predictions.append(prediction)
		targets.append(target)
		durations.append(duration)
		details.append({"prediction": prediction, "prompt": prompt, "target": target, "ttlt": duration})
	scores = {f"mean_{metric.name}": metric(predictions, targets, durations) for metric in metrics}
	return scores, details


def instantiate_metrics(names):
	available = {metric.__name__: metric for metric in [NLS, TTLT]}
	return [available[name]() for name in names]
