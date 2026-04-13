import dataclasses, math, random
import torch, torch.nn as nn, torch.nn.functional as F
import src.utils


@dataclasses.dataclass
class ModelConfig:
	depth: int
	sequence_len: int
	vocab_size: int
	n_head: int
	n_embd: int


class RMSNorm(nn.Module):
	def __init__(self, dim):
		super().__init__()
		self.weight = nn.Parameter(torch.ones(dim))

	def forward(self, x):
		return F.rms_norm(x, (x.size(-1),), self.weight)


def apply_rotary(x, cos, sin):
	left = x[..., ::2]
	right = x[..., 1::2]
	rotated = torch.stack((left * cos - right * sin, left * sin + right * cos), dim=-1)
	return rotated.flatten(-2)


class RotaryEmbedding(nn.Module):
	def __init__(self, sequence_len, head_dim):
		super().__init__()
		positions = torch.arange(sequence_len, dtype=torch.float32)
		frequencies = 10000 ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
		angles = torch.outer(positions, frequencies)
		self.register_buffer("cos", angles.cos().unsqueeze(0).unsqueeze(0), persistent=False)
		self.register_buffer("sin", angles.sin().unsqueeze(0).unsqueeze(0), persistent=False)

	def forward(self, q, k):
		cos = self.cos[:, :, :q.size(2)]
		sin = self.sin[:, :, :q.size(2)]
		return apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)


class SelfAttention(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.n_head = config.n_head
		self.head_dim = config.n_embd // config.n_head
		self.qkv = nn.Linear(config.n_embd, config.n_embd * 3, bias=False)
		self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
		self.rotary = RotaryEmbedding(config.sequence_len, self.head_dim)

	def forward(self, x):
		batch_size, sequence_len, _ = x.shape
		q, k, v = self.qkv(x).chunk(3, dim=-1)
		q = q.view(batch_size, sequence_len, self.n_head, self.head_dim).transpose(1, 2)
		k = k.view(batch_size, sequence_len, self.n_head, self.head_dim).transpose(1, 2)
		v = v.view(batch_size, sequence_len, self.n_head, self.head_dim).transpose(1, 2)
		q, k = self.rotary(q, k)
		q = F.rms_norm(q, (self.head_dim,))
		k = F.rms_norm(k, (self.head_dim,))
		y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
		y = y.transpose(1, 2).contiguous().view(batch_size, sequence_len, -1)
		return self.proj(y)


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
		x = x + self.mlp(self.mlp_norm(x))
		return x


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
		return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)


def build_config(depth, tokenizer, sequence_len):
	head_dim = 128
	n_embd = max(head_dim, math.ceil(depth * 64 / head_dim) * head_dim)
	return ModelConfig(depth, sequence_len, len(tokenizer.vocab), n_embd // head_dim, n_embd)


def build_model(depth, tokenizer, sequence_len):
	return GPT(build_config(depth, tokenizer, sequence_len))


def save_checkpoint(path, model, tokenizer, rooms):
	torch.save(
		{
			"config": dataclasses.asdict(model.config),
			"model": model.state_dict(),
			"rooms": list(rooms),
			"tokenizer": tokenizer.to_dict(),
		},
		path,
	)


def load_checkpoint(path, device):
	state = torch.load(path, map_location=device)
	tokenizer = src.utils.Tokenizer.from_dict(state["tokenizer"])
	model = GPT(ModelConfig(**state["config"])).to(device)
	model.load_state_dict(state["model"])
	model.eval()
	return model, tokenizer, list(state["rooms"])


def can_fit_batch(model, examples, tokenizer, device):
	input_ids = None
	labels = None
	loss = None
	try:
		model.zero_grad(set_to_none=True)
		input_ids, labels = src.utils.collate_examples(examples, tokenizer, device)
		loss = model(input_ids, labels)
		loss.backward()
		return True
	except RuntimeError as error:
		if device.type != "cuda" or "out of memory" not in str(error).lower():
			raise
		return False
	finally:
		model.zero_grad(set_to_none=True)
		if device.type == "cuda":
			torch.cuda.empty_cache()
		del input_ids, labels, loss


def largest_batch_size(model, examples, tokenizer, device):
	if device.type != "cuda":
		return min(32, len(examples))
	longest = sorted(examples, key=lambda example: len(example["input_ids"]), reverse=True)
	best = 1
	candidate = 1
	while candidate <= len(longest) and can_fit_batch(model, longest[:candidate], tokenizer, device):
		best = candidate
		candidate *= 2
	if candidate > len(longest):
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
	return best


def train(model, rows, tokenizer, device, run_dir, epochs, run):
	examples = [src.utils.encode_pair(row["input"], row["gold"], tokenizer) for row in rows]
	batch_size = largest_batch_size(model, examples, tokenizer, device)
	optimizer = torch.optim.Adam(model.parameters())
	rooms = sorted({row["gold"] for row in rows})
	for epoch in range(1, epochs + 1):
		order = list(range(len(examples)))
		random.Random(epoch).shuffle(order)
		model.train()
		total_loss = 0.0
		total_steps = 0
		for start in range(0, len(order), batch_size):
			batch = [examples[index] for index in order[start:start + batch_size]]
			input_ids, labels = src.utils.collate_examples(batch, tokenizer, device)
			optimizer.zero_grad(set_to_none=True)
			loss = model(input_ids, labels)
			loss.backward()
			optimizer.step()
			total_loss += float(loss.item())
			total_steps += 1
		save_checkpoint(run_dir / "checkpoints" / f"{epoch:04d}.pt", model, tokenizer, rooms)
		run.log({"epoch": epoch, "train_loss": total_loss / total_steps, "batch_size": batch_size})
	return batch_size


def predict_room(model, tokenizer, device, text, rooms):
	prompt = tokenizer.encode_text(src.utils.normalize(text))
	best_room = rooms[0]
	best_score = None
	model.eval()
	with torch.no_grad():
		for start in range(0, len(rooms), 64):
			batch_rooms = rooms[start:start + 64]
			examples = [src.utils.encode_from_prompt(prompt, room, tokenizer) for room in batch_rooms]
			input_ids, labels = src.utils.collate_examples(examples, tokenizer, device)
			logits = model(input_ids)
			log_probs = F.log_softmax(logits, dim=-1)
			safe_labels = labels.masked_fill(labels < 0, 0)
			scores = log_probs.gather(2, safe_labels.unsqueeze(-1)).squeeze(-1)
			scores = scores.masked_fill(labels < 0, 0.0).sum(dim=1)
			index = int(scores.argmax().item())
			score = float(scores[index].item())
			if best_score is None or score > best_score:
				best_score = score
				best_room = batch_rooms[index]
	return best_room
