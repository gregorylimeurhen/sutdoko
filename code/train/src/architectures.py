import math
import torch
import torch.nn.functional as F
from torch import nn


class CausalSelfAttention(nn.Module):
	def __init__(self, width, heads, dropout):
		super().__init__()
		self.dropout = dropout
		self.heads = heads
		self.qkv = nn.Linear(width, width * 3)
		self.proj = nn.Linear(width, width)
		self.resid_dropout = nn.Dropout(dropout)

	def forward(self, x):
		batch, steps, channels = x.shape
		head_width = channels // self.heads
		query, key, value = self.qkv(x).chunk(3, dim=-1)
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
			weights = F.dropout(F.softmax(weights, dim=-1), p=self.dropout, training=self.training)
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


def build_model(name, tokenizer, block_size):
	if name != "GPT2":
		raise ValueError(f"Unsupported architecture {name}")
	return GPT2(vocab_size=tokenizer.vocab_size, block_size=block_size)


__all__ = ["GPT2", "build_model"]
