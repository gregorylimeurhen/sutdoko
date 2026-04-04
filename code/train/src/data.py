import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset


class TextDataset(Dataset):
	def __init__(self, rows, tokenizer):
		self.rows = [torch.tensor(tokenizer.encode(row), dtype=torch.long) for row in rows if len(row) > 1]

	def __getitem__(self, index):
		return self.rows[index]

	def __len__(self):
		return len(self.rows)


class CharTokenizer:
	def __init__(self, rows):
		characters = sorted({character for row in rows for character in row})
		self.pad_token = "\0"
		self.itos = [self.pad_token] + characters
		self.stoi = {token: index for index, token in enumerate(self.itos)}

	@classmethod
	def from_itos(cls, itos):
		tokenizer = cls.__new__(cls)
		tokenizer.pad_token = "\0"
		tokenizer.itos = list(itos)
		tokenizer.stoi = {token: index for index, token in enumerate(tokenizer.itos)}
		return tokenizer

	def decode(self, ids):
		return "".join(self.itos[index] for index in ids if index and index < len(self.itos))

	def encode(self, text):
		return [self.stoi[character] for character in text]

	@property
	def pad_id(self):
		return 0

	@property
	def vocab_size(self):
		return len(self.itos)


def build_tokenizer(rows):
	return CharTokenizer(rows)


def collate_rows(batch, pad_id):
	max_length = max(item.numel() for item in batch) - 1
	inputs = torch.full((len(batch), max_length), pad_id, dtype=torch.long)
	targets = torch.full((len(batch), max_length), -100, dtype=torch.long)
	for index, item in enumerate(batch):
		inputs[index, : item[:-1].numel()] = item[:-1]
		targets[index, : item[1:].numel()] = item[1:]
	return inputs, targets


def dataset_rows(rows, tokenizer, batch_size, shuffle):
	return DataLoader(TextDataset(rows, tokenizer), batch_size=batch_size, shuffle=shuffle, collate_fn=lambda batch: collate_rows(batch, tokenizer.pad_id))


def infer_max_tokens(rows, tokenizer):
	if not rows:
		return 1
	limits = [len(tokenizer.encode(prompt_and_target(row)[1] + ">")) for row in rows if "<" in row]
	return max(limits, default=1)


def load_rows(path):
	return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def load_test_rows(path):
	return load_rows(path)


def load_training_rows(root):
	data_dir = Path(root) / "data"
	return {
		"finetune": load_rows(data_dir / "finetune.txt"),
		"pretrain": load_rows(data_dir / "pretrain.txt"),
	}


def prompt_and_target(row):
	name, rest = row.split("<", 1)
	return name + "<", rest[:-1] if rest.endswith(">") else rest


def rows_block_size(rows):
	return max(len(row) for row in rows)


__all__ = [
	"CharTokenizer",
	"TextDataset",
	"build_tokenizer",
	"collate_rows",
	"dataset_rows",
	"infer_max_tokens",
	"load_test_rows",
	"load_training_rows",
	"prompt_and_target",
	"rows_block_size",
]
