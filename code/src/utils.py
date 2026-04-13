import dataclasses, json, pathlib, random
import torch

EPSILON = 0.2
AUGMENTATION_COUNT = 100
TRAIN_AUGMENTATION_COUNT = 80
PAD_TOKEN = "<pad>"
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


def set_seed(seed):
	random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def load_neighbors(root):
	return json.loads((pathlib.Path(root) / "data" / "neighbors.json").read_text())


def load_pairs(path):
	rows = []
	for line in pathlib.Path(path).read_text().splitlines():
		if not line:
			continue
		left, right = line.split("\t")
		rows.append({"input": normalize(left), "gold": normalize(right)})
	return rows


def load_training_rows(root):
	return load_pairs(pathlib.Path(root) / "data" / "train.tsv")


def load_test_rows(root):
	return load_pairs(pathlib.Path(root) / "data" / "test.tsv")


def load_room_lookup(root):
	lookup = {}
	for line in (pathlib.Path(root) / "data" / "n2a.tsv").read_text().splitlines():
		if not line:
			continue
		room, address = line.split("\t")
		lookup[normalize(room)] = address.strip()
	return lookup


def load_rooms(root):
	return sorted(load_room_lookup(root))


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


def encode_pair(input_text, output_text, tokenizer):
	prompt = tokenizer.encode_text(normalize(input_text))
	target = tokenizer.encode_text(normalize(output_text))
	tokens = prompt + [tokenizer.sep_id] + target + [tokenizer.eos_id]
	labels = list(tokens[1:])
	if prompt:
		labels[:len(prompt)] = [-100] * len(prompt)
	return {"input_ids": tokens[:-1], "labels": labels}


def encode_from_prompt(prompt_ids, output_text, tokenizer):
	target = tokenizer.encode_text(normalize(output_text))
	tokens = list(prompt_ids) + [tokenizer.sep_id] + target + [tokenizer.eos_id]
	labels = list(tokens[1:])
	if prompt_ids:
		labels[:len(prompt_ids)] = [-100] * len(prompt_ids)
	return {"input_ids": tokens[:-1], "labels": labels}


def collate_examples(examples, tokenizer, device):
	sequence_len = max(len(example["input_ids"]) for example in examples)
	input_ids = torch.full((len(examples), sequence_len), tokenizer.pad_id, dtype=torch.long, device=device)
	labels = torch.full((len(examples), sequence_len), -100, dtype=torch.long, device=device)
	for row_index, example in enumerate(examples):
		length = len(example["input_ids"])
		input_ids[row_index, :length] = torch.tensor(example["input_ids"], dtype=torch.long, device=device)
		labels[row_index, :length] = torch.tensor(example["labels"], dtype=torch.long, device=device)
	return input_ids, labels
