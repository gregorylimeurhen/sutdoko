import random
import torch
from pathlib import Path
from tqdm import tqdm
from .architectures import *
from .data import *


class NLS:
	name = "nls"

	def __call__(self, predictions, targets):
		if not targets:
			return 0.0
		return sum(self.score(prediction, target) for prediction, target in zip(predictions, targets)) / len(targets)

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

	def score(self, prediction, target):
		if not prediction and not target:
			return 1.0
		return 1.0 - self.levenshtein(prediction, target) / max(len(prediction), len(target), 1)


def batch_grad_norm(model):
	total = 0.0
	for parameter in model.parameters():
		if parameter.grad is None:
			continue
		value = parameter.grad.detach().norm(2).item()
		total += value * value
	return total ** 0.5


def evaluate_rows(model, rows, tokenizer, metric, device, max_tokens):
	details = []
	predictions = []
	targets = []
	for row in rows:
		prompt, target = prompt_and_target(row)
		prediction = generate_until_eos(model, tokenizer, prompt, device, max_tokens).split(">", 1)[0]
		details.append({"gold": target, "input": prompt, "output": prediction})
		predictions.append(prediction)
		targets.append(target)
	return {f"mean_{metric.name}": metric(predictions, targets)}, details


def fits_batch_size(model, sample, pad_id, device, batch_size):
	try:
		inputs, targets = collate_rows([sample] * batch_size, pad_id)
		inputs = inputs.to(device)
		targets = targets.to(device)
		model.zero_grad(set_to_none=True)
		_, loss = model(inputs, targets)
		loss.backward()
		if device.type == "cuda":
			torch.cuda.synchronize(device)
		return True
	except RuntimeError as error:
		if "out of memory" not in str(error).lower():
			raise
		return False
	finally:
		model.zero_grad(set_to_none=True)
		if device.type == "cuda":
			torch.cuda.empty_cache()


def generate_until_eos(model, tokenizer, prompt, device, max_tokens):
	model.eval()
	generated = []
	input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
	with torch.no_grad():
		for _ in range(max_tokens):
			logits, _ = model(input_ids[:, -model.block_size :])
			next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
			input_ids = torch.cat([input_ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
			character = tokenizer.decode([next_id])
			generated.append(character)
			if character == ">":
				break
	return "".join(generated)


def infer_batch_size(model, rows, tokenizer, device, maximum=512):
	sample = longest_training_row(rows, tokenizer)
	if sample is None:
		return 1
	if device.type != "cuda":
		return max(1, min(32, 4096 // max(sample.numel() - 1, 1)))
	if not fits_batch_size(model, sample, tokenizer.pad_id, device, 1):
		raise RuntimeError("Unable to fit a single training example on the selected CUDA device")
	low = 1
	high = 2
	while high <= maximum and fits_batch_size(model, sample, tokenizer.pad_id, device, high):
		low = high
		if high == maximum:
			return high
		high = min(high * 2, maximum)
	while low + 1 < high:
		mid = (low + high) // 2
		if fits_batch_size(model, sample, tokenizer.pad_id, device, mid):
			low = mid
			continue
		high = mid
	return low


def instantiate_metric(name):
	return {"NLS": NLS}[name]()


def load_checkpoint(path, architecture, device):
	payload = torch.load(path, map_location=device)
	tokenizer = CharTokenizer.from_itos(payload["tokenizer"])
	block_size = payload["model"]["position_embedding.weight"].shape[0]
	model = build_model(architecture, tokenizer, block_size).to(device)
	model.load_state_dict(payload["model"])
	return model, tokenizer


def longest_training_row(rows, tokenizer):
	dataset = TextDataset(rows, tokenizer)
	if not len(dataset):
		return None
	return max(dataset.rows, key=lambda item: item.numel())


def save_checkpoint(path, model, tokenizer, metadata):
	torch.save({"metadata": metadata, "model": model.state_dict(), "tokenizer": tokenizer.itos}, path)


def set_seed(seed):
	random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
	if hasattr(torch.backends, "cudnn"):
		torch.backends.cudnn.benchmark = False
		torch.backends.cudnn.deterministic = True


def train_epoch(model, rows, tokenizer, optimizer, device, batch_size, description):
	count = 0
	grad_norm = 0.0
	total = 0.0
	for inputs, targets in tqdm(dataset_rows(rows, tokenizer, batch_size, True), desc=description, leave=False, ncols=80):
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


def train_stage(model, stage, rows, tokenizer, device, run_dir, epochs, batch_size, epoch_offset, run, identifier):
	checkpoint_dir = Path(run_dir) / "checkpoints"
	optimizer = torch.optim.AdamW(model.parameters())
	for stage_epoch in range(1, epochs + 1):
		global_epoch = epoch_offset + stage_epoch
		loss, grad_norm = train_epoch(model, rows, tokenizer, optimizer, device, batch_size, f"{stage}:{global_epoch:04d}")
		record = {"id": identifier, "epoch": global_epoch, "stage": stage, "loss": loss, "grad_norm": grad_norm}
		run.log(record, step=global_epoch)
		save_checkpoint(checkpoint_dir / f"{global_epoch:04d}.pt", model, tokenizer, record)
	return epoch_offset + epochs


__all__ = [
	"NLS",
	"evaluate_rows",
	"infer_batch_size",
	"instantiate_metric",
	"load_checkpoint",
	"save_checkpoint",
	"set_seed",
	"train_stage",
]
