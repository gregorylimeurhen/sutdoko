import collections
import fractions
import pathlib
import utils


def transposition_options(text, boundaries):
	options = []
	for boundary_index, boundary in enumerate(text):
		if boundary not in boundaries:
			continue
		left_start = boundary_index
		while left_start > 0 and text[left_start - 1] not in boundaries:
			left_start -= 1
		right_end = boundary_index + 1
		while right_end < len(text) and text[right_end] not in boundaries:
			right_end += 1
		if left_start == boundary_index or right_end == boundary_index + 1:
			continue
		left = text[:left_start]
		mid = text[boundary_index + 1:right_end]
		right = text[left_start:boundary_index]
		tail = text[right_end:]
		options.append(left + mid + boundary + right + tail)
	return sorted(option for option in set(options) if option != text)


def transpose(text, options, transposition_rate, rng):
	if not options or rng.random() >= transposition_rate:
		return text
	return options[rng.randrange(len(options))]


def substitute(text, neighbors, substitution_rate, rng):
	chars = []
	for char in text:
		options = neighbors.get(char, [])
		if options and rng.random() < substitution_rate:
			chars.append(options[rng.randrange(len(options))])
			continue
		chars.append(char)
	return "".join(chars)


def corrupt(text, bounds, nbrs, sub_rate, swap_rate, rng):
	options = transposition_options(text, bounds)
	text = transpose(text, options, swap_rate, rng)
	return substitute(text, nbrs, sub_rate, rng)


def split_targets(corruption_count, data_split):
	train_fraction, val_fraction, test_fraction = data_split
	train_target = fractions.Fraction(str(train_fraction)) * corruption_count
	val_target = fractions.Fraction(str(val_fraction)) * corruption_count
	test_target = fractions.Fraction(str(test_fraction)) * corruption_count
	if min(train_target, val_target, test_target) < 0:
		raise ValueError("split targets must be non-negative")
	if train_target + val_target + test_target != corruption_count:
		raise ValueError("split targets must sum to corruption_count")
	train_den = train_target.denominator
	val_den = val_target.denominator
	test_den = test_target.denominator
	if train_den == val_den == test_den == 1:
		return int(train_target), int(val_target), int(test_target)
	raise ValueError("split targets must be integers")


def split_rows(rooms, bounds, nbrs, count, sub_rate, swap_rate, split, rng):
	train_rows = []
	val_rows = []
	test_rows = []
	if not 0 <= sub_rate <= 1:
		raise ValueError("substitution_rate must be in [0, 1]")
	if not 0 <= swap_rate <= 1:
		raise ValueError("transposition_rate must be in [0, 1]")
	if count and sub_rate == 0 and swap_rate == 0:
		msg = "substitution_rate or transposition_rate must be positive"
		raise ValueError(msg)
	train_target, val_target, test_target = split_targets(count, split)
	for room in rooms:
		pairs = set()
		while len(pairs) < count:
			left = corrupt(room, bounds, nbrs, sub_rate, swap_rate, rng)
			pair = left, room
			if pair[0] == room:
				continue
			pairs.add(pair)
		pairs = sorted(pairs)
		train_pairs = set(rng.sample(pairs, train_target))
		remaining_pairs = [pair for pair in pairs if pair not in train_pairs]
		val_pairs = set(rng.sample(remaining_pairs, val_target))
		test_pairs = [pair for pair in remaining_pairs if pair not in val_pairs]
		train_rows.extend(sorted(train_pairs))
		val_rows.extend(sorted(val_pairs))
		test_rows.extend(test_pairs)
	train_rows.sort()
	val_rows.sort()
	test_rows.sort()
	return train_rows, val_rows, test_rows


def lookup_rows(edges):
	lookup = collections.defaultdict(set)
	for room, address in edges:
		lookup[room].add(address)
	rows = []
	for room, addresses in sorted(lookup.items()):
		rows.append((room, ", ".join(sorted(addresses))))
	return rows


def write_rows(path, rows):
	path.write_text("".join(f"{left}\t{right}\n" for left, right in rows))


def main():
	dir_ = pathlib.Path(__file__).resolve().parent
	config = utils.load_config(dir_, "preprocess")
	rng = utils.Rng(utils.load_seed(dir_))
	bounds = utils.load_boundaries(dir_)
	edges = utils.load_edges(dir_)
	nbrs = utils.load_neighbors(dir_)
	rooms = sorted({room for room, _ in edges})
	count = config["corruption_count"]
	sub = config["substitution_rate"]
	swap = config["transposition_rate"]
	parts = config["data_split"]
	rows = split_rows(rooms, bounds, nbrs, count, sub, swap, parts, rng)
	train_rows, val_rows, test_rows = rows
	write_rows(dir_ / "data" / "train.tsv", train_rows)
	write_rows(dir_ / "data" / "val.tsv", val_rows)
	write_rows(dir_ / "data" / "test.tsv", test_rows)
	write_rows(dir_ / "data" / "n2a.tsv", lookup_rows(edges))


if __name__ == "__main__":
	main()
