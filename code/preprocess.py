import collections, fractions, pathlib, random
import src.utils


def corrupt(text, neighbors, mutation_rate, rng):
	return "".join(
		neighbors[char][rng.randrange(len(neighbors[char]))] if neighbors[char] and rng.random() < mutation_rate else char
		for char in text
	)


def split_targets(mutation_count, training_fraction, dev_fraction):
	dev_fraction = (1 - training_fraction) / 2 if dev_fraction is None else dev_fraction
	train_target = fractions.Fraction(str(training_fraction)) * mutation_count
	dev_target = fractions.Fraction(str(dev_fraction)) * mutation_count
	test_target = fractions.Fraction(mutation_count) - train_target - dev_target
	if min(train_target, dev_target, test_target) < 0:
		raise ValueError("split targets must be non-negative")
	if train_target.denominator == dev_target.denominator == test_target.denominator == 1:
		return int(train_target), int(dev_target)
	raise ValueError("split targets must be integers")


def split_rows(rooms, neighbors, mutation_count, mutation_rate, training_fraction, dev_fraction, rng):
	train_rows = []
	dev_rows = []
	test_rows = []
	train_target, dev_target = split_targets(mutation_count, training_fraction, dev_fraction)
	for room in rooms:
		seen = set()
		train_pairs = set()
		dev_pairs = set()
		test_pairs = set()
		while len(seen) < mutation_count:
			pair = (corrupt(room, neighbors, mutation_rate, rng), room)
			if pair[0] == room:
				continue
			if pair in seen:
				continue
			seen.add(pair)
			if len(train_pairs) < train_target:
				train_pairs.add(pair)
				continue
			if len(dev_pairs) < dev_target:
				dev_pairs.add(pair)
				continue
			test_pairs.add(pair)
		train_rows.extend(train_pairs)
		dev_rows.extend(dev_pairs)
		test_rows.extend(test_pairs)
	train_rows.sort()
	dev_rows.sort()
	test_rows.sort()
	return train_rows, dev_rows, test_rows


def lookup_rows(edges):
	lookup = collections.defaultdict(set)
	for room, address in edges:
		lookup[room].add(address)
	return [(room, ", ".join(sorted(addresses))) for room, addresses in sorted(lookup.items())]


def write_rows(path, rows):
	path.write_text("".join(f"{left}\t{right}\n" for left, right in rows))


def main():
	root = pathlib.Path(__file__).resolve().parent
	rng = random.Random(0)
	config = src.utils.load_config(root, "preprocess")
	edges = src.utils.load_edges(root)
	rooms = sorted({room for room, _ in edges})
	neighbors = src.utils.load_neighbors(root)
	train_rows, dev_rows, test_rows = split_rows(
		rooms,
		neighbors,
		config["mutation_count"],
		config["mutation_rate"],
		config["training_fraction"],
		config.get("dev_fraction"),
		rng,
	)
	write_rows(root / "data" / "train.tsv", train_rows)
	write_rows(root / "data" / "dev.tsv", dev_rows)
	write_rows(root / "data" / "test.tsv", test_rows)
	write_rows(root / "data" / "n2a.tsv", lookup_rows(edges))


if __name__ == "__main__":
	main()
