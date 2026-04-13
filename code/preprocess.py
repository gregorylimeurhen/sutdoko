import collections, json, pathlib, random

EPSILON = 0.2
AUGMENTATION_COUNT = 100
TRAIN_AUGMENTATION_COUNT = 80


def normalize(text):
	return text.strip().lower()


def load_edges(root):
	rows = []
	for line in (root / "data" / "edges.tsv").read_text().splitlines():
		if not line:
			continue
		room, address = line.split("\t")
		rows.append((normalize(room), address.strip()))
	return rows


def load_neighbors(root):
	return json.loads((root / "data" / "neighbors.json").read_text())


def corrupt(text, neighbors, rng):
	return "".join(
		neighbors[char][rng.randrange(len(neighbors[char]))] if neighbors[char] and rng.random() < EPSILON else char
		for char in text
	)


def split_rows(rooms, neighbors, rng):
	train_rows = []
	test_rows = []
	for room in rooms:
		seen = set()
		train_count = 0
		while train_count < TRAIN_AUGMENTATION_COUNT or len(seen) < AUGMENTATION_COUNT:
			pair = (corrupt(room, neighbors, rng), room)
			if pair in seen:
				continue
			seen.add(pair)
			if train_count < TRAIN_AUGMENTATION_COUNT:
				train_rows.append(pair)
				train_count += 1
				continue
			test_rows.append(pair)
	train_rows.sort()
	test_rows.sort()
	return train_rows, test_rows


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
	edges = load_edges(root)
	rooms = sorted({room for room, _ in edges})
	neighbors = load_neighbors(root)
	train_rows, test_rows = split_rows(rooms, neighbors, rng)
	write_rows(root / "data" / "train.tsv", train_rows)
	write_rows(root / "data" / "test.tsv", test_rows)
	write_rows(root / "data" / "n2a.tsv", lookup_rows(edges))


if __name__ == "__main__":
	main()
