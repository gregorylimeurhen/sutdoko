import random, time
import src.models, src.utils


def edit_distance(left, right):
	previous = list(range(len(right) + 1))
	for left_index, left_char in enumerate(left, start=1):
		current = [left_index]
		for right_index, right_char in enumerate(right, start=1):
			current.append(min(current[-1] + 1, previous[right_index] + 1, previous[right_index - 1] + (left_char != right_char)))
		previous = current
	return previous[-1]


def identity_address(text, room_lookup):
	return room_lookup[text] if text in room_lookup else ""


def levenshtein_address(text, room_lookup, rooms, rng):
	best_distance = None
	best_rooms = []
	for room in rooms:
		distance = edit_distance(text, room)
		if best_distance is None or distance < best_distance:
			best_distance = distance
			best_rooms = [room]
			continue
		if distance == best_distance:
			best_rooms.append(room)
	room = best_rooms[rng.randrange(len(best_rooms))]
	return room_lookup[room]


def ours_address(text, model, tokenizer, device, room_lookup, rooms, room_set):
	if text in room_set:
		return room_lookup[text]
	return room_lookup[src.models.predict_room(model, tokenizer, device, text, rooms)]


def evaluate_rows(model, rows, tokenizer, device, room_lookup, rooms):
	room_set = set(rooms)
	rng = random.Random(0)
	stats = {name: {"correct": 0, "latency": 0.0} for name in ["identity", "levenshtein", "ours"]}
	details = []
	for row in rows:
		text = src.utils.normalize(row["input"])
		gold_room = src.utils.normalize(row["gold"])
		gold_address = room_lookup[gold_room]
		start = time.perf_counter()
		identity = identity_address(text, room_lookup)
		stats["identity"]["latency"] += time.perf_counter() - start
		stats["identity"]["correct"] += int(identity == gold_address)
		start = time.perf_counter()
		levenshtein = levenshtein_address(text, room_lookup, rooms, rng)
		stats["levenshtein"]["latency"] += time.perf_counter() - start
		stats["levenshtein"]["correct"] += int(levenshtein == gold_address)
		start = time.perf_counter()
		ours = ours_address(text, model, tokenizer, device, room_lookup, rooms, room_set)
		stats["ours"]["latency"] += time.perf_counter() - start
		stats["ours"]["correct"] += int(ours == gold_address)
		details.append(
			{
				"input": text,
				"gold_room": gold_room,
				"gold": gold_address,
				"identity": identity,
				"levenshtein": levenshtein,
				"ours": ours,
			}
		)
	total = len(rows)
	return {
		name: {
			"accuracy": values["correct"] / total,
			"mean_latency": values["latency"] / total,
		}
		for name, values in stats.items()
	}, details
