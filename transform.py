import json, os, sys

def main():
	if len(sys.argv) != 2:
		sys.stderr.write(f"Usage: {os.path.basename(sys.argv[0])} <input.jsonl>\n")
		return 2

	input_path = sys.argv[1]
	base, ext = os.path.splitext(input_path)
	output_path = f"{base}.txt"

	try:
		with open(input_path, "r", encoding="utf-8") as input_file, open(output_path, "w", encoding="utf-8", newline="\n") as output_file:
			for line_no, raw_line in enumerate(input_file, start=1):
				line = raw_line.strip()
				if not line:
					continue
				try:
					record = json.loads(line)
				except json.JSONDecodeError as exc:
					sys.stderr.write(f"Invalid JSON on line {line_no}: {exc}\n")
					return 1

				name = record.get("name")
				locations = record.get("location")

				if not isinstance(name, str):
					sys.stderr.write(f"Missing or invalid 'name' on line {line_no}\n")
					return 1
				if not isinstance(locations, list) or any(not isinstance(loc, str) for loc in locations):
					sys.stderr.write(f"Missing or invalid 'location' on line {line_no}\n")
					return 1

				output_file.write(f"{name}\t{','.join(locations)}\n")
	except FileNotFoundError:
		sys.stderr.write(f"File not found: {input_path}\n")
		return 1
	except OSError as exc:
		sys.stderr.write(f"I/O error: {exc}\n")
		return 1

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
