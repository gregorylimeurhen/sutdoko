#!/usr/bin/env bash

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ask() {
	local prompt="$1" cmd="$2"
	local sel=0

	while true; do
		if [ $sel -eq 0 ]; then
			echo -ne "\r$prompt    [x] yes  [ ] no  "
		else
			echo -ne "\r$prompt    [ ] yes  [x] no  "
		fi

		IFS= read -rsn1 key
		if [[ $key == $'\x1b' ]]; then
			read -rsn2 key
			if [[ $key == "[D" || $key == "[C" || $key == "[A" || $key == "[B" ]]; then
				sel=$((1 - sel))
			fi
		elif [[ $key == "" ]]; then
			echo
			if [ $sel -eq 0 ]; then
				(cd "$DIR" && eval "$cmd")
			fi
			return
		fi
	done
}

echo
cat "$DIR/config.toml"
echo

ask "preprocess?" "python -B preprocess.py"
ask "train?" "python -B train.py"
ask "test?" "python -B test.py"
