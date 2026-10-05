#!/usr/bin/env bash
# Builds every ```odin block in README.md against this library and fails if
# any of them does not compile. Run from anywhere:
#
#     scripts/check_readme.sh [README.md]
#
# Each block must be a complete `package main` program. Its placeholder import
# path "path/to/odin-tensor" is rewritten to point at this checkout.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
readme=${1:-"$repo_root/README.md"}
work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT

# Split the README into one directory per odin block, named by start line.
awk -v out="$work_dir" '
	/^```odin[[:space:]]*$/ { in_block = 1; dir = out "/line_" NR; system("mkdir -p " dir); file = dir "/main.odin"; next }
	/^```/ && in_block      { in_block = 0; close(file); next }
	in_block                { print > file }
' "$readme"

# A collection rooted at the repository's parent lets the examples import it.
collection="readme_lib=$(dirname "$repo_root")"
import_path="readme_lib:$(basename "$repo_root")"

blocks=0
failures=0
for dir in $(ls -d "$work_dir"/line_* 2>/dev/null | sort -t_ -k2 -n); do
	blocks=$((blocks + 1))
	line=${dir##*_}
	sed -i "s|\"path/to/odin-tensor\"|\"$import_path\"|" "$dir/main.odin"
	if output=$(odin build "$dir" -collection:"$collection" -vet -strict-style -out:"$dir/example" 2>&1); then
		echo "PASS README.md:$line"
	else
		failures=$((failures + 1))
		echo "FAIL README.md:$line"
		echo "$output" | sed "s|$dir/main.odin|README.md block at line $line|"
	fi
done

if [ "$blocks" -eq 0 ]; then
	echo "no \`\`\`odin blocks found in $readme"
	exit 1
fi
echo "$((blocks - failures))/$blocks README examples compiled"
[ "$failures" -eq 0 ]
