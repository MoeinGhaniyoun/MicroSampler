#!/bin/bash

declare -a apps=("v1" "v1_warmup" "v1_fence" "v2" "v2_warmup" "v2_fence" "v3" "v3_warmup" "v3_fence")

for val in "${apps[@]}"; do
	python3 scripts/generate_table.py $val $2 $1
	echo 'Finished with table for '$val
done
