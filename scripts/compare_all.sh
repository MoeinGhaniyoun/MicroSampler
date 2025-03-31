#!/bin/bash

declare -a apps=("vuln" "vuln_warmup" "vuln_fence" "dummy" "dummy_warmup" "dummy_fence" "consttime" "consttime_warmup" "consttime_fence")

for val in "${apps[@]}"; do
	echo 'Comparing states across all keys of '$val
	python3 scripts/compare_key_traces.py $1 $2 $val 100
done
