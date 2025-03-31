#!/bin/bash
echo 'Parsing micro-arch log, collecting state samples...'
time python scripts/parse_trace.py logs/${10}/$7/$8/$9/$1/out-all-asm.log.gz logs/${10}/$7/$8/$9/$1/uarch.pickle $2 $3 $4 $5 "$6" > logs/${10}/$7/$8/$9/$1/parser.log 2>&1; #echo `cat logs/${10}/$7/$8/$9/$1/parser.log` | mail -s "parsing ${10} $8 $1 complete" barber.m.kristin@gmail.com 
