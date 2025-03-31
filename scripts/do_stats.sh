#!/bin/bash

echo 'Running state analysis....'
time python scripts/stats.py logs/$8/$2/$3/$7/$1/uarch.pickle scripts/keys/$1.key logs/$8/$2/$3/$7/$1/sets.pickle logs/$8/$2/$3/$7/$1 $4 $5 $6 $7 > logs/$8/$2/$3/$7/$1/stats-$4_$5.log 2>&1; python3 scripts/miss_stats.py $3 $1 $8 $2
#; python3 scripts/simout_binkey.py $3 $1 $8 $2
