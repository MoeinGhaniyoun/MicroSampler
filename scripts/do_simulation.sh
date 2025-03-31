#!/bin/bash

APP="$(echo $3 | cut -d'_' -f1)"
ATYPE="$(echo $3 | cut -d'_' -f2)"

#echo 'Compiling BOOM.....'
#make CONFIG=SmallBoomConfig
echo 'Starting simulation....'
keyval=`cat scripts/keys/$1.key`
pk=$RISCV/riscv64-unknown-elf/bin/pk

if [ "$2" == "bearssl_comb" ]; then
	if [ "$ATYPE" == "warmup" ]; then
		bin_args="$pk apps/bearssl-0.6/build/testcrypto_warmup $APP $keyval $4"
	else
		bin_args="$pk apps/bearssl-0.6/build/testcrypto $APP $keyval $4"
	fi
elif [ "$2" == "bearssl_single" ]; then
	bin_args="$pk apps/bearssl-0.6/build/testcrypto-$3 $keyval"
elif [ "$2" == "bearssl_synthetic" ]; then
	bin_args="$pk apps/bearssl-0.6/microsampler_tests/build/$3 $keyval $4"
elif [ "$2" == "openssl" ]; then
	bin_args="$pk $HOME/local/openssl/test/microsampler_test/exptest $3 $keyval"
elif [ "$2" == "microbench" ]; then
	bin_args="apps/microbench/$3/$1/$3"
fi

echo "command: $SIM_ROOT/BOOM_simulator +verbose $bin_args > logs/$5/$2/$3/$4/$1/stdout.txt 2> logs/$5/$2/$3/$4/$1/out-all.log"
time $SIM_ROOT/BOOM_simulator +verbose $bin_args > logs/$5/$2/$3/$4/$1/stdout.txt 2> logs/$5/$2/$3/$4/$1/out-all.log
echo "Running spike-dasm on output log..."
cat logs/$5/$2/$3/$4/$1/out-all.log | spike-dasm > logs/$5/$2/$3/$4/$1/out-all-asm.log
echo "Compressing output log into gzip format..."
gzip -f logs/$5/$2/$3/$4/$1/out-all-asm.log
echo "Cleaning up....."
rm logs/$5/$2/$3/$4/$1/out-all.log
echo "done."
#echo `cat logs/$5/$2/$3/$4/$1/stdout.txt` | mail -s "simulation $5 $3 $1 complete" barber.m.kristin@gmail.com 
