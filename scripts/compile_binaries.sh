#!/bin/bash

destfile=~/local/bearssl-0.6/test/test_crypto.c

cd keys
for srcfile in rand*; do
    { sed -n '1,8293p' $destfile; sed -n '1,$p' $srcfile; sed -n '8313,$p' $destfile; } > temp.tmp && mv temp.tmp $destfile
    cd ~/local/bearssl-0.6
    make CONF=riscv clean
    make CONF=riscv tests
    mv build/testcrypto build/testcrypto"$srcfile"
done


