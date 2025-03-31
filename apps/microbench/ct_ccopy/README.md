
# Overview
With baremetal we don't have a way to pass input (keys). The key value needs to be hard-coded into the binary. That's why there is a binary for every key.
This is essentially running in M-mode with no kernel or runtime env.

The ct_ccopy test includes the constant-time copy primitive from BearSSL and calls it in a loop. Doing so mimics the way it would be used in the basic square-and-multiply algorithm.

# Compile the Test Baremetal

<code> riscv64-unknown-elf-gcc -Os -fno-inline -fno-common -fno-builtin-printf -specs=htif_nano.specs -c ct.c </code>

<code> riscv64-unknown-elf-gcc -Os -static -specs=htif_nano.specs ct.o -o ct_ccopy </code>

## Debugging

<code> riscv64-unknown-elf-objdump --disassemble-all --disassemble-zeroes --section=.text --section=.text.startup --section=.text.init --section=.data ct_ccopy > ct_ccopy.dump </code>


## Adding <code> nop </code> for Fine-Tuned Delays

1. Run the compiler, generate assembly file

<code> riscv64-unknown-elf-gcc -Os -fno-inline -fno-common -fno-builtin-printf -specs=htif_nano.specs -S -o ct.S ct.c </code>

2. Edit the assembly (.S) file directly to insert nops (pipe bubbles, delays that compiler may remove), then call assembler to generate object code 

<code> riscv64-unknown-elf-as ct.S -o ct.o </code>

3. Compile binary using object file from previous step

<code> riscv64-unknown-elf-gcc -Os -static -specs=htif_nano.specs ct.o -o ct_ccopy </code>
