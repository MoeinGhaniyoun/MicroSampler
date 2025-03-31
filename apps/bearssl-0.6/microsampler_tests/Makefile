CC=riscv64-unknown-elf-gcc
OBJDUMP=riscv64-unknown-elf-objdump
CFLAGS=-Os --static -I../src -I../inc
BDIR=build
SDIR=src
SOURCES=../src/codec/ccopy.c ../src/int/i31_tmont.c ../src/int/i31_montmul.c ../src/int/i31_modpow.c ../src/int/i31_ninv31.c ../src/int/i31_muladd.c ../src/util/util.c ../src/int/i31_sub.c ../src/int/i31_add.c ../src/int/i32_div32.c


_BIN = v1 v2 v3 v1_warmup v2_warmup v3_warmup v1_fence v2_fence v3_fence
BIN = $(patsubst %,$(BDIR)/%,$(_BIN))

$(BDIR)/%: $(SDIR)/%.c
	$(CC) -o $@ $< $(CFLAGS) $(SOURCES)
	$(OBJDUMP) -d $@ > $@.dump

all: $(BIN)

.PHONY: clean

clean:
	rm -f $(BDIR)/*
