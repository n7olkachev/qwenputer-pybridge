# MicroPython for the chipformer RV32 machine as the P4 bridge (Qwenputer's `python` tool).
# Derived from the author's env/mpy/Makefile (ports/minimal based); built inside a MicroPython
# tree as ports/chipformer by ../build.sh:
#   make CONFIG=basic MAP=rom|qemu BUILD=... ENV=<qwenputer env dir> [HOT=...] [COPT=...]
CONFIG ?= basic
MAP ?= rom
STACK_KB ?= 6
BUILD ?= build-$(CONFIG)-$(MAP)
include ../../py/mkenv.mk
FROZEN_MANIFEST ?= $(abspath manifest.py)
QSTR_DEFS = qstrdefsport.h
MICROPY_ROM_TEXT_COMPRESSION ?= 1
include $(TOP)/py/py.mk
include $(TOP)/extmod/extmod.mk
CROSS_COMPILE ?= riscv64-unknown-elf-
ENV ?= $(abspath ../../../env)
PICO ?= /usr/lib/picolibc/riscv64-unknown-elf
PICOLIB ?= $(PICO)/lib/rv32im/ilp32
MARCH ?= rv32im_zba
ARCH = -march=$(MARCH) -mabi=ilp32
INC += -I. -I$(TOP) -I$(BUILD) -I$(ENV) -isystem $(PICO)/include
CFLAGS += $(INC) -Wall -std=gnu99 $(ARCH) $(COPT) -DRV_CFG_$(CONFIG)=1 -DHEAP_KB=0
CFLAGS += -DSTACK_BYTES=$(shell echo $$(( $(STACK_KB) * 1024 - 512 ))) $(if $(filter qemu,$(MAP)),-DQEMU)
CFLAGS += $(OPTFLAGS) -DNDEBUG -fdata-sections -ffunction-sections
OPTFLAGS ?= -Os
CSUPEROPT = $(OPTFLAGS)
AFLAGS = $(ARCH)
LIBGCC = $(shell $(CC) $(ARCH) -print-libgcc-file-name)
LIBS = -Wl,--start-group $(PICOLIB)/libc.a $(PICOLIB)/libm.a $(LIBGCC) -Wl,--end-group
LD = $(CC)
LDFLAGS = $(ARCH) -nostdlib -nostartfiles -static -Wl,--gc-sections -Wl,-Map=$(BUILD)/firmware.map \
  -T$(BUILD)/mpy.ld
SRC_C = main.c shared/runtime/gchelper_native.c shared/runtime/stdout_helpers.c
SRC_S = shared/runtime/gchelper_rv32i.s
HOT ?=
HOTFLAGS ?= -O3
OBJ += $(PY_O)
OBJ += $(addprefix $(BUILD)/, $(SRC_C:.c=.o))
OBJ += $(addprefix $(BUILD)/, $(SRC_S:.s=.o))
OBJ += $(BUILD)/crt0.o $(BUILD)/fastlib.o $(BUILD)/sys.o
$(foreach f,$(HOT),$(eval $(BUILD)/$(f).o: CFLAGS += $(HOTFLAGS)))
# FIRST sets for the parser (gen_first.c on the host, with this port's config): parse.c skips
# alternatives that cannot start with the current token. FIRST_SETS=0 disables it.
FIRST_SETS ?= 1
ifeq ($(FIRST_SETS),1)
HOSTCC ?= cc
GENFLAGS = -I. -I$(TOP) -I$(BUILD) $(COPT) -DRV_CFG_$(CONFIG)=1 -DHEAP_KB=0 -DSTACK_BYTES=0 -DNDEBUG
$(BUILD)/first_sets.h: gen_first.c $(TOP)/py/grammar.h $(TOP)/py/lexer.h mpconfigport.h | $(HEADER_BUILD)/qstrdefs.generated.h
	$(ECHO) "GEN $@"
	$(Q)$(HOSTCC) $(GENFLAGS) -o $(BUILD)/gen_first gen_first.c
	$(Q)$(BUILD)/gen_first > $@
$(BUILD)/py/parse.o: $(BUILD)/first_sets.h
$(BUILD)/py/parse.o: CFLAGS += -DMICROPY_PARSE_FIRST_SETS=1
endif
$(BUILD)/crt0.o: $(ENV)/crt0.S
	$(Q)mkdir -p $(BUILD)
	$(Q)$(CC) $(ARCH) -c -o $@ $<
$(BUILD)/sys.o: $(ENV)/sys.c
	$(Q)mkdir -p $(BUILD)
	$(Q)$(CC) $(ARCH) -O2 -I$(ENV) -isystem $(PICO)/include $(if $(filter qemu,$(MAP)),-DQEMU) -c -o $@ $<
$(BUILD)/fastlib.o: fastlib.c
	$(Q)mkdir -p $(BUILD)
	$(Q)$(CC) $(ARCH) -O2 -isystem $(PICO)/include -fno-builtin -fno-tree-loop-distribute-patterns -c -o $@ $<
$(BUILD)/mpy.ld: $(ENV)/$(MAP).ld
	$(Q)mkdir -p $(BUILD)
	$(Q)sed 's/__heap_end = 0x7000;.*/__heap_end = $(shell printf 0x%x $$(( 0x8000 - $(STACK_KB) * 1024 )));/' $< > $@
all: $(BUILD)/firmware.elf
$(BUILD)/firmware.elf: $(OBJ) $(BUILD)/mpy.ld
	$(ECHO) "LINK $@"
	$(Q)$(LD) $(LDFLAGS) -o $@ $(OBJ) $(LIBS)
	$(Q)$(SIZE) -A $@ | egrep '^\.(text|rodata|data|bss|tdata|tbss)|Total'
include $(TOP)/py/mkrules.mk
