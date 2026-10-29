# ==============================================================================
# SMR-Engine ARM Cortex-M7 Firmware Makefile
# Venue: MLSys 2027 Artifact Evaluation
# ==============================================================================

CC = arm-none-eabi-gcc
OBJCOPY = arm-none-eabi-objcopy
SIZE = arm-none-eabi-size
NM = arm-none-eabi-nm

CFLAGS = -mcpu=cortex-m7 -mthumb -mfpu=fpv5-d16 -mfloat-abi=hard \
         -O3 -Wall -Wextra -ffunction-sections -fdata-sections \
         -fno-builtin-malloc -fno-builtin-free -fno-builtin-realloc \
         -std=c11

LDFLAGS = -Tstm32h743.ld -nostartfiles -nodefaultlibs -lc -lm \
          -Wl,--gc-sections -Wl,-Map=smr_cortex_m7.map

TARGET = smr_cortex_m7.elf

all: $(TARGET) size audit

$(TARGET): stm32h7_smr_arena.c stm32h743.ld
	$(CC) $(CFLAGS) stm32h7_smr_arena.c $(LDFLAGS) -o $@

size: $(TARGET)
	@echo "--- Section Size Breakdown ---"
	$(SIZE) -B $(TARGET)
	$(SIZE) -A $(TARGET)

audit: $(TARGET)
	@echo "--- Auditing Heap Symbols (Expect ZERO references to _sbrk/malloc) ---"
	@if $(NM) $(TARGET) | grep -E "malloc|free|_sbrk" | grep -v "smr"; then \
		echo "AUDIT FAIL: Dynamic heap symbols detected!"; exit 1; \
	else \
		echo "AUDIT PASS: Zero dynamic heap symbols confirmed in ELF binary."; \
	fi

clean:
	rm -f $(TARGET) smr_cortex_m7.map smr_cortex_m7.bin
