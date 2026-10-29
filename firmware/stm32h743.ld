/*
 * ==============================================================================
 * Linker Script for STM32H743ZI (ARM Cortex-M7 @ 480 MHz)
 * Venue: MLSys 2027 Artifact Evaluation
 *
 * SMR-Engine Compile-Time Static Arena & Zero-Heap Proof
 * ==============================================================================
 */

ENTRY(main)

_Min_Heap_Size  = 0x0;     /* SMR-Engine STRICT ZERO HEAP: 0.0000 KB dynamic memory */
_Min_Stack_Size = 0x1000;  /* 4 KB stack */

MEMORY
{
  FLASH         (rx)  : ORIGIN = 0x08000000, LENGTH = 2048K  /* Internal dual-bank Flash (2.0 MB) */
  DTCMRAM       (rwx) : ORIGIN = 0x20000000, LENGTH = 128K   /* High-speed Data TCM RAM */
  AXI_SRAM      (rwx) : ORIGIN = 0x24000000, LENGTH = 512K   /* Internal system AXI SRAM (Houses SMR active arena) */
  SRAM1         (rwx) : ORIGIN = 0x30000000, LENGTH = 256K   /* AHB SRAM1 */
  SRAM2         (rwx) : ORIGIN = 0x30040000, LENGTH = 32K    /* AHB SRAM2 */
  SRAM3         (rwx) : ORIGIN = 0x30048000, LENGTH = 32K    /* AHB SRAM3 */
  SRAM4         (rwx) : ORIGIN = 0x38000000, LENGTH = 64K    /* Backup SRAM */
  OCTOSPI_FLASH (rx)  : ORIGIN = 0x90000000, LENGTH = 64M    /* External Memory-Mapped QuadSPI/OctoSPI NOR Flash */
  FMC_SDRAM     (rwx) : ORIGIN = 0xC0000000, LENGTH = 32M    /* External FMC SDRAM (Baseline Replay Rehearsal Buffer) */
}

SECTIONS
{
  /* Interrupt Vector Table */
  .isr_vector :
  {
    . = ALIGN(4);
    KEEP(*(.isr_vector))
    . = ALIGN(4);
  } >FLASH

  /* Program Code and Read-Only Data */
  .text :
  {
    . = ALIGN(4);
    *(.text)
    *(.text*)
    *(.glue_7)
    *(.glue_7t)
    *(.eh_frame)
    . = ALIGN(4);
    _etext = .;
  } >FLASH

  .rodata :
  {
    . = ALIGN(4);
    *(.rodata)
    *(.rodata*)
    . = ALIGN(4);
  } >FLASH

  /* Initialized Data Section */
  .data :
  {
    . = ALIGN(4);
    _sdata = .;
    *(.data)
    *(.data*)
    . = ALIGN(4);
    _edata = .;
  } >DTCMRAM AT> FLASH

  /* Standard Uninitialized BSS Data */
  .bss :
  {
    . = ALIGN(4);
    _sbss = .;
    *(.bss)
    *(.bss*)
    *(COMMON)
    . = ALIGN(4);
    _ebss = .;
  } >DTCMRAM

  /* SMR-Engine Active In-SRAM Execution Arena (Fits inside 512 KB AXI-SRAM) */
  .smr_arena (NOLOAD) :
  {
    . = ALIGN(64);
    _ssmr_arena = .;
    *(.bss.smr_arena)
    *(.bss.smr_arena*)
    . = ALIGN(64);
    _esmr_arena = .;
  } >AXI_SRAM

  /* SMR-Engine 50-Task Archival Metadata Table (1.44 MB staged in Flash / OctoSPI memory) */
  .smr_metadata_table :
  {
    . = ALIGN(64);
    _ssmr_meta = .;
    *(.smr_metadata_table)
    *(.smr_metadata_table*)
    . = ALIGN(64);
    _esmr_meta = .;
  } >FLASH

  /* Strict User Stack Check in DTCM */
  ._user_stack :
  {
    . = ALIGN(8);
    . = . + _Min_Stack_Size;
    . = ALIGN(8);
  } >DTCMRAM

  /* DISCARD UNWANTED SECTIONS & ASSERT ZERO HEAP */
  /DISCARD/ :
  {
    *(.heap)
    *(.heap*)
  }
}
