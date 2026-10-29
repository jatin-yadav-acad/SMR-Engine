#!/usr/bin/env bash
# ==============================================================================
# SMR-Engine: Bare-Metal Profiling Harness on AMD Ryzen 9 9900X
# Architecture: Zen 5 (12 Cores / 24 Threads, 64 MB L3, Dual CCD, AVX-512)
# Telemetry: Linux RAPL Energy (sysfs) + Linux perf hardware cache counters
# ==============================================================================

set -e
OUTPUT_DIR="./results_zen5"
mkdir -p "$OUTPUT_DIR"

echo "=== SMR-Engine vs DER++ Profiling on AMD Ryzen 9 9900X ==="

# 1. Isolate physical cores on CCD0 to eliminate Infinity Fabric interconnect jitter
if command -v systemctl &> /dev/null; then
    sudo systemctl set-property --runtime -- user.slice AllowedCPUs=0,1,4-23 || true
fi
if command -v cpupower &> /dev/null; then
    sudo cpupower frequency-set -g performance || true
fi

# 2. Check and enable Linux RAPL package energy read access
RAPL_PKG="/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"
if [ -f "$RAPL_PKG" ]; then
    echo "Linux RAPL sysfs interface detected: $RAPL_PKG"
    if [ ! -r "$RAPL_PKG" ]; then
        sudo chmod 444 /sys/class/powercap/intel-rapl/intel-rapl:*/energy_uj || true
    fi
else
    echo "Notice: Non-native Linux environment or RAPL virtualized."
fi

# 3. Profile SMR-Engine Forward Recalibration (K=20 microbatches)
echo "[1/2] Profiling SMR-Engine Forward Recalibration (AVX-512 SIMD)..."
if command -v perf &> /dev/null; then
    taskset -c 2 perf stat -r 10 \
        -e power/energy-pkg/ \
        -e cycles,instructions,cache-misses,cache-references \
        -e L1-dcache-load-misses,L1-dcache-loads \
        -e LLC-load-misses,LLC-loads \
        -o "${OUTPUT_DIR}/smr_recal_perf.txt" \
        ./smr_zen5_runner --mode recal --k 20 --batch 128 || ./smr_zen5_runner.exe --mode recal --k 20 --batch 128
else
    ./smr_zen5_runner --mode recal --k 20 --batch 128 2>&1 | tee "${OUTPUT_DIR}/smr_recal_output.txt" || \
    ./smr_zen5_runner.exe --mode recal --k 20 --batch 128
fi

# 4. Profile Rehearsal Replay Step (DER++ Exemplar stream)
echo "[2/2] Profiling Rehearsal Replay Step (DER++ 2000 Exemplars)..."
if command -v perf &> /dev/null; then
    taskset -c 2 perf stat -r 10 \
        -e power/energy-pkg/ \
        -e cycles,instructions,cache-misses,cache-references \
        -e L1-dcache-load-misses,L1-dcache-loads \
        -e LLC-load-misses,LLC-loads \
        -o "${OUTPUT_DIR}/der_rehearsal_perf.txt" \
        ./smr_zen5_runner --mode replay --buffer 2000 --batch 128 || ./smr_zen5_runner.exe --mode replay --buffer 2000 --batch 128
else
    ./smr_zen5_runner --mode replay --buffer 2000 --batch 128 2>&1 | tee "${OUTPUT_DIR}/der_rehearsal_output.txt" || \
    ./smr_zen5_runner.exe --mode replay --buffer 2000 --batch 128
fi

echo "=== AMD Ryzen 9 9900X Profiling Complete ==="
