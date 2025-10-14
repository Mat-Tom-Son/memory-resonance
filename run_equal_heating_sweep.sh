#!/bin/bash
# Generate equal_heating sweep for quantum pseudomode model
# Matching the theta values from equal_carrier sweep

set -e

THETA_VALUES=(0.3 0.44 0.65 0.7 0.8 0.9 0.95 1.0 1.1 1.2 1.3 1.39 2.04 3.0)
OUTPUT_BASE="results/quantum_eqheat_sweep"

echo "Starting equal_heating sweep for ${#THETA_VALUES[@]} theta values..."
echo "Output directory: $OUTPUT_BASE"

for theta in "${THETA_VALUES[@]}"; do
    echo ""
    echo "=================================================="
    echo "Running Θ = $theta"
    echo "=================================================="

    .venv/bin/python3 stage3_parameter_sweep.py \
        --theta "$theta" \
        --calibrations equal_heating \
        --cutoffs 4 \
        --hierarchy_cutoff 5 \
        --nbar 0.02 \
        --omega_c_scale 1.0 \
        --t_final 8.0 \
        --n_time 300 \
        --quant_engine gaussian \
        --psd_norm onesided \
        --output_dir "${OUTPUT_BASE}_theta_$(printf '%.2f' $theta)" \
        --tuner_tol_occ 0.01 \
        --tuner_max_nudges 20

    if [ $? -eq 0 ]; then
        echo "✓ Completed Θ = $theta"
    else
        echo "✗ Failed Θ = $theta"
        exit 1
    fi
done

echo ""
echo "=================================================="
echo "All runs complete! Consolidating results..."
echo "=================================================="

# Consolidate all summary.csv files into one
OUTPUT_CONSOLIDATED="${OUTPUT_BASE}_consolidated.csv"
FIRST=1

for theta in "${THETA_VALUES[@]}"; do
    SUMMARY_FILE="${OUTPUT_BASE}_theta_$(printf '%.2f' $theta)/summary.csv"

    if [ -f "$SUMMARY_FILE" ]; then
        if [ $FIRST -eq 1 ]; then
            # Include header from first file
            cat "$SUMMARY_FILE" > "$OUTPUT_CONSOLIDATED"
            FIRST=0
        else
            # Skip header for subsequent files
            tail -n +2 "$SUMMARY_FILE" >> "$OUTPUT_CONSOLIDATED"
        fi
    fi
done

echo "Consolidated results saved to: $OUTPUT_CONSOLIDATED"
echo "Total lines: $(wc -l < $OUTPUT_CONSOLIDATED)"
