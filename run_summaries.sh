#!/bin/bash

set -e

# XRP-USDT LINK-USDT FLOCK-USDT BCH-USDT USELESS-USDT INJ-USDT UAI-USDT IOST-USDT LTC-USDT FIL-USDT UNI-USDT
SYMBOLS=(
    "SUI-USDT"
    "BTC-USDT"
    "DOGE-USDT"
    "XRP-USDT"
    "LINK-USDT"
    "FLOCK-USDT"
    "BCH-USDT"
    "USELESS-USDT"
    "INJ-USDT"
    "UAI-USDT"
    "IOST-USDT"
    "LTC-USDT"
    "FIL-USDT"
    "UNI-USDT"
)

for SYMBOL in "${SYMBOLS[@]}"; do
    DIR="data/$SYMBOL/lead_quote"

    if [ ! -d "$DIR" ]; then
        echo "[$SYMBOL] ERROR: directory not found: $DIR"
        continue
    fi

    LATEST=$(ls -t "$DIR" | head -1)

    if [ -z "$LATEST" ]; then
        echo "[$SYMBOL] ERROR: no files found in $DIR"
        continue
    fi

    echo ""
    echo "========================================"
    echo "$SYMBOL"
    echo "File: $DIR/$LATEST"
    echo "========================================"

    python backend/run_lead_quote.py --summary "$DIR/$LATEST"
done