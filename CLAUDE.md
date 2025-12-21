# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EVE Online Mutated Module Price Finder - a tool to find the lowest contract prices for equivalent mutated (abyssal) modules by comparing dogma attributes.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run with default values (your Republic Fleet Gyro stats)
python find_prices.py

# Run with custom stats
python find_prices.py -d 1.145 -r 12.49 -c 18.25

# Filter for Republic Fleet Gyro mutations only
python find_prices.py -rf

# Show both worse and better items for comparison
python find_prices.py --show-better
```

## Architecture

- `find_prices.py` - Main script that:
  1. Downloads EVE Ref public contract data (cached for 30 minutes)
  2. Extracts mutated items and their dogma attributes from CSV files
  3. Filters by source type (e.g., Republic Fleet Gyrostabilizer)
  4. Compares damage modifier, ROF bonus, and CPU usage
  5. Outputs pricing recommendations

## Data Source

Uses [EVE Ref Public Contract Snapshots](https://data.everef.net/public-contracts/) which includes dynamic item dogma attributes. Data is updated every 30 minutes.

## Key EVE Dogma Attribute IDs

- `64` - damageMultiplier (Damage Modifier)
- `204` - speedMultiplier (Rate of Fire)
- `50` - cpu (CPU Usage)
