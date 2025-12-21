#!/usr/bin/env python3
"""
EVE Online Mutated Module ROI Calculator

Calculates expected ROI for rolling mutated modules based on:
1. Base item cost (Jita sell orders)
2. Mutaplasmid cost (Jita sell orders)
3. Success rate based on probability of rolling "good enough" stats
4. Expected sale price based on contract distribution

Uses Monte Carlo simulation to calculate success probability where:
- Primary stats need to be >= median threshold (e.g., DPS for damage mods)
- Secondary stats must NOT be catastrophic (not in worst 10% of rolls)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TypedDict, Any

import pandas as pd
import numpy as np


class StatAnalysisResult(TypedDict, total=False):
    """Result from stat-based contract analysis."""
    n_total: int
    n_sellable: int
    n_outliers: int
    n_used: int
    expected_price: float
    method: str
    base_stat: float
    max_stat: float
    midpoint_stat: float
    anchor_stat: float
    anchor_price: float
    slope: float
    data_min_stat: float
    data_max_stat: float
    coverage_pct: float


class ContractDistribution(TypedDict):
    """Distribution of contracts for a target."""
    count: int
    stats: np.ndarray
    prices: np.ndarray
    stat_analysis: StatAnalysisResult

# Import from refactored modules
from src.config.loader import load_all_targets, load_constants, load_attributes
from src.data.fuzzwork import get_jita_prices
from src.data.everef import download_contract_archive, extract_csv_from_archive
from src.analysis.monte_carlo import get_success_rate
from src.analysis.regression import fit_constrained_regression
from src.analysis.risk import calculate_roi, calculate_bankroll_risk
from src.formatters.isk import format_isk
from src.formatters.stats import format_stat

# Load configuration
_constants = load_constants()
_attributes = load_attributes()

# Configuration values
NUM_SAMPLES = _constants['simulation']['num_samples']
ATTR_DAMAGE = _attributes['damage']

# Load roll targets from YAML
ROLL_TARGETS = load_all_targets()


def analyze_contracts_stat_based(
    archive_path: Path,
    muta_type_id: int,
    primary_attr_id: int,
    base_stat: float,
    max_stat: float,
) -> ContractDistribution:
    """
    Analyze contract prices using stat-based filtering.

    Looks at ALL mutated items with the given mutaplasmid (any source type),
    filters by actual stat values, and uses regression-based pricing.
    """
    contracts = extract_csv_from_archive(archive_path, 'contracts.csv')
    dynamic_items = extract_csv_from_archive(archive_path, 'contract_dynamic_items.csv')
    dogma_attrs = extract_csv_from_archive(archive_path, 'contract_dynamic_items_dogma_attributes.csv')

    # Filter for item exchange contracts
    item_contracts = contracts[contracts['type'] == 'item_exchange'][['contract_id', 'price']].copy()

    # Get ALL items with this mutaplasmid (any source type)
    matching_items = dynamic_items[dynamic_items['mutator_type_id'] == muta_type_id].copy()

    # Only keep contracts with exactly 1 dynamic item (no bundles)
    total_items_per_contract = dynamic_items.groupby('contract_id').size()
    single_item_contracts = total_items_per_contract[total_items_per_contract == 1].index
    matching_items = matching_items[matching_items['contract_id'].isin(single_item_contracts)]

    if matching_items.empty:
        return {
            'count': 0,
            'stats': np.array([]),
            'prices': np.array([]),
            'stat_analysis': {
                'n_total': 0,
                'n_sellable': 0,
                'n_outliers': 0,
                'n_used': 0,
                'expected_price': 0,
                'method': 'no_data',
            },
        }

    # Join with dogma attributes to get primary stat value
    primary_attrs = dogma_attrs[dogma_attrs['attribute_id'] == primary_attr_id][['item_id', 'value']].copy()
    primary_attrs = primary_attrs.rename(columns={'value': 'stat_value'})

    items_with_stats = matching_items.merge(primary_attrs, on='item_id', how='inner')

    # Join with prices
    items_with_prices = items_with_stats.merge(item_contracts, on='contract_id', how='inner')

    if items_with_prices.empty:
        return {
            'count': 0,
            'stats': np.array([]),
            'prices': np.array([]),
            'stat_analysis': {
                'n_total': 0,
                'n_sellable': 0,
                'n_outliers': 0,
                'n_used': 0,
                'expected_price': 0,
                'method': 'no_stats',
            },
        }

    stats = items_with_prices['stat_value'].values
    prices = items_with_prices['price'].values

    # Use shared regression function
    stat_analysis = fit_constrained_regression(stats, prices, base_stat, max_stat)

    return {
        'count': len(stats),
        'stats': stats,
        'prices': prices,
        'stat_analysis': stat_analysis,
    }


def main() -> None:
    """Main entry point for the ROI calculator CLI."""
    parser = argparse.ArgumentParser(
        description="Calculate ROI for rolling mutated modules"
    )
    parser.add_argument(
        '--target', '-t',
        choices=list(ROLL_TARGETS.keys()),
        default=None,
        help='Specific target to analyze (default: all)'
    )
    parser.add_argument(
        '--bankroll', '-b',
        type=float,
        default=1_000_000_000,
        help='Bankroll in ISK for risk analysis (default: 1B)'
    )
    parser.add_argument(
        '--confidence', '-c',
        type=float,
        default=95.0,
        help='Minimum probability of profit %% for "safe" items (default: 95)'
    )
    parser.add_argument(
        '--max-sells', '-m',
        type=int,
        default=5,
        help='Maximum number of successful sales to reach profit (default: 5)'
    )

    args = parser.parse_args()
    bankroll = args.bankroll
    confidence_threshold = args.confidence / 100
    max_sells = args.max_sells

    print("=" * 70)
    print("EVE Online Mutated Module ROI Calculator")
    print("=" * 70)
    print(f"\nUsing {NUM_SAMPLES:,} Monte Carlo samples per item")
    print(f"Bankroll: {format_isk(bankroll)} | Confidence: {args.confidence:.0f}% | Max sells: {max_sells}")

    # Get targets to analyze
    if args.target:
        targets = {args.target: ROLL_TARGETS[args.target]}
    else:
        targets = ROLL_TARGETS

    # Collect all type IDs we need prices for
    all_type_ids = set()
    for target in targets.values():
        all_type_ids.add(target.base_type_id)
        all_type_ids.add(target.muta_type_id)

    # Get market prices
    print("\nFetching Jita market prices...")
    prices = get_jita_prices(list(all_type_ids))

    # Download contract data
    archive_path = download_contract_archive()

    # Analyze each target
    results = []

    print("\nAnalyzing each item...")
    for key, target in targets.items():
        print(f"  {target.name}...")

        # Get costs
        base_price = prices[target.base_type_id]['min']
        muta_price = prices[target.muta_type_id]['min']
        roll_cost = base_price + muta_price

        # Get primary attribute info from muta_ranges
        primary_range = target.muta_ranges[0]
        base_stat = target.base_stats.get(primary_range.attr_id, 0)
        max_stat = base_stat * primary_range.max_mult

        # Analyze contracts
        dist = analyze_contracts_stat_based(
            archive_path,
            target.muta_type_id,
            primary_range.attr_id,
            base_stat,
            max_stat,
        )

        if dist['count'] == 0:
            print(f"    No contracts found, skipping...")
            continue

        stat_analysis = dist['stat_analysis']
        realistic_sale_price = stat_analysis['expected_price']

        # Calculate success rate using shared module
        success_result = get_success_rate(target)
        success_rate = success_result.success_rate
        p_primary = success_result.p_primary
        p_secondary = success_result.p_secondary
        threshold = success_result.threshold

        # Build price analysis dict for display
        price_analysis = {
            **stat_analysis,
            'base_stat': base_stat,
            'max_stat': max_stat,
            'stat_based': True,
            'module_type': target.module_type,
            'primary_attr_id': primary_range.attr_id,
        }

        # Calculate ROI using shared module
        roi = calculate_roi(roll_cost, success_rate, realistic_sale_price)

        # Calculate risk metrics using shared module
        risk = calculate_bankroll_risk(success_rate, roll_cost, realistic_sale_price, bankroll)

        results.append({
            'key': key,
            'target': target,
            'base_price': base_price,
            'muta_price': muta_price,
            'roll_cost': roll_cost,
            'distribution': dist,
            'price_analysis': price_analysis,
            'roi': roi,
            'risk': risk,
            'success_rate': success_rate,
            'p_primary': p_primary,
            'p_secondary': p_secondary,
            'threshold': threshold,
            'sale_price': realistic_sale_price,
        })

    # Sort by expected profit per roll
    results.sort(key=lambda x: x['roi']['expected_profit'], reverse=True)

    # Display results
    print("\n" + "=" * 70)
    print("ROI ANALYSIS RESULTS (sorted by Profit/Roll)")
    print("=" * 70)

    for r in results:
        target = r['target']
        dist = r['distribution']
        pa = r['price_analysis']
        roi = r['roi']

        print(f"\n{'-' * 70}")
        print(f"  {target.name}")
        print(f"{'-' * 70}")

        print(f"\n  COSTS:")
        print(f"    Base item:   {format_isk(r['base_price']):>12}  ({target.base_name})")
        print(f"    Mutaplasmid: {format_isk(r['muta_price']):>12}  ({target.muta_name})")
        print(f"    Roll cost:   {format_isk(r['roll_cost']):>12}")

        module_type = pa.get('module_type', 'dps')
        attr_id = pa.get('primary_attr_id', ATTR_DAMAGE)

        print(f"\n  CONTRACT ANALYSIS ({dist['count']} contracts, stat-based):")
        base_fmt = format_stat(pa['base_stat'], module_type, attr_id)
        max_fmt = format_stat(pa['max_stat'], module_type, attr_id)
        mid_fmt = format_stat(pa['midpoint_stat'], module_type, attr_id)
        print(f"    Stat range:    [{base_fmt} -> {max_fmt}]")
        print(f"    Midpoint:      {mid_fmt}")
        print(f"    Total items:               {pa['n_total']:>3}")
        print(f"    Below base (unsellable):   {pa['n_total'] - pa['n_sellable']:>3} (discarded)")
        print(f"    Price outliers (IQR):      {pa['n_outliers']:>3} (discarded)")
        print(f"    Used for regression:       {pa['n_used']:>3}")
        print(f"    Method:        {pa['method']}")
        if pa['method'] == 'constrained_regression':
            anchor_fmt = format_stat(pa['anchor_stat'], module_type, attr_id)
            print(f"    Anchor point:  ({anchor_fmt}, {format_isk(pa['anchor_price'])})")
            print(f"    Slope:         {format_isk(pa['slope'])}/unit stat")
            coverage = pa.get('coverage_pct', 0)
            data_min = pa.get('data_min_stat', 0)
            data_max = pa.get('data_max_stat', 0)
            data_min_fmt = format_stat(data_min, module_type, attr_id)
            data_max_fmt = format_stat(data_max, module_type, attr_id)
            coverage_warning = " ** LOW CONFIDENCE **" if coverage < 50 else ""
            print(f"    Data coverage: [{data_min_fmt} -> {data_max_fmt}] = {coverage:.0f}%{coverage_warning}")
        print(f"    Exp. price at midpoint: {format_isk(pa['expected_price']):>12}")

        print(f"\n  SUCCESS PROBABILITY (Monte Carlo):")
        print(f"    Primary:     {target.primary_desc}")
        print(f"                 {r['p_primary']*100:>10.1f}%")
        print(f"    Secondary:   {target.secondary_desc}")
        print(f"                 {r['p_secondary']*100:>10.1f}%")
        print(f"    Sellable:    {r['success_rate']*100:>10.1f}%  (stat > base + OK secondary)")

        print(f"\n  ROI ANALYSIS:")
        print(f"    Exp. sale:     {format_isk(pa['expected_price']):>12}  (regression midpoint)")
        print(f"    Exp. value:    {format_isk(roi['expected_value']):>12}  per roll")
        print(f"    Exp. profit:   {format_isk(roi['expected_profit']):>12}  per roll")
        print(f"    ROI:           {roi['roi_pct']:>11.1f}%")
        print(f"    Rolls/success: {roi['rolls_per_success']:>11.1f}")

        # Verdict
        if roi['roi_pct'] > 100:
            verdict = "EXCELLENT"
        elif roi['roi_pct'] > 50:
            verdict = "GOOD"
        elif roi['roi_pct'] > 0:
            verdict = "MARGINAL"
        else:
            verdict = "AVOID"

        coverage = pa.get('coverage_pct', 100)
        if coverage < 50:
            verdict += " (LOW CONFIDENCE)"

        print(f"\n  VERDICT: {verdict}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - RANKED BY PROFIT PER ROLL")
    print("=" * 70)
    print(f"\n{'Rank':<4} {'Profit/Roll':>12} {'ROI':>8} {'Success':>8} {'Coverage':>8} {'Roll Cost':>12}  Item")
    print("-" * 90)

    for i, r in enumerate(results[:10], 1):
        roi = r['roi']
        pa = r['price_analysis']
        profit = roi['expected_profit']
        profit_str = format_isk(profit) if profit >= 0 else f"-{format_isk(abs(profit))}"
        coverage = pa.get('coverage_pct', 100)
        coverage_str = f"{coverage:.0f}%"
        low_conf = "*" if coverage < 50 else ""
        print(f"{i:<4} {profit_str:>12} {roi['roi_pct']:>7.1f}% {r['success_rate']*100:>7.1f}% {coverage_str:>8}{low_conf} {format_isk(r['roll_cost']):>11}  {r['target'].name}")

    if any(r['price_analysis'].get('coverage_pct', 100) < 50 for r in results[:10]):
        print("\n* = Low confidence (data coverage < 50%)")

    # Risk-Adjusted Analysis
    print("\n" + "=" * 70)
    print(f"RISK ANALYSIS - {format_isk(bankroll)} BANKROLL")
    print("=" * 70)
    print(f"\nItems with >= {args.confidence:.0f}% probability of profit AND <= {max_sells} sales to profit:")
    print(f"(Practical constraint: you don't want to roll/sell hundreds of items)\n")

    # Filter to "safe" items
    safe_items = [
        r for r in results
        if r['risk']['prob_profitable'] >= confidence_threshold
        and r['risk']['min_successes_needed'] <= max_sells
    ]
    safe_items.sort(key=lambda x: x['risk']['expected_profit'], reverse=True)

    if not safe_items:
        print(f"  No items meet both criteria with {format_isk(bankroll)} bankroll.")
        print(f"  Try: increasing bankroll (-b), lowering confidence (-c), or raising max-sells (-m)")
    else:
        print(f"{'Rank':<4} {'P(Profit)':>9} {'Rolls':>6} {'Need':>5} {'E[Profit]':>12} {'5th%':>10} {'Median':>10} {'ROI/Roll':>9}  Item")
        print("-" * 105)

        for i, r in enumerate(safe_items[:15], 1):
            risk = r['risk']
            roi = r['roi']
            prob_pct = risk['prob_profitable'] * 100
            n_rolls = risk['n_rolls']
            need = risk['min_successes_needed']
            exp_profit = risk['expected_profit']
            p5 = risk['profit_at_5pct']
            p50 = risk['profit_at_50pct']

            print(f"{i:<4} {prob_pct:>8.1f}% {n_rolls:>6} {need:>5} {format_isk(exp_profit):>12} {format_isk(p5):>10} {format_isk(p50):>10} {roi['roi_pct']:>8.1f}%  {r['target'].name}")

        best = safe_items[0]
        risk = best['risk']
        print(f"\n  RECOMMENDED: {best['target'].name}")
        print(f"  With {format_isk(bankroll)} you can do {risk['n_rolls']} rolls")
        print(f"  Need {risk['min_successes_needed']}/{risk['n_rolls']} successes to profit (breakeven: {risk['breakeven_k']:.2f})")
        print(f"  Probability of profit: {risk['prob_profitable']*100:.1f}%")
        print(f"  Expected profit: {format_isk(risk['expected_profit'])}")
        print(f"  Worst case (5th pct): {format_isk(risk['profit_at_5pct'])}")
        print(f"  Median outcome: {format_isk(risk['profit_at_50pct'])}")
        print(f"  Best case (95th pct): {format_isk(risk['profit_at_95pct'])}")

    # High confidence but too many sells
    too_many_sells = [
        r for r in results
        if r['risk']['prob_profitable'] >= confidence_threshold
        and r['risk']['min_successes_needed'] > max_sells
    ]
    too_many_sells.sort(key=lambda x: x['risk']['expected_profit'], reverse=True)

    if too_many_sells:
        print(f"\n" + "-" * 70)
        print(f"HIGH CONFIDENCE BUT TOO MANY SELLS (need > {max_sells} sales):")
        print("-" * 70)
        print(f"{'P(Profit)':>9} {'Rolls':>6} {'Need':>5} {'ROI/Roll':>9}  Item")

        for r in too_many_sells[:5]:
            risk = r['risk']
            roi = r['roi']
            print(f"{risk['prob_profitable']*100:>8.1f}% {risk['n_rolls']:>6} {risk['min_successes_needed']:>5} {roi['roi_pct']:>8.1f}%  {r['target'].name}")

        print(f"\n  These are statistically safe but require too many rolls/sales.")
        print(f"  Use -m to increase max-sells if you have time for high-volume trading.")

    # Risky high-ROI items
    risky_items = [
        r for r in results
        if r['risk']['prob_profitable'] < confidence_threshold
        and r['roi']['roi_pct'] > 50
        and r['risk']['min_successes_needed'] <= max_sells
    ]
    risky_items.sort(key=lambda x: x['roi']['expected_profit'], reverse=True)

    if risky_items:
        print(f"\n" + "-" * 70)
        print(f"HIGH ROI BUT RISKY (< {args.confidence:.0f}% confidence with {format_isk(bankroll)}):")
        print("-" * 70)
        print(f"{'P(Profit)':>9} {'Rolls':>6} {'Need':>5} {'ROI/Roll':>9} {'Profit/Roll':>12}  Item")

        for r in risky_items[:5]:
            risk = r['risk']
            roi = r['roi']
            print(f"{risk['prob_profitable']*100:>8.1f}% {risk['n_rolls']:>6} {risk['min_successes_needed']:>5} {roi['roi_pct']:>8.1f}% {format_isk(roi['expected_profit']):>12}  {r['target'].name}")

        print(f"\n  These items have great ROI but you can't do enough rolls with {format_isk(bankroll)}")
        print(f"  to reliably overcome RNG variance. Consider a larger bankroll.")


if __name__ == '__main__':
    main()
