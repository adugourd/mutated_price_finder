#!/usr/bin/env python3
"""
EVE Online Mutated Module Demand Analyzer

Analyzes zkillboard data to find which mutated (abyssal) modules are commonly
lost in highsec and lowsec, helping gauge market demand for Jita sales.

Sources:
- zkillboard API: https://github.com/zKillboard/zKillboard/wiki/API-(Killmails)
- ESI API: https://esi.evetech.net/
- Fuzzwork SDE: https://www.fuzzwork.co.uk/dump/latest/
"""

from __future__ import annotations

import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import TypedDict

import requests
import pandas as pd


class SDEData(TypedDict):
    """SDE data loaded for analysis."""
    systems: pd.DataFrame
    regions: pd.DataFrame
    types: pd.DataFrame
    groups: pd.DataFrame


class SystemInfo(TypedDict):
    """Information about a solar system."""
    security: float
    sec_class: str
    region: str


class AbyssalItem(TypedDict):
    """An abyssal item found on a killmail."""
    killmail_id: int
    type_id: int
    type_name: str
    quantity: int
    destroyed: bool
    dropped: bool
    system_id: int
    sec_class: str
    security: float
    flag: int


class RegionResults(TypedDict):
    """Results from analyzing a region's killmails."""
    abyssal_items: list[AbyssalItem]
    killmail_count: int
    abyssal_killmail_count: int

# Cache directory
CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

# API endpoints
ZKB_API = "https://zkillboard.com/api"
ESI_API = "https://esi.evetech.net/latest"
FUZZWORK_SDE = "https://www.fuzzwork.co.uk/dump/latest"

# Request headers
HEADERS = {
    "User-Agent": "EVE Mutated Module Demand Analyzer - Contact: github.com/mutated-price-finder",
    "Accept-Encoding": "gzip",
}


def download_sde_file(filename: str, cache_hours: int = 168) -> pd.DataFrame:
    """Download and cache SDE CSV file from Fuzzwork."""
    cache_path = CACHE_DIR / filename

    # Check cache
    if cache_path.exists():
        age_hours = (datetime.now().timestamp() - cache_path.stat().st_mtime) / 3600
        if age_hours < cache_hours:
            print(f"  Using cached {filename} (age: {age_hours:.1f} hours)")
            return pd.read_csv(cache_path)

    print(f"  Downloading {filename}...")
    url = f"{FUZZWORK_SDE}/{filename}"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()

    cache_path.write_bytes(response.content)
    return pd.read_csv(cache_path)


def load_sde_data() -> SDEData:
    """Load required SDE data for system security and type info."""
    print("Loading SDE data...")

    data: SDEData = {}  # type: ignore[typeddict-item]

    # Load solar systems with security status
    systems = download_sde_file("mapSolarSystems.csv")
    data["systems"] = systems[["solarSystemID", "solarSystemName", "security", "regionID"]].copy()

    # Load regions
    regions = download_sde_file("mapRegions.csv")
    data["regions"] = regions[["regionID", "regionName"]].copy()

    # Load types to identify abyssal modules
    types = download_sde_file("invTypes.csv")
    data["types"] = types[["typeID", "typeName", "groupID"]].copy()

    # Load groups to identify module categories
    groups = download_sde_file("invGroups.csv")
    data["groups"] = groups[["groupID", "groupName", "categoryID"]].copy()

    return data


def get_security_class(security: float) -> str:
    """Classify a system by security status."""
    if security >= 0.5:
        return "highsec"
    elif security > 0.0:
        return "lowsec"
    elif security > -0.5:
        return "nullsec"
    else:
        return "wormhole"


def build_system_lookup(sde_data: SDEData) -> dict[int, SystemInfo]:
    """Build lookup tables for system security and region info."""
    systems = sde_data["systems"]
    regions = sde_data["regions"]

    # Merge region names
    systems = systems.merge(regions, on="regionID", how="left")

    # Build lookup dict: system_id -> {security, region_name, sec_class}
    lookup: dict[int, SystemInfo] = {}
    for _, row in systems.iterrows():
        lookup[int(row["solarSystemID"])] = {
            "security": row["security"],
            "sec_class": get_security_class(row["security"]),
            "region": row.get("regionName", "Unknown"),
        }

    return lookup


def find_abyssal_type_ids(sde_data: SDEData) -> dict[int, str]:
    """Find all abyssal module type IDs and their names."""
    types = sde_data["types"]

    # Abyssal modules have "Abyssal" in the name
    abyssal = types[types["typeName"].str.contains("Abyssal", case=False, na=False)]

    return dict(zip(abyssal["typeID"], abyssal["typeName"]))


def find_mutaplasmid_type_ids(sde_data: SDEData) -> dict[int, str]:
    """Find all mutaplasmid type IDs and their names."""
    types = sde_data["types"]

    # Mutaplasmids have "Mutaplasmid" in the name
    mutaplasmids = types[types["typeName"].str.contains("Mutaplasmid", case=False, na=False)]

    return dict(zip(mutaplasmids["typeID"], mutaplasmids["typeName"]))


def fetch_zkb_killmails(region_id: int, page: int = 1) -> list:
    """Fetch killmails from zkillboard for a specific region."""
    url = f"{ZKB_API}/kills/regionID/{region_id}/page/{page}/"

    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        if response.status_code == 429:
            print(f"    Rate limited, waiting 10s...")
            time.sleep(10)
            return fetch_zkb_killmails(region_id, page)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"    Error fetching zkb page {page}: {e}")
        return []


def fetch_esi_killmail(killmail_id: int, killmail_hash: str) -> dict | None:
    """Fetch full killmail details from ESI."""
    url = f"{ESI_API}/killmails/{killmail_id}/{killmail_hash}/"

    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return None




def get_highsec_lowsec_regions(sde_data: SDEData) -> list[int]:
    """Get list of region IDs that contain highsec or lowsec systems."""
    systems = sde_data["systems"]

    # Filter for highsec and lowsec (security > 0)
    hs_ls_systems = systems[systems["security"] > 0.0]

    # Get unique region IDs
    region_ids = hs_ls_systems["regionID"].unique().tolist()

    # Filter out wormhole regions (region ID >= 11000000)
    region_ids = [r for r in region_ids if r < 11000000]

    return sorted(region_ids)


# Popular PvP regions for focused scanning
POPULAR_PVP_REGIONS = {
    # Trade hub regions
    10000002: "The Forge (Jita)",
    10000043: "Domain (Amarr)",
    10000032: "Sinq Laison (Dodixie)",
    10000042: "Metropolis (Hek)",
    10000030: "Heimatar (Rens)",

    # Faction Warfare regions (lots of lowsec PvP)
    10000048: "Placid",
    10000064: "Essence",
    10000068: "Verge Vendor",
    10000016: "Lonetrek",
    10000020: "Tash-Murkon",
    10000038: "The Bleak Lands",
    10000036: "Devoid",

    # Other active lowsec regions
    10000028: "Molden Heath",
    10000052: "Kador",
    10000054: "Aridia",
}


def analyze_killmails_for_region(
    region_id: int,
    region_name: str,
    system_lookup: dict[int, SystemInfo],
    abyssal_types: dict[int, str],
    max_pages: int = 5,
    sec_filter: set[str] | None = None,
) -> RegionResults:
    """Analyze killmails for a specific region."""
    if sec_filter is None:
        sec_filter = {"highsec", "lowsec"}

    results: RegionResults = {
        "abyssal_items": [],
        "killmail_count": 0,
        "abyssal_killmail_count": 0,
    }

    print(f"  Scanning {region_name} (ID: {region_id})...")

    for page in range(1, max_pages + 1):
        killmails = fetch_zkb_killmails(region_id, page)

        if not killmails:
            break

        for km in killmails:
            killmail_id = km.get("killmail_id")
            zkb = km.get("zkb", {})
            killmail_hash = zkb.get("hash")

            if not killmail_id or not killmail_hash:
                continue

            results["killmail_count"] += 1

            # Fetch full killmail from ESI
            full_km = fetch_esi_killmail(killmail_id, killmail_hash)
            if not full_km:
                continue

            # Check system security
            system_id = full_km.get("solar_system_id")
            system_info = system_lookup.get(system_id, {})
            sec_class = system_info.get("sec_class", "unknown")

            if sec_class not in sec_filter:
                continue

            # Check victim's items for abyssal modules
            victim = full_km.get("victim", {})
            items = victim.get("items", [])

            has_abyssal = False
            for item in items:
                type_id = item.get("item_type_id")
                item_id = item.get("item_id")  # Unique instance ID for abyssal

                if type_id in abyssal_types:
                    has_abyssal = True

                    # Note: ESI killmails don't include item_id for abyssal modules,
                    # so we can only count by generic type (e.g., "Abyssal Gyrostabilizer")
                    # without knowing the specific source module or mutaplasmid used.
                    qty_destroyed = item.get("quantity_destroyed", 0)
                    qty_dropped = item.get("quantity_dropped", 0)

                    results["abyssal_items"].append({
                        "killmail_id": killmail_id,
                        "type_id": type_id,
                        "type_name": abyssal_types[type_id],
                        "quantity": qty_destroyed + qty_dropped,
                        "destroyed": qty_destroyed > 0,
                        "dropped": qty_dropped > 0,
                        "system_id": system_id,
                        "sec_class": sec_class,
                        "security": system_info.get("security", 0),
                        "flag": item.get("flag"),  # slot info
                    })

            if has_abyssal:
                results["abyssal_killmail_count"] += 1

        # Rate limiting
        time.sleep(0.5)

    return results


def format_isk(value: float) -> str:
    """Format ISK value with suffix."""
    if value >= 1_000_000_000:
        return f"{value/1_000_000_000:.2f}B"
    elif value >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"{value/1_000:.1f}K"
    return f"{value:.0f}"


def main() -> None:
    """Main entry point for the killmail demand analyzer CLI."""
    parser = argparse.ArgumentParser(
        description="Analyze zkillboard for mutated module demand in highsec/lowsec"
    )
    parser.add_argument(
        "--pages", "-p",
        type=int,
        default=3,
        help="Number of zkillboard pages to fetch per region (default: 3)"
    )
    parser.add_argument(
        "--regions", "-r",
        type=int,
        nargs="+",
        default=None,
        help="Specific region IDs to scan (default: all highsec/lowsec)"
    )
    parser.add_argument(
        "--highsec-only",
        action="store_true",
        help="Only analyze highsec kills"
    )
    parser.add_argument(
        "--lowsec-only",
        action="store_true",
        help="Only analyze lowsec kills"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output CSV file for detailed results"
    )
    parser.add_argument(
        "--popular",
        action="store_true",
        help="Focus on popular PvP regions (trade hubs, faction warfare)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("EVE Online Mutated Module Demand Analyzer")
    print("=" * 70)

    # Load SDE data
    sde_data = load_sde_data()

    # Build lookups
    print("\nBuilding lookup tables...")
    system_lookup = build_system_lookup(sde_data)
    abyssal_types = find_abyssal_type_ids(sde_data)

    print(f"  Found {len(abyssal_types)} abyssal module types")
    print(f"  Loaded {len(system_lookup)} solar systems")

    # Determine regions to scan
    if args.regions:
        region_ids = args.regions
    elif args.popular:
        region_ids = list(POPULAR_PVP_REGIONS.keys())
        print(f"\nUsing {len(region_ids)} popular PvP regions")
    else:
        region_ids = get_highsec_lowsec_regions(sde_data)

    # Determine security filter
    sec_filter = {"highsec", "lowsec"}
    if args.highsec_only:
        sec_filter = {"highsec"}
    elif args.lowsec_only:
        sec_filter = {"lowsec"}

    # Get region names
    regions_df = sde_data["regions"]
    region_names = dict(zip(regions_df["regionID"], regions_df["regionName"]))

    print(f"\nScanning {len(region_ids)} regions for {'/'.join(sec_filter)} kills...")
    print(f"Pages per region: {args.pages}")

    # Analyze killmails
    all_abyssal_items = []
    total_killmails = 0
    total_abyssal_killmails = 0

    for region_id in region_ids:
        region_name = region_names.get(region_id, f"Region {region_id}")

        results = analyze_killmails_for_region(
            region_id=region_id,
            region_name=region_name,
            system_lookup=system_lookup,
            abyssal_types=abyssal_types,
            max_pages=args.pages,
            sec_filter=sec_filter,
        )

        all_abyssal_items.extend(results["abyssal_items"])
        total_killmails += results["killmail_count"]
        total_abyssal_killmails += results["abyssal_killmail_count"]

        if results["abyssal_items"]:
            print(f"    Found {len(results['abyssal_items'])} abyssal modules")

    # Aggregate results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nTotal killmails scanned: {total_killmails}")
    print(f"Killmails with abyssal modules: {total_abyssal_killmails}")
    print(f"Total abyssal modules found: {len(all_abyssal_items)}")

    if not all_abyssal_items:
        print("\nNo abyssal modules found in the scanned killmails.")
        return

    # Create DataFrame for analysis
    df = pd.DataFrame(all_abyssal_items)

    # Filter to only FITTED modules (flags 11-34 are module slots)
    # Flag 5 is cargo, 87+ are special holds
    fitted_modules = df[df["flag"].between(11, 34)]

    print(f"\nFitted abyssal modules (in module slots): {len(fitted_modules)}")
    print(f"Cargo/other abyssal items: {len(df) - len(fitted_modules)}")

    # Count by abyssal type (fitted only)
    print("\n" + "=" * 60)
    print("TOP ABYSSAL MODULES LOST (fitted modules only)")
    print("=" * 60)
    type_counts = fitted_modules.groupby("type_name").size().sort_values(ascending=False)
    for type_name, count in type_counts.head(25).items():
        print(f"  {count:>4}x  {type_name}")

    # Also show cargo items if interesting
    cargo_items = df[~df["flag"].between(11, 34)]
    if not cargo_items.empty:
        print("\n--- Top Abyssal Items in Cargo ---")
        cargo_counts = cargo_items.groupby("type_name").size().sort_values(ascending=False)
        for type_name, count in cargo_counts.head(10).items():
            print(f"  {count:>4}x  {type_name}")

    # Security breakdown
    print("\n--- Fitted Modules by Security Class ---")
    sec_counts = fitted_modules.groupby("sec_class").size()
    for sec_class, count in sec_counts.items():
        pct = count / len(fitted_modules) * 100
        print(f"  {sec_class}: {count} ({pct:.1f}%)")

    # Destruction rate
    destroyed = fitted_modules[fitted_modules["destroyed"]].shape[0]
    dropped = fitted_modules[fitted_modules["dropped"]].shape[0]
    print(f"\n--- Destruction vs Drop Rate ---")
    print(f"  Destroyed: {destroyed} ({destroyed/len(fitted_modules)*100:.1f}%)")
    print(f"  Dropped:   {dropped} ({dropped/len(fitted_modules)*100:.1f}%)")

    # Save to CSV if requested
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
