#!/usr/bin/env python3
"""
EVE Online Hangar Checker - Evaluate your abyssal modules

This script authenticates with EVE's ESI API to read your character's assets
and evaluate abyssal modules by fetching their rolled stats.

Setup:
1. Go to https://developers.eveonline.com/applications
2. Create a new application with:
   - Name: Whatever you want
   - Connection type: Authentication & API Access
   - Permissions (scopes):
     - esi-assets.read_assets.v1
   - Callback URL: http://localhost:8080/callback
3. Copy your Client ID and Secret Key
4. Run this script with your credentials

Usage:
    python check_hangar.py --client-id YOUR_CLIENT_ID --client-secret YOUR_SECRET

Or set environment variables:
    set EVE_CLIENT_ID=your_client_id
    set EVE_CLIENT_SECRET=your_secret
    python check_hangar.py
"""

from __future__ import annotations

import os
import sys
import json
import webbrowser
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlencode, urlparse, parse_qs
from pathlib import Path
from datetime import datetime
from typing import TypedDict, ClassVar

import requests


class TokenData(TypedDict, total=False):
    """OAuth token data."""
    access_token: str
    refresh_token: str
    expires_in: int
    expires_at: float
    character_id: int
    character_name: str


class HeatsinkEvaluation(TypedDict):
    """Evaluation result for an abyssal heatsink."""
    damage_mult: float
    rof_mult: float
    cpu: float
    dps_mult: float
    base_dps_mult: float
    dps_change_pct: float
    damage_change_pct: float
    rof_change_pct: float
    cpu_change_pct: float
    quality: str
    source_type_id: int | None
    mutator_type_id: int | None

# ESI endpoints
ESI_AUTH_URL = "https://login.eveonline.com/v2/oauth/authorize"
ESI_TOKEN_URL = "https://login.eveonline.com/v2/oauth/token"
ESI_VERIFY_URL = "https://esi.evetech.net/verify/"
ESI_BASE = "https://esi.evetech.net/latest"

# Scopes we need
SCOPES = [
    "esi-assets.read_assets.v1",
]

# Cache directory
CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
TOKEN_FILE = CACHE_DIR / "esi_token.json"

# Attribute IDs for heatsinks
ATTR_DAMAGE = 64
ATTR_ROF = 204
ATTR_CPU = 50

# Base stats for Imperial Navy Heat Sink
IN_HEATSINK_BASE = {
    'type_id': 15810,
    'name': 'Imperial Navy Heat Sink',
    ATTR_DAMAGE: 1.12,  # Damage modifier
    ATTR_ROF: 0.89,     # ROF multiplier (lower = faster)
    ATTR_CPU: 20,       # CPU usage
}

# Abyssal Heat Sink type ID
ABYSSAL_HEATSINK_TYPE_ID = 47745  # "Abyssal Heat Sink"


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """Handle OAuth callback from EVE login."""

    code: ClassVar[str | None] = None

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        if 'code' in query:
            OAuthCallbackHandler.code = query['code'][0]
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'''
                <html><body>
                <h1>Authorization successful!</h1>
                <p>You can close this window and return to the terminal.</p>
                </body></html>
            ''')
        else:
            self.send_response(400)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'<html><body><h1>Authorization failed</h1></body></html>')

    def log_message(self, format: str, *args: object) -> None:
        pass  # Suppress logging


def authenticate(client_id: str, client_secret: str, callback_port: int = 8080) -> TokenData:
    """Perform OAuth2 authentication with EVE Online."""

    callback_url = f"http://localhost:{callback_port}/callback"

    # Build authorization URL
    auth_params = {
        'response_type': 'code',
        'redirect_uri': callback_url,
        'client_id': client_id,
        'scope': ' '.join(SCOPES),
        'state': 'hangar_checker',
    }

    auth_url = f"{ESI_AUTH_URL}?{urlencode(auth_params)}"

    print("\n" + "=" * 60)
    print("EVE ONLINE AUTHENTICATION")
    print("=" * 60)
    print("\nOpening browser for EVE login...")
    print("If browser doesn't open, visit this URL manually:")
    print(f"\n{auth_url}\n")

    # Start local server to catch callback
    server = HTTPServer(('localhost', callback_port), OAuthCallbackHandler)
    server.timeout = 120  # 2 minute timeout

    # Open browser
    webbrowser.open(auth_url)

    print("Waiting for authorization (timeout: 2 minutes)...")

    # Wait for callback
    OAuthCallbackHandler.code = None
    while OAuthCallbackHandler.code is None:
        server.handle_request()

    auth_code = OAuthCallbackHandler.code
    server.server_close()

    print("Authorization received, exchanging for token...")

    # Exchange code for token
    token_data = {
        'grant_type': 'authorization_code',
        'code': auth_code,
        'redirect_uri': callback_url,
    }

    response = requests.post(
        ESI_TOKEN_URL,
        data=token_data,
        auth=(client_id, client_secret),
        timeout=30
    )
    response.raise_for_status()
    tokens = response.json()

    # Verify token and get character info
    headers = {'Authorization': f"Bearer {tokens['access_token']}"}
    verify_response = requests.get(ESI_VERIFY_URL, headers=headers, timeout=30)
    verify_response.raise_for_status()
    char_info = verify_response.json()

    tokens['character_id'] = char_info['CharacterID']
    tokens['character_name'] = char_info['CharacterName']
    tokens['expires_at'] = datetime.now().timestamp() + tokens['expires_in']

    # Save token
    with open(TOKEN_FILE, 'w') as f:
        json.dump(tokens, f)

    print(f"\nAuthenticated as: {char_info['CharacterName']}")

    return tokens


def load_token() -> TokenData | None:
    """Load cached token if valid."""
    if not TOKEN_FILE.exists():
        return None

    with open(TOKEN_FILE) as f:
        tokens = json.load(f)

    # Check if expired
    if datetime.now().timestamp() >= tokens.get('expires_at', 0):
        return None

    return tokens


def refresh_token(client_id: str, client_secret: str, refresh_token: str) -> TokenData:
    """Refresh an expired access token."""
    token_data = {
        'grant_type': 'refresh_token',
        'refresh_token': refresh_token,
    }

    response = requests.post(
        ESI_TOKEN_URL,
        data=token_data,
        auth=(client_id, client_secret),
        timeout=30
    )
    response.raise_for_status()
    tokens = response.json()

    # Verify and get char info
    headers = {'Authorization': f"Bearer {tokens['access_token']}"}
    verify_response = requests.get(ESI_VERIFY_URL, headers=headers, timeout=30)
    verify_response.raise_for_status()
    char_info = verify_response.json()

    tokens['character_id'] = char_info['CharacterID']
    tokens['character_name'] = char_info['CharacterName']
    tokens['expires_at'] = datetime.now().timestamp() + tokens['expires_in']

    # Save updated token
    with open(TOKEN_FILE, 'w') as f:
        json.dump(tokens, f)

    return tokens


def get_character_assets(access_token: str, character_id: int) -> list:
    """Get all character assets."""
    headers = {'Authorization': f"Bearer {access_token}"}

    all_assets = []
    page = 1

    while True:
        url = f"{ESI_BASE}/characters/{character_id}/assets/"
        params = {'page': page}

        response = requests.get(url, headers=headers, params=params, timeout=30)

        if response.status_code == 404:
            break

        response.raise_for_status()
        assets = response.json()

        if not assets:
            break

        all_assets.extend(assets)

        # Check for more pages
        total_pages = int(response.headers.get('X-Pages', 1))
        if page >= total_pages:
            break
        page += 1

    return all_assets


def get_item_dogma(type_id: int, item_id: int) -> dict | None:
    """Get dynamic dogma attributes for an abyssal item."""
    url = f"{ESI_BASE}/dogma/dynamic/items/{type_id}/{item_id}/"

    response = requests.get(url, timeout=30)

    if response.status_code == 404:
        return None

    response.raise_for_status()
    return response.json()


def evaluate_heatsink(dogma_data: dict | None) -> HeatsinkEvaluation | None:
    """Evaluate an abyssal heatsink's quality."""

    if not dogma_data:
        return None

    # Extract relevant attributes
    attrs = {}
    for attr in dogma_data.get('dogma_attributes', []):
        attrs[attr['attribute_id']] = attr['value']

    damage_mult = attrs.get(ATTR_DAMAGE, IN_HEATSINK_BASE[ATTR_DAMAGE])
    rof_mult = attrs.get(ATTR_ROF, IN_HEATSINK_BASE[ATTR_ROF])
    cpu = attrs.get(ATTR_CPU, IN_HEATSINK_BASE[ATTR_CPU])

    # Calculate DPS multiplier: damage / rof (higher is better)
    dps_mult = damage_mult / rof_mult
    base_dps_mult = IN_HEATSINK_BASE[ATTR_DAMAGE] / IN_HEATSINK_BASE[ATTR_ROF]

    # Calculate % change from base
    dps_change = ((dps_mult / base_dps_mult) - 1) * 100
    damage_change = ((damage_mult / IN_HEATSINK_BASE[ATTR_DAMAGE]) - 1) * 100
    rof_change = ((rof_mult / IN_HEATSINK_BASE[ATTR_ROF]) - 1) * 100
    cpu_change = ((cpu / IN_HEATSINK_BASE[ATTR_CPU]) - 1) * 100

    # Determine quality
    if dps_change >= 2:
        quality = "EXCELLENT"
    elif dps_change >= 1:
        quality = "GOOD"
    elif dps_change >= 0:
        quality = "DECENT"
    elif dps_change >= -1:
        quality = "POOR"
    else:
        quality = "BAD"

    return {
        'damage_mult': damage_mult,
        'rof_mult': rof_mult,
        'cpu': cpu,
        'dps_mult': dps_mult,
        'base_dps_mult': base_dps_mult,
        'dps_change_pct': dps_change,
        'damage_change_pct': damage_change,
        'rof_change_pct': rof_change,
        'cpu_change_pct': cpu_change,
        'quality': quality,
        'source_type_id': dogma_data.get('source_type_id'),
        'mutator_type_id': dogma_data.get('mutator_type_id'),
    }


def get_type_names(type_ids: list[int] | set[int]) -> dict[int, str]:
    """Get type names from ESI."""
    if not type_ids:
        return {}

    url = f"{ESI_BASE}/universe/names/"
    response = requests.post(url, json=list(type_ids), timeout=30)

    if response.status_code != 200:
        return {}

    return {item['id']: item['name'] for item in response.json()}


def main() -> int:
    """Main entry point for the hangar checker CLI."""
    parser = argparse.ArgumentParser(description="Check your EVE Online abyssal heatsinks")
    parser.add_argument('--client-id', help='EVE application client ID')
    parser.add_argument('--client-secret', help='EVE application client secret')
    parser.add_argument('--refresh', action='store_true', help='Force re-authentication')

    args = parser.parse_args()

    # Get credentials from args or environment
    client_id = args.client_id or os.environ.get('EVE_CLIENT_ID')
    client_secret = args.client_secret or os.environ.get('EVE_CLIENT_SECRET')

    if not client_id or not client_secret:
        print("=" * 70)
        print("EVE ONLINE HANGAR CHECKER")
        print("=" * 70)
        print("\nTo use this tool, you need to create an EVE application:")
        print("\n1. Go to: https://developers.eveonline.com/applications")
        print("2. Create a new application with:")
        print("   - Connection type: Authentication & API Access")
        print("   - Permissions: esi-assets.read_assets.v1")
        print("   - Callback URL: http://localhost:8080/callback")
        print("3. Copy your Client ID and Secret Key")
        print("\nThen run with:")
        print("  python check_hangar.py --client-id YOUR_ID --client-secret YOUR_SECRET")
        print("\nOr set environment variables:")
        print("  set EVE_CLIENT_ID=your_client_id")
        print("  set EVE_CLIENT_SECRET=your_secret")
        return 1

    # Try to load existing token
    tokens = None
    if not args.refresh:
        tokens = load_token()
        if tokens:
            print(f"Using cached credentials for: {tokens['character_name']}")

    # Authenticate if needed
    if not tokens:
        tokens = authenticate(client_id, client_secret)

    character_id = tokens['character_id']
    access_token = tokens['access_token']

    print("\n" + "=" * 60)
    print(f"SCANNING ASSETS FOR {tokens['character_name'].upper()}")
    print("=" * 60)

    # Get all assets
    print("\nFetching character assets...")
    assets = get_character_assets(access_token, character_id)
    print(f"Found {len(assets)} items total")

    # Find abyssal heatsinks
    print("\nSearching for abyssal heatsinks...")
    abyssal_heatsinks = [a for a in assets if a['type_id'] == ABYSSAL_HEATSINK_TYPE_ID]

    if not abyssal_heatsinks:
        print("\nNo abyssal heatsinks found in your assets.")
        print("(Looking for type_id 47745 - Abyssal Heat Sink)")

        # Show what abyssal items were found, if any
        # Abyssal modules typically have type IDs in the 47xxx range
        abyssal_items = [a for a in assets if 47700 <= a['type_id'] <= 47900]
        if abyssal_items:
            type_ids = set(a['type_id'] for a in abyssal_items)
            names = get_type_names(type_ids)
            print(f"\nFound {len(abyssal_items)} other abyssal items:")
            for type_id in type_ids:
                count = sum(1 for a in abyssal_items if a['type_id'] == type_id)
                name = names.get(type_id, f"Type {type_id}")
                print(f"  {count}x {name}")
        return 0

    print(f"Found {len(abyssal_heatsinks)} abyssal heatsink(s)")

    # Get type names for source/mutator
    type_ids_needed = set()

    # Evaluate each heatsink
    print("\n" + "=" * 60)
    print("ABYSSAL HEATSINK EVALUATION")
    print("=" * 60)

    for i, heatsink in enumerate(abyssal_heatsinks, 1):
        item_id = heatsink['item_id']

        print(f"\n--- Heatsink #{i} (Item ID: {item_id}) ---")

        # Get dogma attributes
        dogma = get_item_dogma(ABYSSAL_HEATSINK_TYPE_ID, item_id)

        if not dogma:
            print("  Could not fetch item attributes (may not be a valid abyssal item)")
            continue

        # Collect type IDs for names
        if dogma.get('source_type_id'):
            type_ids_needed.add(dogma['source_type_id'])
        if dogma.get('mutator_type_id'):
            type_ids_needed.add(dogma['mutator_type_id'])

        # Evaluate
        eval_result = evaluate_heatsink(dogma)

        if not eval_result:
            print("  Could not evaluate item")
            continue

        # Get names
        names = get_type_names(type_ids_needed)
        source_name = names.get(eval_result['source_type_id'], 'Unknown')
        mutator_name = names.get(eval_result['mutator_type_id'], 'Unknown')

        print(f"\n  Source:    {source_name}")
        print(f"  Mutaplasmid: {mutator_name}")

        print(f"\n  STATS vs BASE ({IN_HEATSINK_BASE['name']}):")
        print(f"    Damage Mult:  {eval_result['damage_mult']:.4f}  ({eval_result['damage_change_pct']:+.2f}%)")
        print(f"    ROF Mult:     {eval_result['rof_mult']:.4f}  ({eval_result['rof_change_pct']:+.2f}%)")
        print(f"    CPU:          {eval_result['cpu']:.1f}  ({eval_result['cpu_change_pct']:+.1f}%)")

        print(f"\n  DPS MULTIPLIER:")
        print(f"    Your roll:   {eval_result['dps_mult']:.4f}")
        print(f"    Base item:   {eval_result['base_dps_mult']:.4f}")
        print(f"    Improvement: {eval_result['dps_change_pct']:+.2f}%")

        print(f"\n  QUALITY: {eval_result['quality']}")

        # Verdict
        if eval_result['dps_change_pct'] > 0:
            print("\n  This is an IMPROVEMENT over the base Imperial Navy Heat Sink")
        else:
            print("\n  This is WORSE than the base Imperial Navy Heat Sink")

    return 0


if __name__ == '__main__':
    sys.exit(main())
