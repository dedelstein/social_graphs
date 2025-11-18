"""
Build ground truth ID mapping in multiple stages:
1. ICIJ Reconciliation API (searches all datasets)
2. Local CSV exact match
3. Local CSV partial match (conservative)
4. Local CSV token match (only as last resort with strict requirements)
"""
import httpx
import pickle
from tqdm import tqdm
import json
from collections import defaultdict
import pandas as pd
from pathlib import Path
import time

# Load ground truth names
print("Loading ground truth names...")
with open("names.pkl", "rb") as f:
    GROUND_TRUTH_NAMES = pickle.load(f)

DATA_DIR = Path('data')

# Track results
ground_truth_ids = {}  # gt_name -> (node_id, matched_name, node_type, source)
stage_stats = defaultdict(int)

# Load existing results if available
existing_file = 'ground_truth_ids.pkl'
if Path(existing_file).exists():
    print(f"Found existing {existing_file}, loading previous results...")
    with open(existing_file, 'rb') as f:
        existing_ids = pickle.load(f)

    # Convert simple dict to full format
    for name, node_id in existing_ids.items():
        if name in GROUND_TRUTH_NAMES:
            ground_truth_ids[name] = (node_id, 'Previously found', 'unknown', 'Loaded from file')
            stage_stats['loaded_from_file'] += 1

    print(f"Loaded {len(ground_truth_ids)} previously found IDs")
    print(f"Need to find {len(GROUND_TRUTH_NAMES) - len(ground_truth_ids)} remaining names\n")
else:
    print(f"Need to find {len(GROUND_TRUTH_NAMES)} ground truth names\n")

# =============================================================================
# STAGE 1: ICIJ Reconciliation API
# =============================================================================
print("="*70)
print("STAGE 1: ICIJ Reconciliation API")
print("="*70)

DATASETS = [
    "panama-papers",
    "paradise-papers",
    "pandora-papers",
    "bahamas-leaks",
    "offshore-leaks",
]

BASE_URL = "https://offshoreleaks.icij.org/api/v1/reconcile"

NODE_TYPES = [
    "https://offshoreleaks.icij.org/schema/oldb/officer",
    "https://offshoreleaks.icij.org/schema/oldb/entity",
    "https://offshoreleaks.icij.org/schema/oldb/intermediary",
    "https://offshoreleaks.icij.org/schema/oldb/address",
]

def search_reconcile_api(query, client, rate_limit_hit):
    """Search reconciliation API across all datasets and node types"""
    all_matches = []

    for dataset in DATASETS:
        url = f"{BASE_URL}/{dataset}"

        for node_type in NODE_TYPES:
            try:
                payload = {
                    "queries": {
                        "q1": {
                            "query": query,
                            "type": node_type
                        }
                    }
                }

                response = client.post(url, json=payload, timeout=15.0)
                data = response.json().get("q1", {}).get("result", [])

                # Only keep TRUE matches (match: True flag from API)
                # We trust the API's match determination - score is secondary
                true_matches = [
                    {
                        'id': item.get('id'),
                        'name': item.get('name'),
                        'score': item.get('score'),
                        'dataset': dataset,
                        'node_type': node_type.split('/')[-1]
                    }
                    for item in data
                    if item.get('match') is True  # Only accept when API says it's a true match
                ]

                all_matches.extend(true_matches)

            except Exception as e:
                error_str = str(e)
                # Check if this is a rate limit error (empty JSON response)
                if "Expecting value: line 1 column 1" in error_str:
                    rate_limit_hit[0] = True
                    print("\n RATE LIMIT HIT - Stopping API search")
                    return all_matches
                pass

    return all_matches


try:
    rate_limit_hit = [False]  # Use list so it's mutable in nested function

    with httpx.Client(timeout=20.0) as client:
        for gt_name in tqdm(GROUND_TRUTH_NAMES, desc="API search"):
            # Check if rate limit was hit
            if rate_limit_hit[0]:
                break

            if gt_name in ground_truth_ids:
                continue

            try:
                matches = search_reconcile_api(gt_name, client, rate_limit_hit)

                # Check again in case rate limit was hit during this call
                if rate_limit_hit[0]:
                    break

                if matches:
                    # Sort by score and take best match
                    matches.sort(key=lambda x: x['score'], reverse=True)
                    best = matches[0]

                    ground_truth_ids[gt_name] = (
                        best['id'],
                        best['name'],
                        best['node_type'],
                        f"API:{best['dataset']}"
                    )
                    stage_stats['stage1_api'] += 1
                    print(f"   API: {gt_name} -> {best['name']} (ID: {best['id']}, score: {best['score']})")

                time.sleep(30.0)  # Long delay to avoid rate limits

            except Exception as e:
                print(f"  ✗ API error for {gt_name}: {e}")
                pass

except Exception as e:
    print(f"Warning: API search failed ({e}). Continuing with CSV search...")

print(f"\nStage 1 complete: Found {stage_stats['stage1_api']} via API")

# =============================================================================
# STAGE 2: Local CSV - Exact Match
# =============================================================================
print("\n" + "="*70)
print("STAGE 2: Local CSV - Exact Match")
print("="*70)

csv_files = [
    ('nodes-officers.csv', 'name', 'officer'),
    ('nodes-entities.csv', 'name', 'entity'),
    ('nodes-intermediaries.csv', 'name', 'intermediary'),
    ('nodes-addresses.csv', 'address', 'address'),
    ('nodes-others.csv', 'name', 'other')
]

for csv_file, name_column, node_type in csv_files:
    print(f"\nSearching {csv_file}...")
    df = pd.read_csv(DATA_DIR / csv_file, low_memory=False)
    df['name_upper'] = df[name_column].fillna('').str.upper()

    for gt_name in tqdm(GROUND_TRUTH_NAMES, desc=f"Exact match {node_type}", leave=False):
        if gt_name in ground_truth_ids:
            continue

        gt_name_upper = gt_name.upper()

        # Exact match only
        exact_match = df[df['name_upper'] == gt_name_upper]
        if not exact_match.empty:
            row = exact_match.iloc[0]
            ground_truth_ids[gt_name] = (
                row['node_id'],
                row[name_column],
                node_type,
                f"CSV-exact:{csv_file}"
            )
            stage_stats['stage2_exact'] += 1
            print(f"   EXACT: {gt_name} -> {row[name_column]}")

print(f"\nStage 2 complete: Found {stage_stats['stage2_exact']} exact matches")

# =============================================================================
# STAGE 3: Local CSV - Partial Match (Conservative)
# =============================================================================
print("\n" + "="*70)
print("STAGE 3: Local CSV - Partial Match (Conservative)")
print("="*70)

# Names to completely skip (too many false positives)
SKIP_NAMES = {
    'Queen Elizabeth II',  # Too many streets, hospitals, institutions with this name
}

# Blacklist for known bad matches
PARTIAL_MATCH_BLACKLIST = {
    'Queen Elizabeth II': ['QUEEN ELIZABETH II STREET', 'QUEEN ELIZABETH HOSPITAL'],  # Street name, not the person
}

for csv_file, name_column, node_type in csv_files:
    print(f"\nSearching {csv_file}...")
    df = pd.read_csv(DATA_DIR / csv_file, low_memory=False)
    df['name_upper'] = df[name_column].fillna('').str.upper()

    for gt_name in tqdm(GROUND_TRUTH_NAMES, desc=f"Partial match {node_type}", leave=False):
        if gt_name in ground_truth_ids or gt_name in SKIP_NAMES:
            continue

        gt_name_upper = gt_name.upper()

        # Partial match: ground truth name appears in database name
        # Must be longer than 5 characters to avoid false positives
        if len(gt_name_upper) >= 6:
            partial_match = df[df['name_upper'].str.contains(gt_name_upper, na=False, regex=False)]
            if not partial_match.empty:
                row = partial_match.iloc[0]
                matched_name = row[name_column]

                # Check blacklist
                if gt_name in PARTIAL_MATCH_BLACKLIST:
                    blacklist_terms = PARTIAL_MATCH_BLACKLIST[gt_name]
                    matched_upper = str(matched_name).upper()
                    if any(term in matched_upper for term in blacklist_terms):
                        print(f"  ✗ BLACKLISTED: {gt_name} -> {matched_name}")
                        continue

                ground_truth_ids[gt_name] = (
                    row['node_id'],
                    matched_name,
                    node_type,
                    f"CSV-partial:{csv_file}"
                )
                stage_stats['stage3_partial'] += 1
                print(f"   PARTIAL: {gt_name} -> {matched_name}")

print(f"\nStage 3 complete: Found {stage_stats['stage3_partial']} partial matches")

# =============================================================================
# STAGE 4: Local CSV - Token Match (Last Resort, Very Strict)
# =============================================================================
print("\n" + "="*70)
print("STAGE 4: Local CSV - Token Match (Last Resort, Very Strict)")
print("="*70)

# Common words to exclude from token matching
COMMON_WORDS = {'AND', 'THE', 'OF', 'IN', 'FOR', 'TO', 'A', 'AN', 'S.A.', 'LTD',
                'LIMITED', 'INC', 'CORP', 'LLC', 'CO', 'GROUP', 'FAMILY'}

for csv_file, name_column, node_type in csv_files:
    print(f"\nSearching {csv_file}...")
    df = pd.read_csv(DATA_DIR / csv_file, low_memory=False)
    df['name_upper'] = df[name_column].fillna('').str.upper()

    for gt_name in tqdm(GROUND_TRUTH_NAMES, desc=f"Token match {node_type}", leave=False):
        if gt_name in ground_truth_ids:
            continue

        gt_name_upper = gt_name.upper()

        # Extract tokens (words longer than 3 chars, excluding common words)
        gt_tokens = set(gt_name_upper.split())
        gt_tokens = {t for t in gt_tokens if len(t) > 3 and t not in COMMON_WORDS}

        # Need at least 2 significant tokens
        if len(gt_tokens) < 2:
            continue

        # Try to find rows where ALL significant tokens appear
        mask = pd.Series([True] * len(df))
        for token in gt_tokens:
            mask &= df['name_upper'].str.contains(token, na=False, regex=False)

        token_matches = df[mask]
        if not token_matches.empty:
            row = token_matches.iloc[0]
            ground_truth_ids[gt_name] = (
                row['node_id'],
                row[name_column],
                node_type,
                f"CSV-token:{csv_file}"
            )
            stage_stats['stage4_token'] += 1
            print(f"   TOKEN-ALL: {gt_name} -> {row[name_column]} (matched all: {gt_tokens})")

print(f"\nStage 4 complete: Found {stage_stats['stage4_token']} token matches")

# =============================================================================
# STAGE 5: Local CSV - Last Name Match (Very Flexible, Manual Review)
# =============================================================================
print("\n" + "="*70)
print("STAGE 5: Local CSV - Last Name Match (Flexible)")
print("="*70)

# Ask user if they want to proceed with Stage 5
remaining = len([n for n in GROUND_TRUTH_NAMES if n not in ground_truth_ids and n not in SKIP_NAMES])
print(f"\n{remaining} names still missing.")
print("Stage 5 will search by last name only - this may produce false positives.")
response = input("\nProceed with Stage 5? (y/n): ").strip().lower()

if response != 'y':
    print("Skipping Stage 5...")
    stage_stats['stage5_lastname'] = 0
else:
    # Extract last names from remaining names
    def extract_last_name(full_name):
        """Extract likely last name from full name"""
        # Skip family/group names
        if any(word in full_name.lower() for word in ['family', 'children', 'and ', ' & ']):
            return None

        parts = full_name.strip().split()
        if len(parts) == 0:
            return None

        # Common patterns
        # Handle "bin/ibn" pattern (Arabic names) - use the part after last bin/ibn
        if 'bin' in parts or 'ibn' in parts:
            # Find last occurrence of bin/ibn and take next word
            for i in range(len(parts) - 1, -1, -1):
                if parts[i].lower() in ['bin', 'ibn']:
                    if i + 1 < len(parts):
                        return parts[i + 1]
            return parts[-1]  # Fallback

        # Handle "de/da/van/von" pattern - include the particle
        if len(parts) >= 2 and parts[-2].lower() in ['de', 'da', 'van', 'von', 'del', 'di']:
            return ' '.join(parts[-2:])

        # Default: last word
        return parts[-1]

    for csv_file, name_column, node_type in csv_files:
        print(f"\nSearching {csv_file}...")
        df = pd.read_csv(DATA_DIR / csv_file, low_memory=False)
        df['name_upper'] = df[name_column].fillna('').str.upper()

        for gt_name in tqdm(GROUND_TRUTH_NAMES, desc=f"Last name {node_type}", leave=False):
            if gt_name in ground_truth_ids or gt_name in SKIP_NAMES:
                continue

            last_name = extract_last_name(gt_name)
            if not last_name or len(last_name) < 4:
                continue

            last_name_upper = last_name.upper()

            # Search for last name in database
            last_name_match = df[df['name_upper'].str.contains(last_name_upper, na=False, regex=False)]
            if not last_name_match.empty:
                # Show match and ask for confirmation
                row = last_name_match.iloc[0]
                print("     LASTNAME MATCH FOUND:")
                print(f"     Ground truth: {gt_name}")
                print(f"     Last name: {last_name}")
                print(f"     Database match: {row[name_column]}")
                print(f"     Node ID: {row['node_id']}")
                print(f"     Type: {node_type}")

                response = input("     Accept this match? (y/n/s to skip all remaining): ").strip().lower()

                if response == 's':
                    print("     Skipping all remaining Stage 5 matches...")
                    break  # Exit inner loop
                elif response == 'y':
                    ground_truth_ids[gt_name] = (
                        row['node_id'],
                        row[name_column],
                        node_type,
                        f"CSV-lastname:{csv_file}"
                    )
                    stage_stats['stage5_lastname'] += 1
                    print("      ACCEPTED")
                    break  # Found, move to next name
                else:
                    print("      REJECTED")
                    # Continue searching other CSV files

print(f"\nStage 5 complete: Found {stage_stats['stage5_lastname']} last name matches")
print("  Stage 5 matches should be manually reviewed for accuracy!")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
print(f"Total ground truth names: {len(GROUND_TRUTH_NAMES)}")
print(f"Successfully found: {len(ground_truth_ids)}")
print(f"Still missing: {len(GROUND_TRUTH_NAMES) - len(ground_truth_ids)}")

print("\nBreakdown by stage:")
if stage_stats.get('loaded_from_file'):
    print(f"  Loaded from file: {stage_stats['loaded_from_file']}")
print(f"  Stage 1 (API): {stage_stats['stage1_api']}")
print(f"  Stage 2 (Exact): {stage_stats['stage2_exact']}")
print(f"  Stage 3 (Partial): {stage_stats['stage3_partial']}")
print(f"  Stage 4 (Token): {stage_stats['stage4_token']}")
print(f"  Stage 5 (Last Name): {stage_stats['stage5_lastname']}")

# Find missing names
missing_names = [name for name in GROUND_TRUTH_NAMES if name not in ground_truth_ids]

if missing_names:
    print(f"\n{'='*70}")
    print(f"MISSING NAMES ({len(missing_names)}):")
    print("="*70)
    for name in sorted(missing_names):
        print(f"  - {name}")

# =============================================================================
# SAVE RESULTS
# =============================================================================
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

# Save as pickle (node_id only)
node_id_dict = {name: data[0] for name, data in ground_truth_ids.items()}
with open('ground_truth_ids.pkl', 'wb') as f:
    pickle.dump(node_id_dict, f)
print(f" Saved node IDs to ground_truth_ids.pkl ({len(node_id_dict)} IDs)")

# Save full details as JSON
output_json = {
    'ground_truth_ids': {
        name: {
            'node_id': str(data[0]),  # Convert to string for JSON
            'matched_name': str(data[1]),
            'node_type': str(data[2]),
            'source': str(data[3])
        }
        for name, data in ground_truth_ids.items()
    },
    'missing': missing_names,
    'summary': {
        'total': len(GROUND_TRUTH_NAMES),
        'found': len(ground_truth_ids),
        'missing': len(missing_names),
        'stage_breakdown': dict(stage_stats)
    }
}

with open('ground_truth_ids_full.json', 'w') as f:
    json.dump(output_json, f, indent=2)
print("Saved full details to ground_truth_ids_full.json")

# Save human-readable text report
with open('ground_truth_ids_report.txt', 'w') as f:
    f.write("Ground Truth ID Mapping - Full Report\n")
    f.write("="*70 + "\n\n")

    f.write(f"Total: {len(GROUND_TRUTH_NAMES)}\n")
    f.write(f"Found: {len(ground_truth_ids)}\n")
    f.write(f"Missing: {len(missing_names)}\n\n")

    f.write("FOUND NAMES:\n")
    f.write("-"*70 + "\n")
    for gt_name in sorted(ground_truth_ids.keys()):
        node_id, matched_name, node_type, source = ground_truth_ids[gt_name]
        f.write(f"\n{gt_name}\n")
        f.write(f"  -> {matched_name}\n")
        f.write(f"  -> ID: {node_id}\n")
        f.write(f"  -> Type: {node_type}\n")
        f.write(f"  -> Source: {source}\n")

    if missing_names:
        f.write("\n" + "="*70 + "\n")
        f.write("MISSING NAMES:\n")
        f.write("-"*70 + "\n")
        for name in sorted(missing_names):
            f.write(f"  - {name}\n")

print("Saved report to ground_truth_ids_report.txt")

print(f"\n{'='*70}")
print("DONE!")
print("="*70)
print(f"Successfully mapped {len(ground_truth_ids)} out of {len(GROUND_TRUTH_NAMES)} ground truth names")
if len(ground_truth_ids) == len(GROUND_TRUTH_NAMES):
    print("ALL GROUND TRUTH NAMES FOUND!")
else:
    print(f"Still need to find {len(missing_names)} names manually")
