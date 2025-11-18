"""
Panama Papers Network Builder - HPC Optimized Two-Phase Approach

Phase 1: Aggressive initial construction (depth 6, ~250K nodes)
Phase 2: Intelligent filtering to target size (~50K nodes) with centrality-based sampling
"""

import networkx as nx
import pandas as pd
import numpy as np
from collections import deque, Counter
import pickle
import argparse
import logging
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import sys
from multiprocessing import Pool, cpu_count

# Ground truth names from power players list
with open("names.pkl", "rb") as f:
    GROUND_TRUTH_NAMES = pickle.load(f)

DATA_DIR = Path('data')
NUM_CORES = cpu_count()

def setup_logging(log_file):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def load_single_node_csv(node_type, file_name):
    """Load a single node CSV file and return processed data"""
    df = pd.read_csv(DATA_DIR / file_name, low_memory=False, dtype={'node_id': str})

    node_dict = {}

    if node_type == 'entity':
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {node_type}", leave=False):
            node_dict[row['node_id']] = {
                'name': row['name'],
                'node_type': 'entity',
                'countries': row.get('countries', ''),
                'jurisdiction': row.get('jurisdiction', ''),
                'sourceID': row.get('sourceID', '')
            }
    elif node_type == 'officer':
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {node_type}", leave=False):
            node_dict[row['node_id']] = {
                'name': row['name'],
                'node_type': 'officer',
                'countries': row.get('countries', ''),
                'sourceID': row.get('sourceID', '')
            }
    elif node_type == 'intermediary':
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {node_type}", leave=False):
            node_dict[row['node_id']] = {
                'name': row['name'],
                'node_type': 'intermediary',
                'countries': row.get('countries', ''),
                'sourceID': row.get('sourceID', '')
            }
    elif node_type == 'address':
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {node_type}", leave=False):
            node_dict[row['node_id']] = {
                'name': row.get('address', 'Unknown Address'),
                'node_type': 'address',
                'countries': row.get('countries', ''),
                'sourceID': row.get('sourceID', '')
            }
    elif node_type == 'other':
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {node_type}", leave=False):
            node_dict[row['node_id']] = {
                'name': row.get('name', 'Unknown'),
                'node_type': 'other',
                'countries': row.get('countries', ''),
                'sourceID': row.get('sourceID', '')
            }

    return node_type, node_dict, df if node_type == 'officer' else None

def load_node_data(logger):
    """Load all node CSVs in parallel and create lookup dictionaries"""
    logger.info(f"Loading node data using {NUM_CORES} CPU cores...")

    node_files = [
        ('entity', 'nodes-entities.csv'),
        ('officer', 'nodes-officers.csv'),
        ('intermediary', 'nodes-intermediaries.csv'),
        ('address', 'nodes-addresses.csv'),
        ('other', 'nodes-others.csv')
    ]

    logger.info("Loading and processing CSV files in parallel...")
    with Pool(processes=min(5, NUM_CORES)) as pool:
        results = pool.starmap(load_single_node_csv, node_files)

    node_data = {}
    officers_df = None

    for node_type, node_dict, df in results:
        logger.info(f"Processing {len(node_dict)} {node_type} nodes...")
        node_data.update(node_dict)
        if node_type == 'officer':
            officers_df = df

    logger.info(f"Loaded {len(node_data)} total nodes")

    return node_data, officers_df

def process_relationship_chunk(chunk_data):
    """Process a chunk of relationships and return adjacency lists"""
    adjacency_out = {}
    adjacency_in = {}

    for _, row in tqdm(chunk_data.iterrows(), total=len(chunk_data), desc="Processing chunk", leave=False):
        source = row['node_id_start']
        target = row['node_id_end']
        rel_type = row['rel_type']

        if source not in adjacency_out:
            adjacency_out[source] = []
        adjacency_out[source].append((target, rel_type))

        if target not in adjacency_in:
            adjacency_in[target] = []
        adjacency_in[target].append((source, rel_type))

    return adjacency_out, adjacency_in

def build_network_bfs(relationships_file, ground_truth_ids, node_data,
                      max_nodes, max_depth, logger):
    """Build network using BFS expansion from ground truth nodes

    Note: NetworkX DiGraph automatically prevents duplicate nodes and edges.
    max_nodes=0 means unlimited (explore to max_depth without node limit).
    """
    logger.info("="*70)
    logger.info("PHASE 1: INITIAL NETWORK CONSTRUCTION")
    logger.info("="*70)
    logger.info(f"Max nodes: {'UNLIMITED' if max_nodes == 0 else max_nodes}")
    logger.info(f"Max depth: {max_depth}")
    logger.info(f"Starting from {len(ground_truth_ids)} ground truth nodes")

    G = nx.DiGraph()

    # Set effective max_nodes (0 = unlimited)
    effective_max_nodes = float('inf') if max_nodes == 0 else max_nodes

    # Load relationships and build adjacency lists using parallel processing
    logger.info(f"Loading relationships from CSV using {NUM_CORES} CPU cores...")
    chunk_size = 500000
    adjacency_out = {}  # node -> [(target, rel_type), ...]
    adjacency_in = {}   # node -> [(source, rel_type), ...]

    logger.info("Reading relationship chunks...")
    chunks = list(tqdm(pd.read_csv(relationships_file, chunksize=chunk_size, low_memory=False,
                                     dtype={'node_id_start': str, 'node_id_end': str}),
                       desc="Loading chunks"))

    # Process chunks in parallel
    logger.info(f"Processing {len(chunks)} chunks in parallel...")
    with Pool(processes=NUM_CORES) as pool:
        results = list(tqdm(pool.imap(process_relationship_chunk, chunks),
                           total=len(chunks),
                           desc="Processing relationships"))

    logger.info("Merging adjacency lists...")
    total_relationships = 0
    for adj_out, adj_in in tqdm(results, desc="Merging"):
        for source, targets in adj_out.items():
            if source not in adjacency_out:
                adjacency_out[source] = []
            adjacency_out[source].extend(targets)
            total_relationships += len(targets)

        for target, sources in adj_in.items():
            if target not in adjacency_in:
                adjacency_in[target] = []
            adjacency_in[target].extend(sources)

    logger.info(f"Loaded {total_relationships} total relationships")
    logger.info(f"Nodes with outgoing edges: {len(adjacency_out)}")
    logger.info(f"Nodes with incoming edges: {len(adjacency_in)}")

    logger.info("\nExpanding network with BFS...")
    visited = set()
    queue = deque([(node_id, 0) for node_id in ground_truth_ids])

    for node_id in ground_truth_ids:
        visited.add(node_id)

    depth_stats = Counter()

    edges_to_add = [] # Collect edges separately - add them only after both endpoints are in graph

    pbar = tqdm(total=None if max_nodes == 0 else max_nodes, desc="Building network")

    with pbar:
        while queue and len(G.nodes()) < effective_max_nodes:
            current_node, depth = queue.popleft()

            if current_node in G:
                continue

            if current_node not in node_data:
                continue

            G.add_node(current_node, **node_data[current_node], depth=depth)
            depth_stats[depth] += 1
            pbar.update(1)

            if depth >= max_depth:
                continue

            if current_node in adjacency_out: # Collect outgoing edges and enqueue neighbors
                for target, rel_type in adjacency_out[current_node]:
                    if target in node_data:
                        edges_to_add.append((current_node, target, rel_type))

                        if target not in visited and len(G.nodes()) < effective_max_nodes:
                            visited.add(target)
                            queue.append((target, depth + 1))

            if current_node in adjacency_in:
                for source, rel_type in adjacency_in[current_node]:
                    if source in node_data:
                        edges_to_add.append((source, current_node, rel_type))

                        if source not in visited and len(G.nodes()) < effective_max_nodes:
                            visited.add(source)
                            queue.append((source, depth + 1))

    logger.info(f"\nAdding edges (total collected: {len(edges_to_add)})...")
    edges_added = 0
    for source, target, rel_type in tqdm(edges_to_add, desc="Adding edges"):
        if source in G and target in G:
            G.add_edge(source, target, rel_type=rel_type)
            edges_added += 1

    logger.info(f"Added {edges_added} edges ({len(edges_to_add) - edges_added} skipped due to missing endpoints)")

    logger.info(f"\nBuilt network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    logger.info(f"Depth distribution: {dict(depth_stats)}")

    unique_nodes = len(set(G.nodes()))
    unique_edges = len(set(G.edges()))
    logger.info(f"  Unique nodes: {unique_nodes} (total: {G.number_of_nodes()}) - OK")
    logger.info(f"  Unique edges: {unique_edges} (total: {G.number_of_edges()}) - OK")

    return G

def get_largest_connected_component(G, logger):
    """Extract largest connected component from directed graph"""
    logger.info("\nExtracting largest connected component...")

    # For directed graphs, use weakly connected components
    components = list(nx.weakly_connected_components(G))
    components.sort(key=len, reverse=True)

    logger.info(f"Found {len(components)} weakly connected components")
    logger.info(f"Top 5 component sizes: {[len(c) for c in components[:5]]}")

    # Get LCC
    lcc_nodes = components[0]
    G_lcc = G.subgraph(lcc_nodes).copy(as_view=True)

    logger.info(f"LCC: {G_lcc.number_of_nodes()} nodes, {G_lcc.number_of_edges()} edges")

    return G_lcc

def print_network_stats(G, name, logger):
    """Print comprehensive network statistics"""
    logger.info("\n" + "="*70)
    logger.info(f"{name} Statistics")
    logger.info("="*70)
    logger.info(f"Nodes: {G.number_of_nodes():,}")
    logger.info(f"Edges: {G.number_of_edges():,}")
    logger.info(f"Density: {nx.density(G):.8f}")

    # Node type distribution
    node_types = Counter()
    for _, attrs in G.nodes(data=True):
        node_types[attrs.get('node_type', 'unknown')] += 1

    logger.info("\nNode type distribution:")
    for nt, count in sorted(node_types.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {nt}: {count:,} ({100*count/G.number_of_nodes():.1f}%)")

    # Edge type distribution
    edge_types = Counter()
    for _, _, attrs in G.edges(data=True):
        edge_types[attrs.get('rel_type', 'unknown')] += 1

    logger.info("\nTop 10 relationship types:")
    for et, count in sorted(edge_types.items(), key=lambda x: x[1], reverse=True)[:10]:
        logger.info(f"  {et}: {count:,} ({100*count/G.number_of_edges():.1f}%)")

    # Degree statistics
    if G.number_of_nodes() > 0:
        if G.is_directed():
            in_degrees = [d for n, d in G.in_degree()]
            out_degrees = [d for n, d in G.out_degree()]
            logger.info(f"\nIn-degree - Min: {min(in_degrees)}, Max: {max(in_degrees)}, "
                       f"Mean: {np.mean(in_degrees):.2f}, Median: {np.median(in_degrees):.2f}")
            logger.info(f"Out-degree - Min: {min(out_degrees)}, Max: {max(out_degrees)}, "
                       f"Mean: {np.mean(out_degrees):.2f}, Median: {np.median(out_degrees):.2f}")
        else:
            degrees = [d for n, d in G.degree()]
            logger.info(f"\nDegree - Min: {min(degrees)}, Max: {max(degrees)}, "
                       f"Mean: {np.mean(degrees):.2f}, Median: {np.median(degrees):.2f}")

def phase1_initial_construction(args, logger):
    """Phase 1: Build large initial network"""
    node_data, _ = load_node_data(logger)

    logger.info("Loading ground truth IDs from ground_truth_ids.pkl...")
    with open('ground_truth_ids.pkl', 'rb') as f:
        ground_truth_id_dict = pickle.load(f)

    logger.info(f"Loaded {len(ground_truth_id_dict)} ground truth IDs")

    ground_truth_ids = []
    found_mapping = []
    missing_ids = []

    for gt_name, node_id in ground_truth_id_dict.items():
        node_id_str = str(node_id)

        if node_id_str in node_data:
            ground_truth_ids.append(node_id_str)
            canonical_name = node_data[node_id_str]['name']
            node_type = node_data[node_id_str]['node_type']
            found_mapping.append((gt_name, canonical_name, node_type, node_id_str))
            logger.info(f"  ✓ {gt_name} -> {canonical_name} ({node_type}, ID: {node_id_str})")
        else:
            missing_ids.append((gt_name, node_id_str))
            logger.warning(f"{gt_name}: ID {node_id_str} not found in node_data")

    logger.info(f"\nSuccessfully loaded {len(ground_truth_ids)} ground truth nodes")
    if missing_ids:
        logger.warning(f"WARNING: {len(missing_ids)} IDs not found in CSV data")

    # Save ground truth mapping
    mapping_file = 'ground_truth_mapping.txt'
    with open(mapping_file, 'w') as f:
        f.write("Ground Truth Name -> Canonical Name (Type, ID)\n")
        f.write("="*70 + "\n")
        for gt_name, canonical_name, node_type, node_id in found_mapping:
            f.write(f"{gt_name}\n")
            f.write(f"  -> {canonical_name}\n")
            f.write(f"  -> Type: {node_type}\n")
            f.write(f"  -> ID: {node_id}\n\n")
        if missing_ids:
            f.write("\n" + "="*70 + "\n")
            f.write(f"MISSING IDs ({len(missing_ids)}):\n")
            f.write("-"*70 + "\n")
            for gt_name, node_id in missing_ids:
                f.write(f"  {gt_name}: {node_id}\n")
    logger.info(f"Ground truth mapping saved to {mapping_file}")

    # Build network
    relationships_file = DATA_DIR / 'relationships.csv'
    G = build_network_bfs(
        relationships_file,
        ground_truth_ids,
        node_data,
        max_nodes=args.max_nodes,
        max_depth=args.max_depth,
        logger=logger
    )

    # Print stats
    print_network_stats(G, "Initial Network", logger)

    # Extract LCC
    G_lcc = get_largest_connected_component(G, logger)
    print_network_stats(G_lcc, "Largest Connected Component", logger)

    # Save networks
    logger.info("\nSaving networks...")
    if args.output:
        output_file = args.output
    else:
        node_label = 'unlimited' if args.max_nodes == 0 else str(args.max_nodes)
        output_file = f'panama_network_initial_{node_label}.pkl'

    lcc_file = output_file.replace('.pkl', '_lcc.pkl')

    logger.info(f"Saving initial network to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"  Saved: {output_file}")

    logger.info(f"Saving LCC to {lcc_file}...")
    with open(lcc_file, 'wb') as f:
        pickle.dump(G_lcc, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"  Saved: {lcc_file}")

    return G, G_lcc

def main():
    parser = argparse.ArgumentParser(
        description='Panama Papers Network Builder - Two-Phase HPC Approach',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    Build initial network:
    python build_network.py--max-nodes 250000 --max-depth 6
        """
    )

    parser.add_argument('--max-nodes', type=int, default=250000,
                       help='Maximum nodes for initial network (Phase 1)')
    parser.add_argument('--max-depth', type=int, default=6,
                       help='Maximum BFS depth (Phase 1)')
    parser.add_argument('--output', type=str,
                       help='Output file name (default: auto-generated)')
    parser.add_argument('--log-file', type=str,
                       default=f'network_builder_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
                       help='Log file name')

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_file)
    logger.info("Panama Papers Network Builder")
    logger.info(f"Arguments: {vars(args)}")

    phase1_initial_construction(args, logger)

    logger.info("\n" + "="*70)
    logger.info("SUCCESS: Process completed")
    logger.info("="*70)

if __name__ == "__main__":
    main()