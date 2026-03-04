"""
Data Processing Script for System Dynamics Bibliography
Generates: papers.parquet, author_stats.parquet, coauthor_graph.pkl
"""

import pandas as pd
import numpy as np
import networkx as nx
import pickle
import re
from collections import defaultdict
from itertools import combinations

# =============================================================================
# Configuration
# =============================================================================

PAPERS_CSV = "/mnt/user-data/uploads/260223-Conf_Proceedings_FOR_SAMI__Network_Project_.csv"
ORG_COUNTRY_XLSX = "/mnt/user-data/uploads/People_Country_Org_20260119.xlsx"

OUTPUT_PAPERS = "/home/claude/output/papers_bibliography.parquet"
OUTPUT_AUTHOR_STATS = "/home/claude/output/author_stats_bibliography.parquet"
OUTPUT_GRAPH = "/home/claude/output/coauthor_graph_bibliography.pkl"

# Reference author for "Forrester Number" - can be updated later
REFERENCE_AUTHOR = "Jay Wright Forrester"
KNOWN_COLLABORATORS = ["Dennis L Meadows", "John D Sterman", "Peter M Senge"]

# =============================================================================
# Name Normalization (handles quotes, periods, etc.)
# =============================================================================

_whitespace_re = re.compile(r"\s+")
_punct_re = re.compile(r"[.\u00B7•]")  # periods, middle dots, bullets
_quotes_re = re.compile(r'[\"\'""''`]')  # all types of quotes

def normalize_author_name(name: str) -> str:
    """
    Normalize an author name for matching / graph nodes.
    Handles escaped quotes like: "Mohammad (""MJ"") S. Jalali"
    """
    if name is None:
        return ""
    name = str(name).strip()
    name = _punct_re.sub("", name)  # Remove periods and dots
    name = _quotes_re.sub("", name)  # Remove all types of quotes
    name = _whitespace_re.sub(" ", name)  # Normalize whitespace
    name = name.strip()
    return name.title() if name else ""

def parse_authors(authors_str: str) -> list:
    """
    Parse comma-separated authors string into list of normalized names.
    """
    if pd.isna(authors_str) or not str(authors_str).strip():
        return []
    raw = [a.strip() for a in str(authors_str).split(",")]
    raw = [a for a in raw if a]  # drop empties
    normed = [normalize_author_name(a) for a in raw]
    return [n for n in normed if n]  # drop empty after normalization

# =============================================================================
# Text Cleaning
# =============================================================================

def clean_text(text):
    """
    Remove special characters and extra whitespace from text.
    """
    if pd.isna(text):
        return text
    
    text = str(text)
    
    # Remove specific escaped characters
    text = text.replace('_x000D_', '')
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = text.replace('\t', ' ')
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

# =============================================================================
# Main Processing
# =============================================================================

def main():
    import os
    os.makedirs("/home/claude/output", exist_ok=True)
    
    print("=" * 60)
    print("STEP 1: Load and Clean Papers Data")
    print("=" * 60)
    
    # Load papers
    df = pd.read_csv(PAPERS_CSV)
    print(f"Loaded {len(df):,} papers")
    
    # Select and rename columns
    df = df.rename(columns={
        'Abstract (click arrow to expand and read more)': 'Abstract',
        'Primary Domain': 'Domain'
    })
    
    # Keep relevant columns
    cols_to_keep = ['Title', 'Year', 'Authors', 'Abstract', 'Domain']
    cols_available = [c for c in cols_to_keep if c in df.columns]
    df = df[cols_available].copy()
    
    # Clean text columns
    print("Cleaning text...")
    df['Title'] = df['Title'].apply(clean_text)
    df['Authors'] = df['Authors'].apply(clean_text)
    if 'Abstract' in df.columns:
        df['Abstract'] = df['Abstract'].apply(clean_text)
    
    # Drop rows with missing Authors
    n_before = len(df)
    df = df.dropna(subset=['Authors'])
    df = df[df['Authors'].str.strip() != '']
    print(f"Dropped {n_before - len(df)} rows with missing Authors")
    
    # Convert Year to int where possible
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    
    # Reset index
    df = df.reset_index(drop=True)
    df['paper_id'] = df.index
    
    print(f"Final papers count: {len(df):,}")
    print(f"Year range: {df['Year'].min():.0f} - {df['Year'].max():.0f}")
    
    # ==========================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Build Co-author Graph")
    print("=" * 60)
    
    G = nx.Graph()
    
    # Track papers for each author pair (for showing paper details on edges)
    edge_papers = defaultdict(list)  # (author1, author2) -> [(title, year, paper_id), ...]
    
    for idx, row in df.iterrows():
        authors = parse_authors(row['Authors'])
        if not authors:
            continue
        
        title = row.get('Title', '')
        year = row.get('Year', None)
        paper_id = row.get('paper_id', idx)
        
        # Add nodes
        for a in authors:
            if a not in G:
                G.add_node(a)
        
        # Add edges between all pairs of authors
        unique_authors = sorted(set(authors))
        for a, b in combinations(unique_authors, 2):
            # Store paper info for this collaboration
            edge_key = tuple(sorted([a, b]))
            edge_papers[edge_key].append({
                'title': title,
                'year': year,
                'paper_id': paper_id
            })
            
            # Update edge weight
            if G.has_edge(a, b):
                G[a][b]['weight'] += 1
            else:
                G.add_edge(a, b, weight=1)
    
    # Attach paper details to edges
    for (a, b), papers in edge_papers.items():
        if G.has_edge(a, b):
            G[a][b]['papers'] = papers
    
    print(f"Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    
    # ==========================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Build Author Stats")
    print("=" * 60)
    
    # Explode authors to count papers per author
    ap = df[['paper_id', 'Authors']].copy()
    ap['Author'] = ap['Authors'].apply(parse_authors)
    ap = ap.explode('Author')
    ap = ap[ap['Author'].notna() & (ap['Author'] != '')]
    
    # Count papers per author
    author_papers = (
        ap.groupby('Author')
          .size()
          .rename('NumPapers')
          .reset_index()
    )
    
    # Count coauthors per author (from graph)
    coauthor_counts = []
    for author in G.nodes():
        coauthor_counts.append({
            'Author': author,
            'NumCoauthors': G.degree(author)
        })
    author_coauthors = pd.DataFrame(coauthor_counts)
    
    # Merge
    author_stats = author_papers.merge(author_coauthors, on='Author', how='outer')
    author_stats['NumPapers'] = author_stats['NumPapers'].fillna(0).astype(int)
    author_stats['NumCoauthors'] = author_stats['NumCoauthors'].fillna(0).astype(int)
    
    # Sort by most prolific
    author_stats = author_stats.sort_values(
        ['NumPapers', 'NumCoauthors'], ascending=False
    ).reset_index(drop=True)
    
    print(f"Author stats: {len(author_stats):,} unique authors")
    print(f"Top 10 by papers:")
    print(author_stats.head(10).to_string(index=False))
    
    # ==========================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Join Org/Country Data")
    print("=" * 60)
    
    # Load org/country data
    meta = pd.read_excel(ORG_COUNTRY_XLSX)
    meta = meta.rename(columns={'AUTHORS': 'Author'})
    
    # Normalize author names in meta
    meta['Author'] = meta['Author'].apply(normalize_author_name)
    
    # Deduplicate: keep first non-null Country/Organization per author
    meta_dedup = (
        meta.sort_values(['Country', 'Organization'], na_position='last')
            .drop_duplicates(subset=['Author'], keep='first')
            .copy()
    )
    
    print(f"Org/Country file: {len(meta_dedup):,} unique authors")
    
    # Merge onto author_stats
    author_stats = author_stats.merge(
        meta_dedup[['Author', 'Country', 'Organization']],
        on='Author',
        how='left'
    )
    
    n_with_country = author_stats['Country'].notna().sum()
    n_with_org = author_stats['Organization'].notna().sum()
    print(f"Authors with Country: {n_with_country:,} ({n_with_country/len(author_stats)*100:.1f}%)")
    print(f"Authors with Organization: {n_with_org:,} ({n_with_org/len(author_stats)*100:.1f}%)")
    
    # ==========================================================================
    print("\n" + "=" * 60)
    print("STEP 5: Attach Attributes to Graph Nodes")
    print("=" * 60)
    
    # Create lookup maps
    papers_map = dict(zip(author_stats['Author'], author_stats['NumPapers']))
    coauthors_map = dict(zip(author_stats['Author'], author_stats['NumCoauthors']))
    country_map = dict(zip(author_stats['Author'], author_stats['Country']))
    org_map = dict(zip(author_stats['Author'], author_stats['Organization']))
    
    # Attach to nodes
    for a in G.nodes():
        G.nodes[a]['num_papers'] = int(papers_map.get(a, 0))
        G.nodes[a]['num_coauthors'] = int(coauthors_map.get(a, 0))
        
        country = country_map.get(a)
        org = org_map.get(a)
        
        # Convert NaN to None
        G.nodes[a]['country'] = None if pd.isna(country) else country
        G.nodes[a]['organization'] = None if pd.isna(org) else org
    
    print("Attached num_papers, num_coauthors, country, organization to all nodes")
    
    # ==========================================================================
    print("\n" + "=" * 60)
    print("STEP 6: Check Reference Author (Forrester)")
    print("=" * 60)
    
    # Search for Forrester
    forrester_matches = [n for n in G.nodes() if 'forrester' in n.lower()]
    print(f"Nodes containing 'forrester': {forrester_matches}")
    
    # Check known collaborators
    for collab in KNOWN_COLLABORATORS:
        collab_norm = normalize_author_name(collab)
        matches = [n for n in G.nodes() if collab_norm.lower() in n.lower() or n.lower() in collab_norm.lower()]
        if matches:
            print(f"'{collab}' matches: {matches}")
            for m in matches:
                neighbors = list(G.neighbors(m))[:10]
                print(f"  First 10 neighbors of '{m}': {neighbors}")
    
    # Check if Forrester is connected to known collaborators
    ref_norm = normalize_author_name(REFERENCE_AUTHOR)
    if ref_norm in G:
        print(f"\n'{ref_norm}' IS in the graph!")
        print(f"  Papers: {G.nodes[ref_norm].get('num_papers', 0)}")
        print(f"  Coauthors: {G.nodes[ref_norm].get('num_coauthors', 0)}")
        print(f"  Neighbors: {list(G.neighbors(ref_norm))}")
    else:
        print(f"\n'{ref_norm}' is NOT in the graph")
        print("Will need to add edges manually to known collaborators")
    
    # ==========================================================================
    print("\n" + "=" * 60)
    print("STEP 7: Save Output Files")
    print("=" * 60)
    
    # Save papers
    df.to_parquet(OUTPUT_PAPERS, index=False)
    print(f"Saved: {OUTPUT_PAPERS}")
    
    # Save author stats
    author_stats.to_parquet(OUTPUT_AUTHOR_STATS, index=False)
    print(f"Saved: {OUTPUT_AUTHOR_STATS}")
    
    # Save graph
    with open(OUTPUT_GRAPH, 'wb') as f:
        pickle.dump(G, f)
    print(f"Saved: {OUTPUT_GRAPH}")
    
    # ==========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Papers: {len(df):,}")
    print(f"Authors: {len(author_stats):,}")
    print(f"Graph nodes: {G.number_of_nodes():,}")
    print(f"Graph edges: {G.number_of_edges():,}")
    print(f"Year range: {df['Year'].min():.0f} - {df['Year'].max():.0f}")
    
    return df, author_stats, G

if __name__ == "__main__":
    df, author_stats, G = main()
