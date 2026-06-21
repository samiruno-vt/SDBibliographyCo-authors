"""
System Dynamics Bibliography Explorer
Features: Top Authors, Co-author Network, Forrester Number
"""

import os
import sys
import pickle
import re
import pandas as pd
import numpy as np
import streamlit as st
import networkx as nx
import plotly.graph_objects as go
from rapidfuzz import process, fuzz
from collections import defaultdict
from itertools import combinations


# Canonical display order for the Publication type filter (Authors tab).
PUB_TYPE_ORDER = [
    "Conference Paper",
    "Journal Article",
    "D-memo",
    "Book",
    "Book Section",
    "Thesis",
    "Textbook",
]


def edge_visual(weight):
    """Return (width, color) for a co-authorship edge so the number of shared
    papers is easy to tell apart: thin and light for a single shared paper,
    progressively thicker and darker as the count grows."""
    w = max(int(weight or 1), 1)
    width = min(1.2 + 1.4 * (w - 1), 11)
    alpha = min(0.30 + 0.13 * (w - 1), 0.9)
    shade = int(max(150 - 18 * (w - 1), 60))
    return width, f"rgba({shade},{shade},{shade},{alpha:.2f})"


# =============================================================================
# Name Normalization Functions
# =============================================================================

_whitespace_re = re.compile(r"\s+")
_punct_re = re.compile(r"[.\u00B7•]")
_quotes_re = re.compile(r'[\"\'""''`]')

def normalize_author_name(name: str) -> str:
    if name is None:
        return ""
    name = str(name).strip()
    name = _punct_re.sub("", name)
    name = _quotes_re.sub("", name)
    name = _whitespace_re.sub(" ", name)
    return name.title() if name else ""

def parse_authors(authors_str: str) -> list:
    if pd.isna(authors_str) or not str(authors_str).strip():
        return []
    raw = [a.strip() for a in str(authors_str).split(",")]
    raw = [a for a in raw if a]
    return [normalize_author_name(a) for a in raw if normalize_author_name(a)]


# =============================================================================
# Search Functions
# =============================================================================

def search_authors(query, all_authors, limit=10, score_cutoff=60):
    """Search for authors matching the query."""
    q = normalize_author_name(query)
    if not q:
        return []
    
    q_lower = q.lower()
    
    # First, find exact substring matches
    exact_matches = []
    for name in all_authors:
        if q_lower in name.lower():
            score = len(q) / len(name) * 100
            exact_matches.append((name, min(100, score + 50)))
    
    exact_matches.sort(key=lambda x: (-x[1], x[0]))
    
    if exact_matches:
        # keep the top `limit` by relevance, then show them alphabetically
        return sorted(exact_matches[:limit], key=lambda x: x[0].lower())
    
    # Fall back to fuzzy matching
    fuzzy_results = process.extract(q, all_authors, scorer=fuzz.WRatio, limit=limit)
    fuzzy = [(name, score) for name, score, _ in fuzzy_results if score >= score_cutoff]
    return sorted(fuzzy, key=lambda x: x[0].lower())


# =============================================================================
# Co-author Functions
# =============================================================================

@st.cache_data
def get_coauthors_by_degree(_G, author, max_degree=2, mode="full"):
    """Get co-authors up to max_degree hops from author."""
    if author not in _G:
        return []
    
    # Excluded names
    excluded = {'Unknown', 'Anonymous', 'unknown', 'anonymous', ''}
    
    results = []
    visited = {author}
    visited_normalized = {normalize_author_name(author)}
    current_level = {author}
    
    for degree in range(1, max_degree + 1):
        next_level = set()
        degree_data = {}  # Use dict to deduplicate: normalized_name -> {data}
        
        for node in current_level:
            for nbr in _G.neighbors(node):
                nbr_norm = normalize_author_name(nbr)
                
                # Skip if already visited in previous degrees, excluded, or too short
                if nbr_norm in visited_normalized:
                    continue
                if nbr in excluded or len(nbr) <= 2:
                    continue
                if nbr_norm in excluded or len(nbr_norm) <= 2:
                    continue
                
                next_level.add(nbr)
                
                # Only add to degree_data if not already seen (dedup by normalized name)
                if nbr_norm not in degree_data:
                    if degree == 1:
                        weight = _G[node][nbr].get("weight", 1)
                        degree_data[nbr_norm] = {
                            "Co-author": nbr,
                            "Shared Papers": weight
                        }
                    else:
                        degree_data[nbr_norm] = {"Author": nbr}
        
        visited.update(next_level)
        visited_normalized.update(normalize_author_name(n) for n in next_level)
        current_level = next_level
        
        if degree_data:
            df = pd.DataFrame(list(degree_data.values()))
            if degree == 1:
                df = df.sort_values("Shared Papers", ascending=False)
            else:
                df = df.sort_values("Author")
            results.append(df.reset_index(drop=True))
        else:
            results.append(pd.DataFrame())
    
    return results


def build_coauthor_network(G, author, max_degree=2):
    """Build subgraph for visualization."""
    if author not in G:
        return nx.Graph()
    
    H = nx.Graph()
    visited = {author: 0}
    queue = [(author, 0)]
    
    while queue:
        node, level = queue.pop(0)
        if level >= max_degree:
            continue
        
        for nbr in G.neighbors(node):
            if nbr not in visited:
                visited[nbr] = level + 1
                queue.append((nbr, level + 1))
    
    # Add nodes with levels
    for node, level in visited.items():
        H.add_node(node, level=level)
        if node in G.nodes:
            H.nodes[node].update(G.nodes[node])
    
    # Add edges
    for node in visited:
        for nbr in G.neighbors(node):
            if nbr in visited:
                weight = G[node][nbr].get("weight", 1)
                H.add_edge(node, nbr, weight=weight)
                # Copy paper details if available
                if "papers" in G[node][nbr]:
                    H[node][nbr]["papers"] = G[node][nbr]["papers"]
    
    return H


def plot_coauthor_network(H, center_author):
    """Create Plotly figure for co-author network.
    
    Edge midpoints are rendered as invisible clickable markers.
    Clicking one stores the edge key in the point's customdata so the
    caller can display paper details in a panel below the chart.
    """
    if H.number_of_nodes() == 0:
        return None, {}

    n = H.number_of_nodes()

    if n > 500:
        iterations = 50
    elif n > 200:
        iterations = 100
    elif n > 50:
        iterations = 150
    else:
        iterations = 200

    k = 6 / np.sqrt(n) if n > 1 else 1
    pos = nx.spring_layout(H, seed=42, k=k, iterations=iterations, scale=3)

    # Build a lookup: edge_key -> paper list, for use in the panel
    edge_paper_lookup = {}

    edge_traces = []
    mid_x, mid_y, mid_labels, mid_keys = [], [], [], []

    for u, v in H.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2

        papers = H[u][v].get("papers", [])
        weight = H[u][v].get("weight", len(papers) if papers else 1)
        edge_key = f"{u}|||{v}"
        edge_paper_lookup[edge_key] = {"u": u, "v": v, "papers": papers, "weight": weight}

        # Width and shade scaled by number of shared papers
        edge_width, edge_color = edge_visual(weight)

        # Visible line
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=edge_width, color=edge_color),
            hoverinfo="skip",
            showlegend=False
        ))

        mid_x.append(xm)
        mid_y.append(ym)
        mid_labels.append(f"{u} & {v}: {weight} shared paper(s)\n(click to see details below)")
        mid_keys.append(edge_key)

    # Single trace of invisible midpoint markers — clicking triggers selection
    edge_traces.append(go.Scatter(
        x=mid_x, y=mid_y,
        mode="markers",
        marker=dict(size=14, color="rgba(0,0,0,0)", line=dict(width=0)),
        hoverinfo="text",
        hovertext=mid_labels,
        hoverlabel=dict(bgcolor="white", bordercolor="#aaa", font=dict(size=14)),
        customdata=mid_keys,
        showlegend=False,
        name="edges"
    ))

    # Node colors by degree level
    level_colors = {0: "#d62828", 1: "#2a9d8f", 2: "#457b9d", 3: "#8338ec", 4: "#6c757d"}

    node_x, node_y, node_text, node_colors, node_sizes = [], [], [], [], []
    node_names = []

    for node in H.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_names.append(node)

        level = H.nodes[node].get("level", 4)
        num_papers = int(H.nodes[node].get("num_papers") or 0)
        num_coauthors = int(H.nodes[node].get("num_coauthors") or 0)
        country = H.nodes[node].get("country") or ""
        org = H.nodes[node].get("organization") or ""

        hover = f"<b>{node}</b><br>Papers: {num_papers}<br>Co-authors: {num_coauthors}"
        if country:
            hover += f"<br>Country: {country}"
        if org:
            hover += f"<br>Org: {org}"
        node_text.append(hover)

        node_colors.append(level_colors.get(level, "#6c757d"))

        if level == 0:
            node_sizes.append(60)
        elif level == 1:
            node_sizes.append(35)
        else:
            node_sizes.append(22)

    show_labels = n <= 30

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text" if show_labels else "markers",
        text=node_names if show_labels else None,
        textposition="top center",
        textfont=dict(size=17, color="#222222"),
        hoverinfo="text",
        hovertext=node_text,
        hoverlabel=dict(bgcolor="white", bordercolor="#aaa", font=dict(size=15)),
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color="white"),
            opacity=0.9
        ),
        showlegend=False
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        showlegend=False,
        plot_bgcolor="#f8f9fa",
        margin=dict(l=5, r=5, t=5, b=5),
        height=600,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        dragmode="pan",
        hovermode="closest"
    )

    return fig, edge_paper_lookup




# =============================================================================
# Organization Functions
# =============================================================================

def build_org_mapping_from_graph(G):
    """Return {normalized_author: organization} for authors present in G."""
    m = {}
    for n, d in G.nodes(data=True):
        org = d.get("organization")
        if org and str(org).strip():
            m[n] = str(org).strip()
    return m


@st.cache_data
def build_org_network(_G, mode="full"):
    """Organization-level collaboration graph derived from co-author edges.

    Nodes are organizations with >=1 author in _G. An edge between two
    organizations is weighted by the number of shared papers across all
    cross-organization co-author pairs. Intra-org collaboration is kept as
    a node attribute.
    """
    author_org = build_org_mapping_from_graph(_G)

    org_authors = defaultdict(set)
    for a, org in author_org.items():
        org_authors[org].add(a)

    OG = nx.Graph()
    for org, authors in org_authors.items():
        OG.add_node(org, num_authors=len(authors), intra_weight=0)

    inter = defaultdict(int)
    inter_papers = defaultdict(list)
    for u, v, d in _G.edges(data=True):
        ou, ov = author_org.get(u), author_org.get(v)
        if not ou or not ov:
            continue
        w = d.get("weight", 1)
        if ou == ov:
            OG.nodes[ou]["intra_weight"] += w
        else:
            key = tuple(sorted((ou, ov)))
            inter[key] += w
            inter_papers[key].extend(d.get("papers", []) or [])

    for (a, b) in inter:
        # de-duplicate papers shared between the two organizations
        seen, papers = set(), []
        for p in inter_papers[(a, b)]:
            pk = (p.get("title"), p.get("year"))
            if pk not in seen:
                seen.add(pk)
                papers.append(p)
        # weight = number of distinct papers co-authored across the two orgs
        OG.add_edge(a, b, weight=len(papers) if papers else inter[(a, b)], papers=papers)

    return OG


def search_orgs(query, org_names, limit=12, score_cutoff=60):
    """Search organizations by substring first, then fuzzy match."""
    if not query or not query.strip():
        return []
    q = query.strip().lower()
    exact = [(o, min(100, len(q) / max(len(o), 1) * 100 + 50)) for o in org_names if q in o.lower()]
    if exact:
        exact.sort(key=lambda x: (-x[1], x[0]))
        # keep the top `limit` by relevance, then show them alphabetically
        return sorted([o for o, _ in exact[:limit]], key=lambda o: o.lower())
    res = process.extract(query, org_names, scorer=fuzz.WRatio, limit=limit)
    return sorted([o for o, s, _ in res if s >= score_cutoff], key=lambda o: o.lower())


def build_org_ego(OG, center, max_degree=1):
    """Ego subgraph of the organization graph centered on `center`."""
    if center not in OG:
        return nx.Graph()
    H = nx.Graph()
    levels = {center: 0}
    queue = [(center, 0)]
    while queue:
        node, lvl = queue.pop(0)
        if lvl >= max_degree:
            continue
        for nbr in OG.neighbors(node):
            if nbr not in levels:
                levels[nbr] = lvl + 1
                queue.append((nbr, lvl + 1))
    for n, lvl in levels.items():
        H.add_node(n, level=lvl, **OG.nodes[n])
    for n in levels:
        for nbr in OG.neighbors(n):
            if nbr in levels:
                H.add_edge(n, nbr, weight=OG[n][nbr].get("weight", 1),
                           papers=OG[n][nbr].get("papers", []))
    return H


def plot_org_ego(H, center):
    """Plotly ego network of organizations, centered on `center`.

    Returns (figure, edge_paper_lookup) where the lookup maps an edge key
    to its two organizations and the papers they share.
    """
    if H.number_of_nodes() == 0:
        return None, {}

    n = H.number_of_nodes()
    k = 6 / np.sqrt(n) if n > 1 else 1
    pos = nx.spring_layout(H, seed=42, k=k, iterations=200, scale=3)

    edge_paper_lookup = {}
    edge_traces = []
    mid_x, mid_y, mid_labels, mid_keys = [], [], [], []

    for u, v in H.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        w = H[u][v].get("weight", 1)
        papers = H[u][v].get("papers", [])
        edge_key = f"{u}|||{v}"
        edge_paper_lookup[edge_key] = {"u": u, "v": v, "papers": papers, "weight": w}

        _ew, _ec = edge_visual(w)
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=_ew, color=_ec),
            hoverinfo="skip", showlegend=False
        ))
        mid_x.append(xm)
        mid_y.append(ym)
        mid_labels.append(f"{u} & {v}: {w} shared paper(s)\n(click to see details below)")
        mid_keys.append(edge_key)

    # Invisible midpoint markers — clicking one triggers a selection event
    edge_traces.append(go.Scatter(
        x=mid_x, y=mid_y,
        mode="markers",
        marker=dict(size=14, color="rgba(0,0,0,0)", line=dict(width=0)),
        hoverinfo="text",
        hovertext=mid_labels,
        hoverlabel=dict(bgcolor="white", bordercolor="#aaa", font=dict(size=14)),
        customdata=mid_keys,
        showlegend=False,
        name="edges"
    ))

    level_colors = {0: "#d62828", 1: "#2a9d8f", 2: "#457b9d"}
    xs, ys, txt, col, siz, lab = [], [], [], [], [], []
    for node in H.nodes():
        x, y = pos[node]
        xs.append(x); ys.append(y); lab.append(node)
        lvl = H.nodes[node].get("level", 2)
        na = H.nodes[node].get("num_authors", 0)
        txt.append(f"<b>{node}</b><br>Affiliated authors: {na}<br>Degree from center: {lvl}")
        col.append(level_colors.get(lvl, "#6c757d"))
        siz.append(46 if lvl == 0 else (28 if lvl == 1 else 20))

    show_labels = n <= 30
    node_trace = go.Scatter(
        x=xs, y=ys,
        mode="markers+text" if show_labels else "markers",
        text=lab if show_labels else None,
        textposition="top center",
        textfont=dict(size=15, color="#222"),
        hoverinfo="text", hovertext=txt,
        hoverlabel=dict(bgcolor="white", bordercolor="#aaa", font=dict(size=15)),
        marker=dict(size=siz, color=col, line=dict(width=2, color="white"), opacity=0.9),
        showlegend=False
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        showlegend=False, plot_bgcolor="#f8f9fa",
        margin=dict(l=5, r=5, t=5, b=5), height=600,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        dragmode="pan", hovermode="closest"
    )
    return fig, edge_paper_lookup


@st.cache_data
def get_org_papers(_df, _G, org, mode="full"):
    """Papers within scope that include at least one author affiliated with `org`."""
    author_org = build_org_mapping_from_graph(_G)
    members = {a for a, o in author_org.items() if o == org}
    if not members:
        return pd.DataFrame(), []
    rows = []
    for _, row in _df.iterrows():
        auth = set(parse_authors(row["Authors"]))
        hit = members & auth
        if hit:
            rows.append({
                "Year": int(row["Year"]) if pd.notna(row["Year"]) else None,
                "Title": row["Title"],
                "Domain": row.get("Domain") or "",
                "Org authors": ", ".join(sorted(hit)),
                "Link": row.get("Link") or "",
            })
    pdf = pd.DataFrame(rows)
    if not pdf.empty:
        pdf = pdf.sort_values("Year", ascending=False, na_position="last").reset_index(drop=True)
    return pdf, sorted(members)


# =============================================================================
# Data Loading
# =============================================================================

@st.cache_data
def load_dataframe():
    return pd.read_parquet(os.path.join("data", "papers_bibliography.parquet"))

@st.cache_resource
def load_graph():
    with open(os.path.join("data", "coauthor_graph_bibliography.pkl"), "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_author_stats():
    return pd.read_parquet(os.path.join("data", "author_stats_bibliography.parquet"))

@st.cache_data
def get_all_authors_sorted(_G, mode="full"):
    # Filter out problematic author names and deduplicate by normalized name
    excluded = {'Unknown', 'Anonymous', 'unknown', 'anonymous', ''}
    
    # Use a dict to deduplicate by normalized name, keeping the "best" version
    seen_normalized = {}
    for a in _G.nodes():
        if a in excluded or len(a) <= 2:
            continue
        norm = normalize_author_name(a)
        if norm in excluded or len(norm) <= 2:
            continue
        # Keep the version with more info (longer name, or first seen)
        if norm not in seen_normalized or len(a) > len(seen_normalized[norm]):
            seen_normalized[norm] = a
    
    return sorted(seen_normalized.values())

@st.cache_data
def get_all_countries(_author_stats):
    countries = _author_stats['Country'].dropna().unique()
    return sorted([c for c in countries if c])

@st.cache_data
def get_all_orgs(_author_stats):
    orgs = _author_stats['Organization'].dropna().unique()
    return sorted([o.strip() for o in orgs if o and o.strip()])

@st.cache_data
def get_author_org_mapping(_author_stats):
    mapping = {}
    for _, row in _author_stats.iterrows():
        if pd.notna(row.get('Organization')) and str(row['Organization']).strip():
            mapping[row['Author']] = row['Organization'].strip()
            mapping[normalize_author_name(row['Author'])] = row['Organization'].strip()
    return mapping


@st.cache_resource
def load_graph_conference():
    with open(os.path.join("data", "coauthor_graph_conference.pkl"), "rb") as f:
        return pickle.load(f)


# Load data (full dataset + conference-only subset)
df_full = load_dataframe()
G_full = load_graph()
G_conf = load_graph_conference()
author_stats = load_author_stats()

# Dataset-wide filter option lists and author -> org map
all_countries = get_all_countries(author_stats)
all_orgs = get_all_orgs(author_stats)
author_org_mapping = get_author_org_mapping(author_stats)


# =============================================================================
# Page Config
# =============================================================================

st.set_page_config(page_title="System Dynamics Collaboration Explorer (Demo)", layout="wide")

# Custom CSS for tab styling (matching the conference proceedings app)
st.markdown("""
    <style>
    /* Make tabs more prominent */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: white;
        border-radius: 8px;
        border: 1px solid #ddd;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4 !important;
        color: white !important;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)

# Main title
st.title("System Dynamics Collaboration Explorer (Demo)")

# Dataset scope filter (in-page, applies to every tab)
if "conf_only" not in st.session_state:
    st.session_state.conf_only = False
conference_only = st.toggle(
    "Conference proceedings only",
    key="conf_only",
)
mode = "conference" if conference_only else "full"

if conference_only:
    df = df_full[df_full["is_conference"]].copy()
    G = G_conf
else:
    df = df_full
    G = G_full
all_authors_sorted = get_all_authors_sorted(G, mode=mode)

_scope = "Conference proceedings only" if conference_only else "Full bibliography"
if len(df):
    st.caption(
        f"**{_scope}** · Exploring **{len(df):,}** papers and "
        f"**{G.number_of_nodes():,}** authors"
    )
else:
    st.caption(f"**{_scope}** · No papers in scope")


# =============================================================================
# Tab Navigation
# =============================================================================

# Note: the "Forrester Number" tab has been disabled. It is removed from the
# tab bar here and its body is commented out further below (search for
# "Forrester Number tab"). The supporting functions are kept intact so the
# tab can be restored easily if requested.
tab1, tab2, tab_pf, tab_on = st.tabs(
    ["Authors", "Co-authors", "Co-author Distance",
     "Organizations"]
)


# =============================================================================
# Tab 1: Top Authors
# =============================================================================

with tab1:
    st.header("Authors")
    
    st.markdown("Explore authors in the System Dynamics bibliography.")
    
    # Filters
    col_year, col_country, col_org = st.columns([2, 1, 1])
    
    with col_year:
        _yr_min = int(df["Year"].min())
        _yr_max = int(df["Year"].max())
        # Default the lower bound to 1956 (the year Jay Forrester became an MIT
        # professor), but never outside the data's actual range.
        _yr_default_low = min(max(1956, _yr_min), _yr_max)
        year_min, year_max = st.slider(
            "Year range",
            min_value=_yr_min,
            max_value=_yr_max,
            value=(_yr_default_low, _yr_max),
            key="top_authors_year"
        )
    
    with col_country:
        selected_countries = st.multiselect(
            "Filter by country",
            options=all_countries,
            default=[],
            key="top_authors_country"
        )
    
    with col_org:
        selected_orgs = st.multiselect(
            "Filter by organization",
            options=all_orgs,
            default=[],
            key="top_authors_org"
        )

    # Publication type filter (independent of the "Conference proceedings only"
    # scope toggle above; this filters by the document type of each entry).
    if "category" in df.columns:
        present_types = set(df["category"].dropna().unique())
        pub_type_options = [t for t in PUB_TYPE_ORDER if t in present_types]
        # Include any unexpected/extra categories at the end so nothing is hidden.
        pub_type_options += sorted(present_types - set(pub_type_options))
        selected_pub_types = st.multiselect(
            "Publication type",
            options=pub_type_options,
            default=pub_type_options,
            key="top_authors_pubtype",
            help="Filter authors by the type of entry their papers come from. "
                 "This is separate from the Conference proceedings only toggle."
        )
    else:
        selected_pub_types = None
    
    col_min_papers, col_min_coauth, col_top_n = st.columns(3)
    
    with col_min_papers:
        min_papers = st.number_input("Min papers", min_value=1, value=1, key="min_papers")
    
    with col_min_coauth:
        min_coauthors = st.number_input("Min co-authors", min_value=0, value=0, key="min_coauth")
    
    with col_top_n:
        top_n = st.slider("Number of authors to show", 10, 200, 50, key="top_n_authors")
    
    # Filter papers by year
    df_filtered = df[df["Year"].between(year_min, year_max)]
    # Filter by publication type when the column is present and a subset is chosen
    if selected_pub_types is not None and "category" in df_filtered.columns:
        if len(selected_pub_types) < len(pub_type_options):
            df_filtered = df_filtered[df_filtered["category"].isin(selected_pub_types)]
    df_filtered = df_filtered.copy()
    
    # Count papers per author in filtered range
    ap = df_filtered[["Authors"]].copy()
    ap["Author"] = ap["Authors"].apply(parse_authors)
    ap = ap.explode("Author")
    ap = ap[ap["Author"].notna() & (ap["Author"] != "")]
    
    author_counts = ap.groupby("Author").size().rename("NumPapers_Filtered").reset_index()
    
    # Merge with author_stats (properly deduplicated)
    author_stats_norm = author_stats.copy()
    author_stats_norm["Author"] = author_stats_norm["Author"].apply(normalize_author_name)
    author_stats_norm = author_stats_norm.groupby("Author", as_index=False).agg({
        "NumPapers": "sum",
        "NumCoauthors": "max",
        "Country": "first",
        "Organization": "first"
    })
    
    tbl = author_stats_norm.merge(author_counts, on="Author", how="left")
    tbl["NumPapers_Filtered"] = tbl["NumPapers_Filtered"].fillna(0).astype(int)
    
    # Filter out problematic author names
    excluded_authors = {'Unknown', 'Anonymous', 'unknown', 'anonymous', ''}
    tbl = tbl[~tbl["Author"].isin(excluded_authors)]
    tbl = tbl[tbl["Author"].str.len() > 2]  # Filter very short names
    
    # Apply filters
    tbl = tbl[(tbl["NumPapers_Filtered"] >= min_papers) & (tbl["NumCoauthors"] >= min_coauthors)]
    
    if selected_countries:
        tbl = tbl[tbl["Country"].isin(selected_countries)]
    
    if selected_orgs:
        tbl = tbl[tbl["Organization"].isin(selected_orgs)]
    
    st.divider()
    
    # Author table
    st.subheader("Authors")
    st.caption(f"**{len(tbl):,}** authors match the filters")
    
    tbl_show = tbl.sort_values(["NumPapers_Filtered", "NumCoauthors"], ascending=False).head(top_n)
    
    tbl_display = tbl_show[["Author", "NumPapers_Filtered", "NumPapers", "NumCoauthors", "Country", "Organization"]].copy()
    tbl_display = tbl_display.rename(columns={
        "NumPapers_Filtered": "Papers (filtered)",
        "NumPapers": "Total Papers",
        "NumCoauthors": "Co-authors"
    })
    tbl_display.index = range(1, len(tbl_display) + 1)
    
    st.dataframe(tbl_display, use_container_width=True)
    
    # Network visualization
    st.subheader("Network")
    
    max_nodes = st.slider("Max nodes to display", 25, 400, 25, key="max_nodes_tab1")
    
    top_authors = tbl.sort_values(["NumPapers_Filtered", "NumCoauthors"], ascending=False).head(max_nodes)["Author"].tolist()
    
    # Build subgraph
    H = nx.Graph()
    
    for a in top_authors:
        if a in G:
            H.add_node(a, **G.nodes[a])
    
    for a in top_authors:
        if a in G:
            for b in G.neighbors(a):
                if b in H:
                    weight = G[a][b].get("weight", 1)
                    H.add_edge(a, b, weight=weight)

    # Remove authors who share no paper with anyone else in the displayed set.
    # These isolated nodes carry no co-authorship information and otherwise
    # float far from the rest of the layout, which looks like a glitch.
    _isolated = list(nx.isolates(H))
    if _isolated:
        H.remove_nodes_from(_isolated)

    if H.number_of_nodes() > 0:
        n = H.number_of_nodes()
        k = 8 / np.sqrt(n) if n > 1 else 1
        pos = nx.spring_layout(H, seed=42, k=k, iterations=300, scale=3)
        
        # Edges
        edge_traces = []
        for u, v in H.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            weight = H[u][v].get("weight", 1)
            _ew, _ec = edge_visual(weight)
            edge_traces.append(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode="lines",
                line=dict(width=_ew, color=_ec),
                hoverinfo="skip",
                showlegend=False
            ))
        
        # Nodes
        node_x, node_y, node_text, node_sizes, node_colors = [], [], [], [], []
        node_names = []
        
        papers_vals = [int(H.nodes[n].get("num_papers") or 1) for n in H.nodes()]
        max_papers = max(papers_vals) if papers_vals else 1
        min_papers_val = min(papers_vals) if papers_vals else 1
        
        coauth_vals = [int(H.nodes[n].get("num_coauthors") or 0) for n in H.nodes()]
        max_coauth = max(coauth_vals) if coauth_vals else 1
        
        for node in H.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_names.append(node)
            
            num_papers = int(H.nodes[node].get("num_papers") or 1)
            num_coauthors = int(H.nodes[node].get("num_coauthors") or 0)
            country = H.nodes[node].get("country") or ""
            org = H.nodes[node].get("organization") or ""
            
            hover = f"<b>{node}</b><br>Papers: {num_papers}<br>Co-authors: {num_coauthors}"
            if country:
                hover += f"<br>Country: {country}"
            if org:
                hover += f"<br>Org: {org}"
            node_text.append(hover)
            
            # Size by papers
            if max_papers > min_papers_val:
                norm = (num_papers - min_papers_val) / (max_papers - min_papers_val)
                norm = max(0.0, min(1.0, norm))  # clamp to [0, 1]
            else:
                norm = 0.5
            node_sizes.append(float(15 + norm ** 0.5 * 65))
            
            # Color by coauthors
            if max_coauth > 0:
                norm_c = num_coauthors / max_coauth
            else:
                norm_c = 0
            node_colors.append(norm_c)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers",
            hoverinfo="text",
            hovertext=node_text,
            hoverlabel=dict(bgcolor="white", bordercolor="#aaa", font=dict(size=15)),
            marker=dict(
                size=node_sizes,
                color=node_colors,
                colorscale="Tealgrn",
                colorbar=dict(title="Co-authors", thickness=15, x=1.02),
                line=dict(width=2, color="white"),
                opacity=0.9
            ),
            showlegend=False
        )
        
        fig = go.Figure(data=edge_traces + [node_trace])
        fig.update_layout(
            showlegend=False,
            plot_bgcolor="#f8f9fa",
            margin=dict(l=5, r=80, t=5, b=5),
            height=600,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            dragmode="pan",
            hovermode="closest"
        )
        
        _omitted = len(_isolated)
        st.markdown(
            "<div style='font-size:15px; color:#333; line-height:1.6;'>"
            f"Showing <b>{H.number_of_nodes()}</b> authors and "
            f"<b>{H.number_of_edges()}</b> co-authorship links.<br>"
            "<b>Node size</b> = total papers &nbsp;·&nbsp; "
            "<b>Node color</b> = number of co-authors &nbsp;·&nbsp; "
            "<b>Edge thickness</b> = number of shared papers."
            "</div>",
            unsafe_allow_html=True
        )
        if _omitted:
            st.caption(
                f"{_omitted} selected author(s) share no paper with anyone else "
                "in this set and are not shown in the network."
            )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No nodes to display.")


# =============================================================================
# Tab 2: Find Co-authors
# =============================================================================

with tab2:
    st.header("Co-authors")
    
    st.markdown("Search for an author to explore their co-author network and see paper details.")
    
    author_query = st.text_input("Search for an author", key="coauthor_search")
    
    if author_query:
        candidates = search_authors(author_query, all_authors_sorted, limit=10, score_cutoff=60)
        
        if not candidates:
            st.info("No matching authors found.")
        else:
            author_names = [name for name, score in candidates]
            selected_author = st.radio("Select an author:", options=author_names, key="coauthor_select")
            
            if selected_author:
                st.markdown(f"**Selected author:** {selected_author}")
                
                # Author info
                if selected_author in G.nodes:
                    info = G.nodes[selected_author]
                    cols = st.columns(4)
                    cols[0].metric("Total Papers", info.get("num_papers", 0))
                    cols[1].metric("Co-authors", info.get("num_coauthors", 0))
                    cols[2].metric("Country", info.get("country") or "—")
                    cols[3].metric("Organization", info.get("organization") or "—")
                
                # Degree selector
                max_degree = st.radio(
                    "Degrees of separation",
                    options=[1, 2, 3],
                    index=0,
                    horizontal=True,
                    help="1 = direct co-authors only"
                )
                
                # Co-author tables (cached)
                degree_dfs = get_coauthors_by_degree(G, selected_author, max_degree=max_degree, mode=mode)
                
                if degree_dfs:
                    degree_labels = ["1st degree (direct)", "2nd degree", "3rd degree"]
                    cols = st.columns(min(len(degree_dfs), 3))
                    
                    for i, (col, degree_df) in enumerate(zip(cols, degree_dfs)):
                        with col:
                            st.subheader(degree_labels[i])
                            if degree_df.empty:
                                st.write("None found.")
                            else:
                                st.caption(f"{len(degree_df)} authors")
                                degree_df.index = range(1, len(degree_df) + 1)
                                st.dataframe(degree_df, use_container_width=True)
                
                # Network visualization
                st.markdown("---")
                
                H = build_coauthor_network(G, selected_author, max_degree=max_degree)
                
                if H.number_of_nodes() > 0:
                    fig, edge_paper_lookup = plot_coauthor_network(H, selected_author)
                    
                    st.subheader("Co-author Network")
                    
                    # Legend — color key on the first line, reading instructions
                    # on a second line for clarity.
                    _swatch = (
                        lambda color, label:
                        '<span style="display:inline-flex; align-items:center; margin-right:18px;">'
                        f'<span style="width:14px; height:14px; border-radius:50%; '
                        f'background-color:{color}; margin-right:6px;"></span>'
                        f'<span style="color:#333; font-size:15px;">{label}</span></span>'
                    )
                    legend_html = (
                        '<div style="margin-bottom:6px;">'
                        + _swatch("#d62828", "Selected author")
                        + _swatch("#2a9d8f", "1st degree")
                        + _swatch("#457b9d", "2nd degree")
                        + _swatch("#8338ec", "3rd degree")
                        + '</div>'
                        '<div style="color:#333; font-size:15px; line-height:1.6; margin-bottom:8px;">'
                        '<b>Edge thickness</b> = number of shared papers.<br>'
                        'Click a connection (its midpoint) to see the shared papers below.'
                        '</div>'
                    )
                    st.markdown(legend_html, unsafe_allow_html=True)

                    # Use a counter in session state to reset chart selection when cleared
                    clear_key = f"coauthor_net_clear_{selected_author}"
                    if clear_key not in st.session_state:
                        st.session_state[clear_key] = 0

                    # Render chart and capture clicks
                    event = st.plotly_chart(
                        fig,
                        use_container_width=True,
                        on_select="rerun",
                        selection_mode="points",
                        key=f"coauthor_net_{selected_author}_{st.session_state[clear_key]}"
                    )

                    # Panel: show papers for clicked edge
                    clicked_key = None
                    if event and event.selection and event.selection.get("points"):
                        for pt in event.selection["points"]:
                            cd = pt.get("customdata")
                            if cd and "|||" in str(cd):
                                clicked_key = str(cd)
                                break

                    if clicked_key and clicked_key in edge_paper_lookup:
                        edata = edge_paper_lookup[clicked_key]
                        u, v = edata["u"], edata["v"]
                        papers = edata["papers"]
                        col_title, col_clear = st.columns([6, 1])
                        col_title.markdown(f"**Shared papers: {u} & {v}**")
                        if col_clear.button("Clear", key=f"clear_btn_{selected_author}"):
                            st.session_state[clear_key] += 1
                            st.rerun()
                        if papers:
                            for p in sorted(papers, key=lambda x: x.get("year") or 0, reverse=True):
                                year  = p.get("year") or "?"
                                title = p.get("title") or "(no title)"
                                link  = p.get("link")
                                if link:
                                    st.markdown(f"- [{title}]({link}) ({year})")
                                else:
                                    st.markdown(f"- {title} ({year})")
                        else:
                            st.markdown(f"- {edata['weight']} shared paper(s) (no details available)")
                    else:
                        st.caption("Click on a connection line (midpoint) to see the shared papers between two authors.")
                else:
                    st.info("No co-author network to display.")


# =============================================================================
# Tab 3: Forrester Number
# =============================================================================

REFERENCE_AUTHOR = "Jay Wright Forrester"


def _minimize_crossings(nodes_by_level, edges_set):
    """
    Reorder nodes within each level (below level 0) to reduce edge crossings.
    Uses a simple barycenter heuristic: sort each level's nodes by the average
    x-position of their neighbours in the level above.
    Returns a new nodes_by_level dict with reordered lists.
    """
    # Start with a temporary uniform x assignment for level 0
    result = {}
    num_levels = max(nodes_by_level.keys()) + 1
    max_w = max(len(v) for v in nodes_by_level.values())
    chart_width = max(max_w * 3.0, 6.0)

    # Assign initial positions top-down
    temp_x = {}
    for lvl in sorted(nodes_by_level.keys()):
        nodes = list(nodes_by_level[lvl])
        n = len(nodes)
        if n == 1:
            xs = [chart_width / 2]
        else:
            xs = [chart_width * i / (n - 1) for i in range(n)]
        for node, x in zip(nodes, xs):
            temp_x[node] = x
        result[lvl] = nodes

    # Build adjacency: node -> neighbours in adjacent levels
    adj = defaultdict(set)
    for u, v in edges_set:
        adj[u].add(v)
        adj[v].add(u)

    # Barycenter sweep: top-down
    for lvl in range(1, num_levels):
        if lvl not in nodes_by_level:
            continue
        nodes = result[lvl]
        scores = []
        for node in nodes:
            neighbours_above = [nb for nb in adj[node] if nb in temp_x]
            if neighbours_above:
                bary = sum(temp_x[nb] for nb in neighbours_above) / len(neighbours_above)
            else:
                bary = temp_x.get(node, chart_width / 2)
            scores.append((bary, node))
        scores.sort()
        reordered = [node for _, node in scores]
        result[lvl] = reordered
        # Update temp_x for this level
        n = len(reordered)
        if n == 1:
            xs = [chart_width / 2]
        else:
            xs = [chart_width * i / (n - 1) for i in range(n)]
        for node, x in zip(reordered, xs):
            temp_x[node] = x

    return result


def plot_forrester_path_tree(all_paths, reference_node, selected_author, distance_label="Forrester Number"):
    """
    Draw a top-down family tree showing only the nodes involved in the
    shortest path(s) from Jay Forrester down to the selected author.
    Jay Forrester is at the top (level 0). Each row below is one hop further.
    Nodes within each level are reordered to minimise edge crossings.
    """
    if not all_paths:
        return None

    # Collect unique nodes per level and edges across all paths
    nodes_by_level = defaultdict(set)
    edges_set = set()

    for path in all_paths:
        # paths go selected_author → ... → reference_node; reverse so Jay = level 0
        reversed_path = list(reversed(path))
        for i, node in enumerate(reversed_path):
            nodes_by_level[i].add(node)
        for i in range(len(reversed_path) - 1):
            u, v = reversed_path[i], reversed_path[i + 1]
            edges_set.add((u, v))

    # Initial alphabetical sort before crossing-reduction
    nodes_by_level = {lvl: sorted(nodes) for lvl, nodes in nodes_by_level.items()}

    # Reorder to minimise crossings
    nodes_by_level = _minimize_crossings(nodes_by_level, edges_set)

    num_levels = len(nodes_by_level)
    max_nodes_in_level = max(len(nodes) for nodes in nodes_by_level.values())
    chart_width = max(max_nodes_in_level * 3.0, 6.0)
    y_gap = 4.5  # more vertical breathing room between levels

    node_pos = {}
    for lvl, nodes in nodes_by_level.items():
        n = len(nodes)
        if n == 1:
            xs = [chart_width / 2]
        else:
            xs = [chart_width * i / (n - 1) for i in range(n)]
        y = -lvl * y_gap
        for node, x in zip(nodes, xs):
            node_pos[node] = (x, y)

    level_colors = {
        0: "#d4a017",
        1: "#2a9d8f",
        2: "#457b9d",
        3: "#8338ec",
        4: "#e76f51",
        5: "#6c757d",
        6: "#adb5bd",
    }

    traces = []

    # Edges
    edge_x, edge_y = [], []
    for u, v in edges_set:
        if u not in node_pos or v not in node_pos:
            continue
        x0, y0 = node_pos[u]
        x1, y1 = node_pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    traces.append(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=2.5, color="rgba(120,120,120,0.5)"),
        hoverinfo="skip",
        showlegend=False
    ))

    # Nodes
    node_x, node_y, node_labels, node_hover = [], [], [], []
    node_colors, node_sizes, node_borders = [], [], []

    for lvl, nodes in nodes_by_level.items():
        for node in nodes:
            x, y = node_pos[node]
            node_x.append(x)
            node_y.append(y)
            node_labels.append(node)
            node_hover.append(f"<b>{node}</b><br>{distance_label}: {lvl}")

            if node == selected_author:
                node_colors.append("#d62828")
                node_sizes.append(20)
                node_borders.append("#8b0000")
            elif node == reference_node:
                node_colors.append("#d4a017")
                node_sizes.append(20)
                node_borders.append("#7a5200")
            else:
                node_colors.append(level_colors.get(lvl, "#6c757d"))
                node_sizes.append(16)
                node_borders.append("white")

    traces.append(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_labels,
        textposition="top center",
        textfont=dict(size=15, color="#222222"),
        hoverinfo="text",
        hovertext=node_hover,
        hoverlabel=dict(bgcolor="white", bordercolor="#aaa", font=dict(size=15)),
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color=node_borders),
            opacity=0.95
        ),
        showlegend=False
    ))

    fig_height = max(380, num_levels * 190)

    fig = go.Figure(data=traces)
    fig.update_layout(
        showlegend=False,
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="#f8f9fa",
        margin=dict(l=30, r=30, t=30, b=30),
        height=fig_height,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-1, chart_width + 1]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        dragmode="pan",
        hovermode="closest"
    )

    return fig


# =============================================================================
# Precompute all Forrester distances (cached)
# =============================================================================

@st.cache_data
def get_all_forrester_distances(_G, reference_node, mode="full"):
    """
    BFS from reference_node to get shortest path distance for every reachable node.
    Returns a dict: author -> distance
    """
    return nx.single_source_shortest_path_length(_G, reference_node)


# =============================================================================
# Forrester Number tab (DISABLED)
# -----------------------------------------------------------------------------
# This tab has been turned off at the client's request. The body is preserved
# (commented out) so it can be restored later: uncomment the block below and
# add "Forrester Number" back into the st.tabs(...) call near the top, restoring
# the tab3 variable. The helper functions it relies on are left intact above.
# =============================================================================
# with tab3:
#     st.header("Forrester Number")
#
#     st.markdown(
#         f"""
#         Find your Forrester Number — the degrees of co-authorship separation from {REFERENCE_AUTHOR}.
#
#         - Forrester Number 1: You co-authored a paper directly with {REFERENCE_AUTHOR}
#         - Forrester Number 2: You co-authored with someone who co-authored with {REFERENCE_AUTHOR}
#         - And so on
#         """
#     )
#
#     # Check if reference author exists in graph
#     reference_in_graph = None
#     for node in G.nodes():
#         if normalize_author_name(REFERENCE_AUTHOR) == normalize_author_name(node):
#             reference_in_graph = node
#             break
#
#     if reference_in_graph is None:
#         st.error(f"**{REFERENCE_AUTHOR}** not found in the co-author network.")
#     else:
#         info = G.nodes[reference_in_graph]
#         st.caption(
#             f"**{reference_in_graph}** · {info.get('num_papers', 0)} papers · "
#             f"{info.get('num_coauthors', 0)} direct co-authors."
#         )
#
#         author_query_tab3 = st.text_input("Search for an author", key="forrester_search")
#
#         if author_query_tab3:
#             candidates = search_authors(author_query_tab3, all_authors_sorted, limit=10, score_cutoff=60)
#
#             if not candidates:
#                 st.info("No matching authors found.")
#             else:
#                 author_names_tab3 = [name for name, score in candidates]
#                 selected_author_tab3 = st.radio("Select an author:", options=author_names_tab3, key="forrester_select")
#
#                 if selected_author_tab3:
#                     st.markdown("---")
#
#                     if selected_author_tab3 == reference_in_graph:
#                         st.success(f"**{selected_author_tab3}** IS {REFERENCE_AUTHOR}! Forrester Number = **0**")
#
#                     elif not nx.has_path(G, selected_author_tab3, reference_in_graph):
#                         st.warning(
#                             f"**{selected_author_tab3}** is not connected to {REFERENCE_AUTHOR} "
#                             "in the co-author network."
#                         )
#
#                     else:
#                         try:
#                             forrester_number = nx.shortest_path_length(G, selected_author_tab3, reference_in_graph)
#                             st.success(f"**{selected_author_tab3}** has a Forrester Number of **{forrester_number}**")
#
#                             # Collect up to 10 shortest paths
#                             all_paths = []
#                             for i, path in enumerate(nx.all_shortest_paths(G, selected_author_tab3, reference_in_graph)):
#                                 all_paths.append(path)
#                                 if i >= 9:
#                                     break
#
#                             if len(all_paths) == 1:
#                                 st.markdown(f"**Path to {REFERENCE_AUTHOR}:**")
#                             else:
#                                 st.markdown(f"**{len(all_paths)} shortest paths to {REFERENCE_AUTHOR}:**")
#
#                             for path in all_paths:
#                                 st.markdown("- " + " → ".join(path))
#
#                             # Family tree visualization
#                             st.markdown("---")
#                             st.subheader("Path Tree")
#
#                             fig = plot_forrester_path_tree(all_paths, reference_in_graph, selected_author_tab3)
#                             if fig:
#                                 st.plotly_chart(fig, use_container_width=True)
#                                 st.caption(
#                                     "**How to read this:** Jay Wright Forrester is at the top. "
#                                     "Each row below represents one additional degree of separation. "
#                                     "Lines connect co-authors. Where multiple paths exist, "
#                                     "shared intermediaries appear once with lines converging into them. "
#                                     "Hover over any node for details."
#                                 )
#
#                         except Exception as e:
#                             st.error(f"Error finding path: {str(e)}")





# =============================================================================
# Author Connections (distance & paths between any two authors)
# =============================================================================

with tab_pf:
    st.header("Co-author Distance")
    st.markdown(
        "Find the co-authorship distance and the shortest path(s) between any two authors."
    )

    left, right = st.columns([1, 1.5])

    # --- Inputs: Author A above Author B, in the left column ---
    pf_sel_a = pf_sel_b = None
    with left:
        pf_query_a = st.text_input("Author A", key="pf_query_a")
        if pf_query_a:
            pf_cand_a = search_authors(pf_query_a, all_authors_sorted, limit=10, score_cutoff=60)
            if pf_cand_a:
                pf_sel_a = st.radio("Select Author A:", options=[n for n, _ in pf_cand_a], key="pf_sel_a")
            else:
                st.info("No matching author found for Author A.")

        pf_query_b = st.text_input("Author B", key="pf_query_b")
        if pf_query_b:
            pf_cand_b = search_authors(pf_query_b, all_authors_sorted, limit=10, score_cutoff=60)
            if pf_cand_b:
                pf_sel_b = st.radio("Select Author B:", options=[n for n, _ in pf_cand_b], key="pf_sel_b")
            else:
                st.info("No matching author found for Author B.")

    # --- Result: prominent distance and shortest path(s), in the right column ---
    pf_paths = []
    pf_ready = False
    with right:
        if pf_sel_a and pf_sel_b:
            if pf_sel_a == pf_sel_b:
                st.markdown(
                    "<div style='background:#eef4fb; border:1px solid #cfe0f3; border-radius:10px; "
                    "padding:18px 22px;'>"
                    "<div style='font-size:64px; font-weight:700; color:#1f4e79; line-height:1;'>0</div>"
                    "<div style='font-size:17px; color:#333; margin-top:8px;'>"
                    "Author A and Author B are the same person.</div></div>",
                    unsafe_allow_html=True,
                )
            elif pf_sel_a not in G or pf_sel_b not in G:
                st.warning("One of the selected authors isn't in the current network scope.")
            elif not nx.has_path(G, pf_sel_a, pf_sel_b):
                scope_note = " (conference scope)" if mode == "conference" else ""
                st.warning(
                    f"**{pf_sel_a}** and **{pf_sel_b}** are not connected "
                    f"in the co-author network{scope_note}."
                )
            else:
                try:
                    pf_dist = nx.shortest_path_length(G, pf_sel_a, pf_sel_b)
                    step_word = "step" if pf_dist == 1 else "steps"
                    st.markdown(
                        "<div style='background:#eef4fb; border:1px solid #cfe0f3; border-radius:10px; "
                        "padding:18px 22px;'>"
                        f"<div style='font-size:64px; font-weight:700; color:#1f4e79; line-height:1;'>{pf_dist}</div>"
                        f"<div style='font-size:17px; color:#333; margin-top:8px;'>co-authorship {step_word} between<br>"
                        f"<b>{pf_sel_a}</b> and <b>{pf_sel_b}</b></div></div>",
                        unsafe_allow_html=True,
                    )

                    for i, path in enumerate(nx.all_shortest_paths(G, pf_sel_a, pf_sel_b)):
                        pf_paths.append(path)
                        if i >= 9:
                            break

                    st.markdown("")
                    if len(pf_paths) == 1:
                        st.markdown("**Shortest path:**")
                    else:
                        st.markdown(f"**{len(pf_paths)} shortest paths:**")
                    for path in pf_paths:
                        st.markdown("- " + " → ".join(path))
                    pf_ready = True
                except Exception as e:
                    st.error(f"Error finding path: {str(e)}")
        else:
            st.caption("Select an Author A and an Author B to see how far apart they are.")

    # --- Path tree spans the full width below the two columns ---
    if pf_ready and pf_paths:
        st.markdown("---")
        st.subheader("Path Tree")
        pf_fig = plot_forrester_path_tree(pf_paths, pf_sel_b, pf_sel_a, distance_label="Distance")
        if pf_fig:
            st.plotly_chart(pf_fig, use_container_width=True)
            st.caption(
                f"**{pf_sel_b}** is at the top; each row down is one more co-authorship step "
                f"toward **{pf_sel_a}**. Hover any node for details."
            )


# =============================================================================
# Organization Network
# =============================================================================

with tab_on:
    st.header("Organizations")
    st.markdown(
        "Search for an organization to explore the other organizations it has "
        "collaborated with, based on its authors co-authoring papers together."
    )

    OG = build_org_network(G, mode=mode)
    org_names = sorted(OG.nodes())
    st.caption(
        f"**{len(org_names):,}** organizations with a recorded affiliation in scope. "
        "Affiliation data is sparse, so only authors with a known organization appear here."
    )

    if not org_names:
        st.info("No organization data available in the current scope.")
    else:
        org_query = st.text_input("Search for an organization", key="orgnet_search")
        if org_query:
            matches = search_orgs(org_query, org_names)
            if not matches:
                st.info("No matching organizations found.")
            else:
                selected_org = st.radio("Select an organization:", options=matches, key="orgnet_select")
                if selected_org:
                    na = OG.nodes[selected_org].get("num_authors", 0)
                    partners = sorted(
                        ((nbr, OG[selected_org][nbr].get("weight", 1)) for nbr in OG.neighbors(selected_org)),
                        key=lambda x: -x[1],
                    )

                    c1, c2 = st.columns(2)
                    c1.metric("Affiliated authors", na)
                    c2.metric("Partner organizations", len(partners))

                    if partners:
                        st.subheader("Collaborating organizations")
                        ptbl = pd.DataFrame(partners, columns=["Organization", "Shared papers"])
                        ptbl.index = range(1, len(ptbl) + 1)
                        st.dataframe(ptbl, use_container_width=True)

                        st.markdown("---")
                        st.subheader("Organization network")
                        max_degree = st.radio(
                            "Degrees of separation", options=[1, 2], index=0, horizontal=True,
                            key="orgnet_degree",
                        )
                        st.markdown(
                            "<div style='font-size:15px; color:#333; line-height:1.6; margin-bottom:8px;'>"
                            "<b>Center (red)</b> = selected organization &nbsp;·&nbsp; "
                            "<b>Node size</b> = number of affiliated authors &nbsp;·&nbsp; "
                            "<b>Edge thickness</b> = number of shared papers.<br>"
                            "Click a connection (its midpoint) to see the shared papers below."
                            "</div>",
                            unsafe_allow_html=True,
                        )

                        H = build_org_ego(OG, selected_org, max_degree=max_degree)
                        if H.number_of_edges() > 0:
                            org_fig, org_edge_lookup = plot_org_ego(H, selected_org)

                            clear_key = f"orgnet_clear_{selected_org}"
                            if clear_key not in st.session_state:
                                st.session_state[clear_key] = 0

                            event = st.plotly_chart(
                                org_fig,
                                use_container_width=True,
                                on_select="rerun",
                                selection_mode="points",
                                key=f"orgnet_chart_{selected_org}_{st.session_state[clear_key]}"
                            )

                            clicked_key = None
                            if event and event.selection and event.selection.get("points"):
                                for pt in event.selection["points"]:
                                    cd = pt.get("customdata")
                                    if cd and "|||" in str(cd):
                                        clicked_key = str(cd)
                                        break

                            if clicked_key and clicked_key in org_edge_lookup:
                                edata = org_edge_lookup[clicked_key]
                                u, v = edata["u"], edata["v"]
                                papers = edata["papers"]
                                col_title, col_clear = st.columns([6, 1])
                                col_title.markdown(f"**Shared papers: {u} & {v}**")
                                if col_clear.button("Clear", key=f"orgnet_clear_btn_{selected_org}"):
                                    st.session_state[clear_key] += 1
                                    st.rerun()
                                if papers:
                                    for p in sorted(papers, key=lambda x: x.get("year") or 0, reverse=True):
                                        year = p.get("year") or "?"
                                        title = p.get("title") or "(no title)"
                                        link = p.get("link")
                                        if link:
                                            st.markdown(f"- [{title}]({link}) ({year})")
                                        else:
                                            st.markdown(f"- {title} ({year})")
                                else:
                                    st.markdown(f"- {edata['weight']} shared paper(s) (no details available)")
                            else:
                                st.caption("Click on a connection line (midpoint) to see the shared papers between two organizations.")
                    else:
                        st.info("No cross-organization collaborations recorded for this organization.")
