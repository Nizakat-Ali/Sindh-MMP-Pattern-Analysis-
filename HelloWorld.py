"""
Sindh MMP Network Analysis - Visual Outputs (All-in-One Script)
--------------------------------------------------------------
Inputs:
  - Excel file: Sindh MMP Analysis.xlsx
Required columns:
  - ORIGIN DISTRICT
  - DISTRICT NAME
  - # OF CHILDREN
Optional:
  - # OF FAMILIES
  - ORIGIN COUNTRY (filter to PAKISTAN)

Outputs (saved in ./outputs):
  - network_graph.png   (directed weighted network)
  - sankey.html         (top corridors sankey)
  - od_heatmap.png      (origin-destination heatmap)
  - network_metrics.csv (basic node metrics)
"""

import os
import sys
import numpy as np
import pandas as pd

import networkx as nx
import matplotlib.pyplot as plt

import plotly.graph_objects as go


# =========================
# CONFIG (edit if needed)
# =========================
FILE_PATH = "Sindh MMP Analysis.xlsx"
SHEET_NAME = 0  # or "Sheet1"

ORIGIN_COL = "ORIGIN DISTRICT"
DEST_COL   = "DISTRICT NAME"
CHILD_COL  = "# OF CHILDREN"
FAM_COL    = "# OF FAMILIES"
COUNTRY_COL = "ORIGIN COUNTRY"

FILTER_COUNTRY = "PAKISTAN"   # set None to disable
TOP_EDGES_NETWORK = 60        # used in network plot (reduce clutter)
TOP_EDGES_SANKEY  = 30        # used in sankey plot
TOP_ORIGINS_HEAT  = 20
TOP_DESTS_HEAT    = 12

OUTPUT_DIR = "outputs"


# =========================
# HELPERS
# =========================
def clean_text(series: pd.Series) -> pd.Series:
    return (series.astype(str)
                  .str.strip()
                  .str.replace(r"\s+", " ", regex=True)
                  .str.upper())

def require_columns(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns are:\n{list(df.columns)}"
        )

def safe_numeric(df: pd.DataFrame, col: str, fill=0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series([fill] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(fill)

def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)


# =========================
# MAIN
# =========================
def main():
    ensure_output_dir(OUTPUT_DIR)

    # ---- Load ----
    if not os.path.exists(FILE_PATH):
        print(f"ERROR: File not found at: {FILE_PATH}")
        print("Edit FILE_PATH in the script to your local path.")
        sys.exit(1)

    df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)

    # ---- Validate required columns ----
    require_columns(df, [ORIGIN_COL, DEST_COL, CHILD_COL])

    # ---- Clean text columns ----
    df[ORIGIN_COL] = clean_text(df[ORIGIN_COL])
    df[DEST_COL]   = clean_text(df[DEST_COL])

    if COUNTRY_COL in df.columns:
        df[COUNTRY_COL] = clean_text(df[COUNTRY_COL])

    # ---- Numeric safety ----
    df[CHILD_COL] = safe_numeric(df, CHILD_COL, fill=0.0)
    df[FAM_COL]   = safe_numeric(df, FAM_COL, fill=0.0)

    # ---- Optional filter (Pakistan only) ----
    if FILTER_COUNTRY and (COUNTRY_COL in df.columns):
        df = df[df[COUNTRY_COL] == FILTER_COUNTRY].copy()

    # ---- Build edge list (OD aggregation) ----
    edges = (df.groupby([ORIGIN_COL, DEST_COL], as_index=False)
               .agg(children=(CHILD_COL, "sum"),
                    families=(FAM_COL, "sum")))

    edges = edges.sort_values("children", ascending=False)

    print("Total OD edges (origin->destination pairs):", len(edges))
    if len(edges) == 0:
        raise ValueError("No edges left after filtering. Check your filters/columns.")

    # =========================
    # 1) Directed Network Graph
    # =========================
    edges_top = edges.head(TOP_EDGES_NETWORK).copy()

    G = nx.DiGraph()
    for _, r in edges_top.iterrows():
        G.add_edge(r[ORIGIN_COL], r[DEST_COL], weight=float(r["children"]))

    # Node metrics (for sizing)
    in_strength = dict(G.in_degree(weight="weight"))
    out_strength = dict(G.out_degree(weight="weight"))

    # Betweenness (weighted)
    # For large graphs, this can be slow; here graph is limited to TOP_EDGES_NETWORK
    betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)

    # Save node metrics to CSV
    metrics_df = pd.DataFrame({
        "node": list(G.nodes()),
        "in_strength_children": [in_strength.get(n, 0.0) for n in G.nodes()],
        "out_strength_children": [out_strength.get(n, 0.0) for n in G.nodes()],
        "betweenness": [betweenness.get(n, 0.0) for n in G.nodes()],
    }).sort_values(["in_strength_children", "out_strength_children"], ascending=False)

    metrics_path = os.path.join(OUTPUT_DIR, "network_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print("Saved:", metrics_path)

    # Node sizes (scale by sqrt of in-strength)
    node_sizes = []
    for n in G.nodes():
        s = in_strength.get(n, 0.0)
        node_sizes.append(200 + (np.sqrt(s) * 20))

    # Edge widths scale
    edge_weights = np.array([G[u][v]["weight"] for u, v in G.edges()])
    edge_widths = 0.5 + 6 * (edge_weights / edge_weights.max())

    # Layout (seed for reproducibility)
    pos = nx.spring_layout(G, k=0.6, seed=42)

    plt.figure(figsize=(18, 11))
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes)
    nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=12,
        alpha=0.6
    )
    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title("Directed Weighted Network: ORIGIN DISTRICT → SINDH DISTRICT (Weight = # Children)")
    plt.axis("off")
    plt.tight_layout()

    network_img_path = os.path.join(OUTPUT_DIR, "network_graph.png")
    plt.savefig(network_img_path, dpi=300)
    plt.close()
    print("Saved:", network_img_path)

    # =========================
    # 2) Sankey Diagram (HTML)
    # =========================
    sankey_df = edges.head(TOP_EDGES_SANKEY).copy()

    all_nodes = pd.Index(pd.concat([sankey_df[ORIGIN_COL], sankey_df[DEST_COL]]).unique())
    node_to_id = {name: i for i, name in enumerate(all_nodes)}

    sources = sankey_df[ORIGIN_COL].map(node_to_id).tolist()
    targets = sankey_df[DEST_COL].map(node_to_id).tolist()
    values  = sankey_df["children"].tolist()

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=18,
            label=all_nodes.tolist()
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values
        )
    )])

    fig.update_layout(
        title_text=f"Sankey: Top {TOP_EDGES_SANKEY} Migration Corridors (Weight = # Children)",
        font_size=10,
        height=650
    )

    sankey_path = os.path.join(OUTPUT_DIR, "sankey.html")
    fig.write_html(sankey_path)
    print("Saved:", sankey_path)

    # =========================
    # 3) OD Heatmap (PNG)
    # =========================
    top_origins = (edges.groupby(ORIGIN_COL)["children"].sum()
                   .sort_values(ascending=False).head(TOP_ORIGINS_HEAT).index)

    top_dests = (edges.groupby(DEST_COL)["children"].sum()
                 .sort_values(ascending=False).head(TOP_DESTS_HEAT).index)

    sub = edges[edges[ORIGIN_COL].isin(top_origins) & edges[DEST_COL].isin(top_dests)].copy()
    mat = sub.pivot_table(index=ORIGIN_COL, columns=DEST_COL, values="children", aggfunc="sum", fill_value=0)

    plt.figure(figsize=(16, 9))
    plt.imshow(mat.values, aspect="auto")  # no explicit color specified
    plt.xticks(range(mat.shape[1]), mat.columns, rotation=60, ha="right")
    plt.yticks(range(mat.shape[0]), mat.index)
    plt.title(f"OD Heatmap: Top {TOP_ORIGINS_HEAT} Origins × Top {TOP_DESTS_HEAT} Sindh Destinations (Children)")
    plt.xlabel("Sindh Destination District")
    plt.ylabel("Origin District")
    plt.colorbar(label="# Children")
    plt.tight_layout()

    heatmap_path = os.path.join(OUTPUT_DIR, "od_heatmap.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print("Saved:", heatmap_path)

    print("\nDONE ✅")
    print(f"All outputs are in: {os.path.abspath(OUTPUT_DIR)}")
    
    # =========================
    # 4) Generate plots by destination district
    # =========================
    BY_DISTRICT_DIR = os.path.join(OUTPUT_DIR, "by_district")
    ensure_output_dir(BY_DISTRICT_DIR)
    
    TOP_ORIGINS_PER_DISTRICT = 25   # for graphs & bars
    TOP_ORIGINS_SANKEY = 20         # sankey readability
    
    # List of destination districts available
    destinations = edges[DEST_COL].dropna().unique().tolist()
    destinations = sorted(destinations)
    
    print(f"\nCreating district-wise visuals for {len(destinations)} destination districts...")
    
    for dest in destinations:
        # ---- Filter edges for this destination district ----
        e = edges[edges[DEST_COL] == dest].copy()
        if e.empty:
            continue
    
        # Keep top origins to avoid clutter
        e = e.sort_values("children", ascending=False).head(TOP_ORIGINS_PER_DISTRICT)
    
        # Create a safe folder name
        safe_dest = "".join([c if c.isalnum() or c in (" ", "-", "_") else "_" for c in dest]).strip().replace(" ", "_")
        out_dir = os.path.join(BY_DISTRICT_DIR, safe_dest)
        ensure_output_dir(out_dir)
    
        # =========================================================
        # A) Mini Network Graph for this district (origins -> dest)
        # =========================================================
        Gd = nx.DiGraph()
        for _, r in e.iterrows():
            Gd.add_edge(r[ORIGIN_COL], dest, weight=float(r["children"]))
    
        # Node sizes (dest bigger)
        nodes = list(Gd.nodes())
        sizes = []
        for n in nodes:
            if n == dest:
                sizes.append(1800)  # destination node bigger
            else:
                w = Gd[n][dest]["weight"] if Gd.has_edge(n, dest) else 0
                sizes.append(300 + (np.sqrt(w) * 20))
    
        # Edge widths
        wts = np.array([Gd[u][v]["weight"] for u, v in Gd.edges()])
        widths = 0.8 + 6 * (wts / wts.max()) if len(wts) > 0 else 1.5
    
        # Layout: put destination on right
        pos = {}
        origin_nodes = [n for n in nodes if n != dest]
        # Spread origins vertically
        for i, n in enumerate(origin_nodes):
            pos[n] = (0, i)
        pos[dest] = (2, len(origin_nodes) / 2)
    
        plt.figure(figsize=(14, max(6, 0.35 * len(origin_nodes))))
        nx.draw_networkx_nodes(Gd, pos, node_size=sizes)
        nx.draw_networkx_edges(
            Gd, pos, width=widths,
            arrows=True, arrowstyle="-|>", arrowsize=14, alpha=0.7
        )
        nx.draw_networkx_labels(Gd, pos, font_size=9)
    
        plt.title(f"Migration Network into {dest} (Top {len(e)} Origin Corridors) | Weight=# Children")
        plt.axis("off")
        plt.tight_layout()
    
        net_path = os.path.join(out_dir, f"{safe_dest}_network.png")
        plt.savefig(net_path, dpi=300)
        plt.close()
    
        # =========================================================
        # B) Bar Chart: Top origin districts for this destination
        # =========================================================
        bar_df = e.sort_values("children", ascending=True)
        plt.figure(figsize=(12, max(6, 0.35 * len(bar_df))))
        plt.barh(bar_df[ORIGIN_COL], bar_df["children"])
        plt.title(f"Top Origin Districts → {dest} (Children)")
        plt.xlabel("# Children")
        plt.ylabel("Origin District")
        plt.tight_layout()
    
        bar_path = os.path.join(out_dir, f"{safe_dest}_top_origins_bar.png")
        plt.savefig(bar_path, dpi=300)
        plt.close()
    
        # =========================================================
        # C) Sankey (optional): origins -> this destination
        # =========================================================
        s = edges[edges[DEST_COL] == dest].copy().sort_values("children", ascending=False).head(TOP_ORIGINS_SANKEY)
        if not s.empty:
            node_labels = pd.Index(pd.concat([s[ORIGIN_COL], s[DEST_COL]]).unique())
            node_id = {name: i for i, name in enumerate(node_labels)}
    
            sources = s[ORIGIN_COL].map(node_id).tolist()
            targets = s[DEST_COL].map(node_id).tolist()
            values = s["children"].tolist()
    
            fig = go.Figure(data=[go.Sankey(
                node=dict(pad=15, thickness=18, label=node_labels.tolist()),
                link=dict(source=sources, target=targets, value=values)
            )])
    
            fig.update_layout(
                title_text=f"Sankey: Top {TOP_ORIGINS_SANKEY} Corridors into {dest} (Children)",
                font_size=10, height=650
            )
    
            sankey_path = os.path.join(out_dir, f"{safe_dest}_sankey.html")
            fig.write_html(sankey_path)
    
    print("✅ District-wise visuals saved in:", os.path.abspath(BY_DISTRICT_DIR))
    
    # Open visuals in browser
    import webbrowser
    webbrowser.open("file://" + os.path.abspath(network_img_path))
    webbrowser.open("file://" + os.path.abspath(heatmap_path))
    webbrowser.open("file://" + os.path.abspath(sankey_path))

if __name__ == "__main__":
    main()
