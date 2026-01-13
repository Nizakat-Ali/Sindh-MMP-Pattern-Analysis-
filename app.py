"""
Interactive Sindh MMP Analysis Dashboard
-----------------------------------------
Streamlit app with filters for:
- Destination district (Sindh)
- Origin district
- MMP subtype
- Weight type (Children / Families)
- Top-N corridors
- View mode: Sankey / Network / Heatmap / Tables / Metrics
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image
import folium
from streamlit_folium import st_folium

# ================== CONFIG ==================
FILE_PATH = "Sindh MMP Analysis.xlsx"
SHEET_NAME = 0

ORIGIN_COL = "ORIGIN DISTRICT"
DEST_COL = "DISTRICT NAME"
CHILD_COL = "# OF CHILDREN"
FAM_COL = "# OF FAMILIES"
COUNTRY_COL = "ORIGIN COUNTRY"
MMP_COL = "MMP SUBTYPE"

SINDH_DISTRICTS = [
    "KHIEAST", "KHIWEST", "KHISOUTH", "KHICENTRAL", "KHIKEAMARI", "KHIMALIR", "KHIKORANGI",
    "HYDERABAD", "BADIN", "JAMSHORO", "DADU", "MATIARI", "SUKKUR", "LARKANA", "GHOTKI",
    "SHIKARPUR", "JACOBABAD", "KHAIRPUR", "UMERKOT", "THARPARKAR", "MIRPURKHAS", "SANGHAR",
    "NFEROZ", "TMKHAN", "THATTA", "SBENAZIRABAD", "KAMBAR", "KASHMORE", "SUJAWAL"
]

# ================== CACHE & LOAD ==================
@st.cache_data
def load_data():
    if not os.path.exists(FILE_PATH):
        st.error(f"File not found: {FILE_PATH}")
        st.stop()
    
    df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)
    
    # Clean text columns
    for col in [ORIGIN_COL, DEST_COL]:
        df[col] = df[col].astype(str).str.strip().str.upper()
    
    if MMP_COL in df.columns:
        df[MMP_COL] = df[MMP_COL].astype(str).str.strip().str.upper()
    
    if COUNTRY_COL in df.columns:
        df[COUNTRY_COL] = df[COUNTRY_COL].astype(str).str.strip().str.upper()
    
    # Numeric safety
    df[CHILD_COL] = pd.to_numeric(df[CHILD_COL], errors="coerce").fillna(0)
    df[FAM_COL] = pd.to_numeric(df[FAM_COL], errors="coerce").fillna(0)
    
    # Filter Pakistan
    if COUNTRY_COL in df.columns:
        df = df[df[COUNTRY_COL] == "PAKISTAN"].copy()
    
    return df

def clean_text(s):
    return s.astype(str).str.strip().str.upper()

# ================== STREAMLIT CONFIG ==================
st.set_page_config(page_title="Sindh MMP Analysis", layout="wide")
st.title("🔍 Sindh Migration & Mobility Profile (MMP) Analysis")

# Load data
df = load_data()

# ================== SIDEBAR FILTERS ==================
st.sidebar.header("Filters")

# Weight type
weight_type = st.sidebar.radio("Weight Type", ["Children", "Families"])
weight_col = CHILD_COL if weight_type == "Children" else FAM_COL

# Destination district (Sindh only) - MULTI-SELECT
all_dests = sorted(df[DEST_COL].unique())
sindh_dests = [d for d in all_dests if d in SINDH_DISTRICTS]
selected_dest = st.sidebar.multiselect("Destination District (Sindh)", sindh_dests, default=sindh_dests[:1])

# Origin district
all_origins = sorted(df[ORIGIN_COL].unique())
selected_origin = st.sidebar.multiselect("Origin District", all_origins, default=[])

# MMP Subtype (if exists)
mmp_subtypes = []
if MMP_COL in df.columns:
    mmp_subtypes = sorted(df[MMP_COL].dropna().unique())

selected_mmp = st.sidebar.multiselect("MMP Subtype", mmp_subtypes, default=[])

# Top-N corridors
top_n = st.sidebar.slider("Top-N Corridors", 5, 100, 30)

# Map tile style (for Map view)
map_tile = st.sidebar.selectbox("Map Style", [
    "OpenStreetMap",
    "CartoDB Positron",
    "CartoDB Voyager",
    "Stamen Terrain",
    "Stamen Toner",
    "OpenTopoMap"
])

# View mode
view_mode = st.sidebar.selectbox("View Mode", ["Sankey", "Network", "Heatmap", "Tables", "Metrics", "Map"])

# ================== FILTER DATA ==================
filtered_df = df.copy()

# Filter by destination (multi-select)
if selected_dest:
    filtered_df = filtered_df[filtered_df[DEST_COL].isin(selected_dest)]

# Filter by origin
if selected_origin:
    filtered_df = filtered_df[filtered_df[ORIGIN_COL].isin(selected_origin)]

# Filter by MMP subtype
if selected_mmp:
    filtered_df = filtered_df[filtered_df[MMP_COL].isin(selected_mmp)]

# Build aggregated edges
if len(filtered_df) > 0:
    edges = filtered_df.groupby([ORIGIN_COL, DEST_COL], as_index=False).agg({
        weight_col: "sum",
        CHILD_COL: "sum",
        FAM_COL: "sum"
    }).rename(columns={weight_col: "weight"})
    
    edges = edges.sort_values("weight", ascending=False)
    edges_top = edges.head(top_n).copy()
else:
    edges = pd.DataFrame()
    edges_top = pd.DataFrame()

# ================== DISPLAY STATS ==================
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Records", len(filtered_df))
with col2:
    st.metric("Unique Origins", filtered_df[ORIGIN_COL].nunique())
with col3:
    st.metric("Unique Destinations", filtered_df[DEST_COL].nunique())
with col4:
    st.metric("Total Weight", int(filtered_df[weight_col].sum()))

st.divider()

# ================== VIEW MODES ==================

if len(edges_top) == 0:
    st.warning("No data matches the selected filters.")
else:
    
    # ================== SANKEY ==================
    if view_mode == "Sankey":
        st.subheader("📊 Sankey Diagram")
        
        col1, col2 = st.columns(2)
        with col1:
            sankey_direction = st.radio("Flow Direction", ["Origins → Destinations", "Destinations → Origins"], horizontal=False)
        with col2:
            sankey_type = st.radio("Show By", ["OD Pairs", "MMP Type"], horizontal=False)
        
        all_nodes = None
        sources = None
        targets = None
        values = None
        title_suffix = ""
        
        if sankey_type == "OD Pairs":
            # Standard Origin-Destination Sankey
            if sankey_direction == "Origins → Destinations":
                all_nodes = pd.Index(pd.concat([edges_top[ORIGIN_COL], edges_top[DEST_COL]]).unique())
                node_to_id = {name: i for i, name in enumerate(all_nodes)}
                
                sources = edges_top[ORIGIN_COL].map(node_to_id).tolist()
                targets = edges_top[DEST_COL].map(node_to_id).tolist()
            else:  # Destinations → Origins (reversed)
                all_nodes = pd.Index(pd.concat([edges_top[DEST_COL], edges_top[ORIGIN_COL]]).unique())
                node_to_id = {name: i for i, name in enumerate(all_nodes)}
                
                sources = edges_top[DEST_COL].map(node_to_id).tolist()
                targets = edges_top[ORIGIN_COL].map(node_to_id).tolist()
            
            values = edges_top["weight"].tolist()
            title_suffix = f"({sankey_direction})"
        
        else:  # By MMP Type
            # Aggregate by MMP Type
            if MMP_COL not in filtered_df.columns:
                st.warning("MMP Type data not available in the dataset.")
                all_nodes = []
            else:
                if sankey_direction == "Origins → Destinations":
                    # Origin -> MMP Type -> Destination (show MMP type in middle)
                    mmp_agg = filtered_df.groupby([ORIGIN_COL, MMP_COL, DEST_COL], as_index=False).agg({weight_col: "sum"})
                    
                    # Create a multi-level sankey: Origin -> MMP Type
                    origins_mmp = filtered_df.groupby([ORIGIN_COL, MMP_COL], as_index=False).agg({weight_col: "sum"})
                    all_nodes_list = list(origins_mmp[ORIGIN_COL].unique()) + list(origins_mmp[MMP_COL].unique())
                    all_nodes = pd.Index(all_nodes_list).unique()
                    node_to_id = {name: i for i, name in enumerate(all_nodes)}
                    
                    sources = origins_mmp[ORIGIN_COL].map(node_to_id).tolist()
                    targets = origins_mmp[MMP_COL].map(node_to_id).tolist()
                    values = origins_mmp[weight_col].tolist()
                else:
                    # Destination -> MMP Type -> Origin (reversed)
                    dest_mmp = filtered_df.groupby([DEST_COL, MMP_COL], as_index=False).agg({weight_col: "sum"})
                    all_nodes_list = list(dest_mmp[DEST_COL].unique()) + list(dest_mmp[MMP_COL].unique())
                    all_nodes = pd.Index(all_nodes_list).unique()
                    node_to_id = {name: i for i, name in enumerate(all_nodes)}
                    
                    sources = dest_mmp[DEST_COL].map(node_to_id).tolist()
                    targets = dest_mmp[MMP_COL].map(node_to_id).tolist()
                    values = dest_mmp[weight_col].tolist()
                
                title_suffix = f"by MMP Type ({sankey_direction})"
        
        if all_nodes is not None and len(all_nodes) > 0:
            fig = go.Figure(data=[go.Sankey(
                node=dict(pad=15, thickness=18, label=all_nodes.tolist()),
                link=dict(source=sources, target=targets, value=values)
            )])
            
            fig.update_layout(
                title_text=f"Top {top_n} Migration Corridors ({weight_type}) {title_suffix}",
                font_size=10,
                height=700,
                margin=dict(l=50, r=50, t=50, b=50)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Unable to create Sankey diagram with selected filters.")
    
    # ================== NETWORK ==================
    elif view_mode == "Network":
        st.subheader("🕸️ Network Graph")
        
        G = nx.DiGraph()
        for _, r in edges_top.iterrows():
            G.add_edge(r[ORIGIN_COL], r[DEST_COL], weight=float(r["weight"]))
        
        # Node metrics
        in_strength = dict(G.in_degree(weight="weight"))
        out_strength = dict(G.out_degree(weight="weight"))
        
        # Node sizes
        node_sizes = []
        for n in G.nodes():
            s = in_strength.get(n, 0.0)
            node_sizes.append(300 + (np.sqrt(s) * 15))
        
        # Edge widths
        edge_weights = np.array([G[u][v]["weight"] for u, v in G.edges()])
        edge_widths = 0.5 + 4 * (edge_weights / edge_weights.max())
        
        # Layout
        pos = nx.spring_layout(G, k=1.5, seed=42)
        
        fig, ax = plt.subplots(figsize=(14, 9))
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, ax=ax)
        nx.draw_networkx_edges(G, pos, width=edge_widths, arrows=True, arrowstyle="-|>", 
                               arrowsize=12, alpha=0.6, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
        
        ax.set_title(f"Directed Network: {weight_type} Flow", fontsize=14, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
        
        st.pyplot(fig)
    
    # ================== HEATMAP ==================
    elif view_mode == "Heatmap":
        st.subheader("🔥 Origin-Destination Heatmap")
        
        mat = edges_top.pivot_table(index=ORIGIN_COL, columns=DEST_COL, 
                                     values="weight", aggfunc="sum", fill_value=0)
        
        # Mask zero values for better visualization
        mat_masked = np.ma.masked_where(mat == 0, mat)
        
        fig, ax = plt.subplots(figsize=(12, max(6, 0.3 * len(mat))))
        im = ax.imshow(mat_masked, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        ax.set_xticks(range(mat.shape[1]))
        ax.set_yticks(range(mat.shape[0]))
        ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(mat.index, fontsize=9)
        ax.set_title(f"OD Heatmap: {weight_type} (White = No Data)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Destination District")
        ax.set_ylabel("Origin District")
        cbar = plt.colorbar(im, ax=ax, label=f"# {weight_type}")
        plt.tight_layout()
        
        st.pyplot(fig)
    
    # ================== TABLES ==================
    elif view_mode == "Tables":
        st.subheader("📋 Top Corridors (OD Pairs)")
        
        display_cols = [ORIGIN_COL, DEST_COL, CHILD_COL, FAM_COL]
        display_cols = [c for c in display_cols if c in edges_top.columns]
        display_df = edges_top[display_cols].copy()
        
        st.dataframe(display_df.reset_index(drop=True), use_container_width=True)
        
        # Download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"corridors_{weight_type.lower()}.csv",
            mime="text/csv"
        )
        
        st.divider()
        st.subheader("📊 Origins Summary")
        origins_summary = filtered_df.groupby(ORIGIN_COL).agg({
            weight_col: "sum",
            CHILD_COL: "sum",
            FAM_COL: "sum"
        }).rename(columns={weight_col: weight_type}).sort_values(weight_type, ascending=False)
        st.dataframe(origins_summary, use_container_width=True)
        
        st.divider()
        st.subheader("📍 Destinations Summary")
        dests_summary = filtered_df.groupby(DEST_COL).agg({
            weight_col: "sum",
            CHILD_COL: "sum",
            FAM_COL: "sum"
        }).rename(columns={weight_col: weight_type}).sort_values(weight_type, ascending=False)
        st.dataframe(dests_summary, use_container_width=True)
    
    # ================== METRICS ==================
    elif view_mode == "Metrics":
        st.subheader("📈 Network Metrics")
        
        G = nx.DiGraph()
        for _, r in edges_top.iterrows():
            G.add_edge(r[ORIGIN_COL], r[DEST_COL], weight=float(r["weight"]))
        
        in_strength = dict(G.in_degree(weight="weight"))
        out_strength = dict(G.out_degree(weight="weight"))
        betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)
        
        metrics_df = pd.DataFrame({
            "Node": list(G.nodes()),
            "In-Strength": [in_strength.get(n, 0.0) for n in G.nodes()],
            "Out-Strength": [out_strength.get(n, 0.0) for n in G.nodes()],
            "Betweenness": [betweenness.get(n, 0.0) for n in G.nodes()],
        }).sort_values(["In-Strength", "Out-Strength"], ascending=False)
        
        st.dataframe(metrics_df.reset_index(drop=True), use_container_width=True)
        
        # Download
        csv = metrics_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Metrics as CSV",
            data=csv,
            file_name="network_metrics.csv",
            mime="text/csv"
        )
    
    # ================== MAP ==================
    elif view_mode == "Map":
        st.subheader("🗺️ Migration Flow Map")
        
        # Sindh district coordinates (approximate centers)
        DISTRICT_COORDS = {
            # Sindh
            "KHIEAST": [24.8607, 67.0011],
            "KHIWEST": [24.8607, 66.8011],
            "KHISOUTH": [24.7607, 67.0011],
            "KHICENTRAL": [24.8607, 67.0011],
            "KHIKEAMARI": [24.8607, 67.1011],
            "KHIMALIR": [24.9607, 67.1011],
            "KHIKORANGI": [24.9607, 67.0011],
            "HYDERABAD": [25.2768, 68.3639],
            "BADIN": [24.6552, 68.8389],
            "JAMSHORO": [25.2580, 68.2580],
            "DADU": [26.7372, 67.8175],
            "MATIARI": [25.5031, 68.7456],
            "SUKKUR": [27.7054, 68.8656],
            "LARKANA": [27.5614, 68.2170],
            "GHOTKI": [27.7629, 67.8397],
            "SHIKARPUR": [27.9536, 68.6411],
            "JACOBABAD": [28.2805, 68.4386],
            "KHAIRPUR": [27.5269, 68.7647],
            "UMERKOT": [25.3667, 69.7333],
            "THARPARKAR": [24.8000, 70.2500],
            "MIRPURKHAS": [25.5244, 69.0150],
            "SANGHAR": [25.9342, 68.9547],
            "NFEROZ": [26.1333, 68.3833],
            "TMKHAN": [25.8769, 68.4711],
            "THATTA": [24.7606, 67.8667],
            "SBENAZIRABAD": [27.0833, 68.0167],
            "KAMBAR": [27.1872, 68.0558],
            "KASHMORE": [27.5156, 67.9208],
            "SUJAWAL": [24.0794, 67.6594],
            # Punjab
            "LAHORE": [31.5497, 74.3436],
            "FAISALABAD": [31.4181, 72.9879],
            "MULTAN": [30.1575, 71.4243],
            "GUJRANWALA": [32.1814, 74.1855],
            "RAWALPINDI": [33.5731, 73.1492],
            "GUJRAT": [32.1736, 74.0681],
            "SIALKOT": [32.4968, 74.5303],
            "JHANG": [31.2750, 72.3186],
            "OKARA": [30.8057, 73.6516],
            "SAHIWAL": [30.6719, 73.1114],
            "BAHAWALNAGAR": [29.9945, 73.2580],
            "BAHAWALPUR": [29.3992, 71.6837],
            "KASUR": [31.1255, 74.4396],
            "LODHRAN": [29.5397, 71.6324],
            "KHANEWAL": [30.3028, 71.9294],
            "PAKPATTAN": [30.4371, 73.4239],
            "RAJANPUR": [29.1125, 70.3244],
            "MUZAFFARGARH": [30.0756, 71.1939],
            "LAYYAH": [30.6699, 70.9247],
            "SHEIKHUPURA": [31.7164, 73.9769],
            "SARGODHA": [32.0859, 72.6421],
            "KHUSHAB": [32.2735, 72.3525],
            "MIANWALI": [32.5934, 71.5522],
            "BHAKKAR": [31.6315, 71.4867],
            "JHELUM": [32.9314, 73.7387],
            "CHAKWAL": [32.9298, 72.8586],
            "ATTOCK": [33.7687, 72.7265],
            "GUJRANWALA": [32.1814, 74.1855],
            "HAFIZABAD": [32.0521, 73.6850],
            "MANDI BAHAUDDIN": [32.5806, 74.6094],
            "NANKANA SAHIB": [31.9088, 74.6392],
            "CHINIOT": [31.7270, 72.9806],
            "TOBA TEK SINGH": [30.9735, 72.4797],
            # Balochistan
            "QUETTA": [30.1798, 67.0089],
            "PESHAWAR": [34.0151, 71.5249],
            "GWADAR": [25.1207, 62.3306],
            "DALBANDIN": [28.4906, 65.5417],
            "KHUZDAR": [27.8139, 66.6325],
            "ZHOB": [31.3369, 69.1822],
            "LORALAI": [30.3596, 68.5797],
            "SIBI": [29.5511, 67.8733],
            "KECH": [25.5036, 61.8722],
            "PANJGUR": [26.9244, 64.0764],
            "MASTUNG": [29.8159, 66.8725],
            "KALAT": [28.9703, 66.5881],
            "CHAMAN": [30.9213, 65.4587],
            "KANDAHAR": [31.6257, 65.7245],
            "SPINBOLDAK": [31.9719, 65.2910],
            # KP
            "MARDAN": [34.1972, 72.0386],
            "SWAT": [34.7656, 72.4273],
            "PESHAWAR": [34.0151, 71.5249],
            "ABBOTTABAD": [34.1526, 72.2093],
            "KOHAT": [33.5837, 71.4308],
            "BANNU": [32.7733, 70.6158],
            "MANSEHRA": [34.3167, 73.1833],
            "HARIPUR": [33.9964, 72.9244],
            "NOWSHERA": [33.9872, 71.9733],
            "CHARSADA": [34.1500, 71.7333],
            "SWABI": [34.1167, 72.4667],
            "BUNER": [34.5667, 72.5000],
            "MALAKAND": [34.5269, 71.9358],
            "SHANGLA": [34.7500, 72.6167],
            "BATTAGRAM": [34.9333, 72.8333],
            "KOHISTAN": [35.3500, 73.1667],
            "TANK": [32.8333, 70.3667],
            "WAZIR": [32.9333, 70.1667],
            # Gilgit-Baltistan
            "GILGIT": [35.9272, 74.3143],
            "SKARDU": [35.2854, 75.5731],
            "HUNZA": [36.8500, 74.8833],
            "GHIZER": [36.7500, 74.5000],
            "NAGAR": [36.8167, 74.6667],
            # ICT
            "ICT": [33.6844, 73.0479],
            "ISLAMABAD": [33.6844, 73.0479],
            # Additional districts
            "BAGH": [33.5333, 73.6667],
            "BHIMBER": [32.8000, 74.0000],
            "BOLAN": [28.8500, 67.0000],
            "DIAMER": [35.8500, 74.5000],
            "DIKU": [32.9000, 71.0000],
            "KURRAM": [33.8333, 70.0000],
            "KHYBER": [34.3500, 71.5000],
            "KOHISTANUPPER": [34.9000, 73.5000],
            "KOHISTANLOWER": [34.1000, 72.5000],
            "MIRPUR": [33.1500, 73.7500],
            "MUZAFFARABAD": [34.3667, 73.4833],
            "ORAKZAI": [33.8000, 71.1000],
            "POONCH": [33.7000, 73.5000],
            "SUDHNOTI": [33.1500, 73.9000],
            "TANDK": [33.2500, 70.7500],
            "BAGHNEWAZAI": [30.2000, 71.5000],
            "BARKHAN": [30.0833, 69.3667],
            "PISHIN": [31.3667, 66.8333],
            "QUETTA": [30.1798, 67.0089],
            "SHERANI": [30.6333, 68.4833],
            "ZIARAT": [31.8000, 69.4000],
            "DUKKI": [31.2333, 68.9000],
            "HARNAI": [30.5333, 67.9000],
            "JHAL MAGSI": [29.3000, 67.5000],
            "NASIRABAD": [27.3667, 66.5667],
            "WASHUK": [28.0000, 64.0000],
            "CHAGHAI": [29.4167, 65.3667],
            "NUSHKI": [29.6333, 66.3500],
            "AWARAN": [26.5000, 64.5000],
            "KOLAIPALAS": [29.0000, 67.0000],
            "DALBANDI": [28.4906, 65.5417],
            "DERA BUGTI": [28.8333, 68.0000],
            "DERA ISMAIL KHAN": [31.8629, 70.9273],
            "DICKHAN": [30.5333, 70.2000],
            "LAKKI MARWAT": [32.5167, 70.9333],
            "BANNU": [32.7733, 70.6158],
            "WAZIR-N": [32.9333, 70.1667],
            "WAZIR-S Upper": [32.8000, 70.5000],
            "WAZIR-S Lower": [32.7000, 70.3000],
            # Afghanistan
            "KABUL": [34.5264, 69.1789],
            "KANDAHAR": [31.6257, 65.7245],
            "HERAT": [34.3425, 62.2000],
            "NANGARHAR": [34.4167, 70.4500],
            "KUNAR": [35.2333, 71.1667],
            "KHOST": [33.3333, 69.9167],
            "PAKTYA": [33.7667, 69.3167],
            "PAKTIKA": [33.0333, 68.8333],
            "LOGAR": [33.9333, 69.4833],
            "WARDAK": [34.3167, 68.3333],
            "BAGHLAN": [36.7500, 68.7500],
            "BALKH": [36.7500, 67.5000],
            "JAWZJAN": [37.1333, 65.4167],
            "SAR-I PUL": [36.2167, 65.9333],
            "SAMANGAN": [36.5000, 67.5000],
            "TAKHAR": [37.1167, 69.5167],
            "BADAKHSHAN": [36.8333, 71.5000],
            "KUNDUZ": [36.7333, 68.8667],
            "LAGHMAN": [34.6667, 70.5000],
            "NURISTAN": [35.5000, 70.5000],
            "BAMYAN": [34.8000, 67.8000],
            "BADGHIS": [35.3333, 63.4167],
            "FARAH": [32.3667, 62.1000],
            "FARYAB": [37.0000, 64.7500],
            "GHOR": [33.7500, 64.5000],
            "HILMAND": [31.7500, 64.3667],
            "NIMRUZ": [31.0000, 61.5000],
            "URUZGAN": [32.9333, 66.9333],
            "ZABUL": [32.9333, 67.2667],
            "KAPISA": [35.0000, 69.4167],
            "PARWAN": [35.2000, 69.5000],
            "PANSHIR": [35.3000, 69.5000],
        }
        
        # Create base map centered on Pakistan-Afghanistan region
        m = folium.Map(
            location=[33.0, 67.0], 
            zoom_start=6, 
            tiles=map_tile,
            max_bounds=True,
            min_zoom=5,
            max_zoom=12
        )
        
        # Set map bounds to show only Pakistan & Afghanistan
        southwest = [23.5, 60.0]  # Southwest corner
        northeast = [37.5, 77.0]  # Northeast corner
        m.fit_bounds([southwest, northeast])
        
        # Prevent panning outside bounds
        m.options['maxBounds'] = [[23.5, 60.0], [37.5, 77.0]]
        
        # Add destination districts as larger circles
        for dest in selected_dest:
            if dest in DISTRICT_COORDS:
                coords = DISTRICT_COORDS[dest]
                total_incoming = filtered_df[filtered_df[DEST_COL] == dest][weight_col].sum()
                folium.CircleMarker(
                    location=coords,
                    radius=min(25, max(10, total_incoming / 500)),
                    popup=f"<b>{dest}</b><br>Total {weight_type}: {int(total_incoming)}",
                    color="red",
                    fill=True,
                    fillColor="red",
                    fillOpacity=0.7,
                    weight=2
                ).add_to(m)
        
        # Add origin districts as smaller circles
        if len(selected_origin) > 0:
            for origin in selected_origin:
                if origin in DISTRICT_COORDS:
                    coords = DISTRICT_COORDS[origin]
                    total_outgoing = filtered_df[filtered_df[ORIGIN_COL] == origin][weight_col].sum()
                    folium.CircleMarker(
                        location=coords,
                        radius=min(20, max(5, total_outgoing / 500)),
                        popup=f"<b>{origin}</b><br>Total {weight_type}: {int(total_outgoing)}",
                        color="blue",
                        fill=True,
                        fillColor="blue",
                        fillOpacity=0.5,
                        weight=1
                    ).add_to(m)
        
        # Add flow lines (top corridors)
        for _, row in edges_top.head(20).iterrows():
            origin = row[ORIGIN_COL]
            dest = row[DEST_COL]
            weight = row["weight"]
            
            if origin in DISTRICT_COORDS and dest in DISTRICT_COORDS:
                coords_origin = DISTRICT_COORDS[origin]
                coords_dest = DISTRICT_COORDS[dest]
                
                # Line width proportional to flow
                line_weight = max(1, min(5, weight / 1000))
                
                folium.PolyLine(
                    locations=[coords_origin, coords_dest],
                    color="green",
                    weight=line_weight,
                    opacity=0.6,
                    popup=f"{origin} → {dest}: {int(weight)} {weight_type}"
                ).add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 200px; height: 140px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p style="margin: 0;"><b>Legend</b></p>
        <p style="margin: 5px 0;"><i class="fa fa-circle" style="color:red"></i> Destination Districts (Sindh)</p>
        <p style="margin: 5px 0;"><i class="fa fa-circle" style="color:blue"></i> Origin Districts</p>
        <p style="margin: 5px 0;"><i class="fa fa-minus" style="color:green"></i> Migration Flow</p>
        <p style="margin: 5px 0; font-size: 12px;">Circle size = Total {}</p>
        </div>
        '''.format(weight_type)
        m.get_root().html.add_child(folium.Element(legend_html))
        
        st_folium(m, width=1200, height=700)

st.divider()
st.caption("Sindh MMP Analysis Dashboard | Data source: Sindh MMP Analysis.xlsx")
