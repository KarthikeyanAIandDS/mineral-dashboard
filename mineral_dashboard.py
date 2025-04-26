import streamlit as st
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import random
import os

# Set page configuration must be the FIRST Streamlit command
st.set_page_config(
    page_title="Mineral Targeting Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# File path helper function
def get_file_path(relative_path):
    """Get absolute file path from relative path."""
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Join with the relative path
    return os.path.join(script_dir, relative_path)

# Custom CSS with black text for the info-box
st.markdown("""
    <style>
        .block-container {padding: 0.5rem;}
        .main > div {padding: 0.5rem;}
        .info-box {
            background-color: #f0f7ff;
            padding: 10px;
            border-radius: 5px;
            border-left: 3px solid #3b82f6;
            margin-bottom: 10px;
            color: black;
        }
        .warning-box {
            background-color: #fff7ed;
            padding: 10px;
            border-radius: 5px;
            border-left: 3px solid #f97316;
            margin-bottom: 10px;
            color: black;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Mineral Targeting Dashboard")

# Load data function with improved file path handling
def load_data():
    try:
        # Use helper function to get correct file paths
        geomorph_path = get_file_path("datasets/multi_layer_geological_map_of_karnataka_and_andhra_pradesh_25k_scale_v1/25K/lithology_25k_ngdr_20250224140917945/lithology_25k_ngdr.shp")
        mineral_path = get_file_path("datasets/multi_layer_geological_map_of_karnataka_and_andhra_pradesh_25k_scale_v1/25K/mineralization_25k_ngdr_20250224141143411/mineralization_25k_ngdr_20250224141143411.shp")
        
        # Debugging information
        with st.expander("Debug Information"):
            st.write(f"Loading geomorphology from: {geomorph_path}")
            st.write(f"Loading minerals from: {mineral_path}")
            st.write(f"File exists (geomorph): {os.path.exists(geomorph_path)}")
            st.write(f"File exists (mineral): {os.path.exists(mineral_path)}")
        
        # Read the files
        geomorph_gdf = gpd.read_file(geomorph_path)
        mineral_gdf = gpd.read_file(mineral_path)
        
        # Make sure both are in WGS84
        if geomorph_gdf.crs != "EPSG:4326":
            geomorph_gdf = geomorph_gdf.to_crs("EPSG:4326")
        if mineral_gdf.crs != "EPSG:4326":
            mineral_gdf = mineral_gdf.to_crs("EPSG:4326")
        
        # Make a projected copy for calculations
        geomorph_utm = geomorph_gdf.copy().to_crs(epsg=32643)
        mineral_utm = mineral_gdf.copy().to_crs(epsg=32643)
        
        # Calculate area (km²)
        geomorph_gdf['area_sqkm'] = geomorph_utm.geometry.area / 1e6
        
        # Extract mineral centroids
        mineral_gdf['centroid_lat'] = mineral_gdf.geometry.centroid.y
        mineral_gdf['centroid_lon'] = mineral_gdf.geometry.centroid.x
        
        # Spatial join to calculate mineral counts
        joined = gpd.sjoin(geomorph_utm, mineral_utm, how="left", predicate="intersects")
        mineral_counts = joined.groupby(joined.index)['commodity'].count()
        geomorph_gdf['mineral_count'] = mineral_counts
        geomorph_gdf['mineral_count'] = geomorph_gdf['mineral_count'].fillna(0).astype(int)
        
        # Calculate mineral density
        geomorph_gdf['mineral_density'] = geomorph_gdf['mineral_count'] / geomorph_gdf['area_sqkm']
        geomorph_gdf['mineral_density'] = geomorph_gdf['mineral_density'].fillna(0)
        
        # Normalize to 0-100% for accuracy
        max_density = geomorph_gdf['mineral_density'].max()
        if max_density > 0:
            geomorph_gdf['accuracy'] = (geomorph_gdf['mineral_density'] / max_density) * 100
        else:
            geomorph_gdf['accuracy'] = 0
        
        # Calculate counts for each mineral type
        for commodity in mineral_gdf['commodity'].unique():
            if pd.notnull(commodity):
                col_name = f"count_{commodity.replace(' ', '_').replace(',', '_').replace('-', '_')}"
                # Filter for this commodity
                this_mineral = mineral_utm[mineral_utm['commodity'] == commodity]
                if not this_mineral.empty:
                    # Join with geomorphology
                    joined_specific = gpd.sjoin(geomorph_utm, this_mineral, how="left", predicate="intersects")
                    # Count occurrences
                    counts = joined_specific.groupby(joined_specific.index)['commodity'].count()
                    # Create column name based on commodity
                    geomorph_gdf[col_name] = counts
                    geomorph_gdf[col_name] = geomorph_gdf[col_name].fillna(0).astype(int)
        
        return geomorph_gdf, mineral_gdf
    except Exception as e:
        st.error(f"Error loading data: {e}")
        if "fiona" in str(e) and "path" in str(e):
            st.error("Fiona path error: Make sure you have all required packages installed.")
            st.code("pip install geopandas fiona pyproj shapely rtree")
        return None, None

# Load the data only once - use session state
if 'data_loaded' not in st.session_state:
    with st.spinner('Loading geospatial data...'):
        geomorph_gdf, mineral_gdf = load_data()
        st.session_state.data_loaded = True
        st.session_state.geomorph_gdf = geomorph_gdf
        st.session_state.mineral_gdf = mineral_gdf
else:
    geomorph_gdf = st.session_state.geomorph_gdf
    mineral_gdf = st.session_state.mineral_gdf

if geomorph_gdf is None or mineral_gdf is None:
    st.warning("Please provide the correct path to the shapefile data.")
    st.stop()

# Generate bright, distinguishable colors for geomorphology types
def generate_bright_colors(n):
    colors = {}
    # Start with some predefined vibrant colors
    base_colors = [
        '#FF5733', '#C70039', '#900C3F', '#581845',  # Red to purple
        '#1E88E5', '#039BE5', '#00ACC1', '#00897B',  # Blue to teal
        '#43A047', '#7CB342', '#C0CA33', '#FDD835',  # Green to yellow
        '#FFB300', '#FB8C00', '#F4511E', '#6D4C41',  # Orange to brown
        '#8E24AA', '#5E35B1', '#3949AB', '#1E88E5',  # Purple to blue
        '#00ACC1', '#00897B', '#00ACC1', '#D81B60',  # Teal to magenta
    ]
    
    # Repeat and mix if we need more colors than available
    for i in range(n):
        color = base_colors[i % len(base_colors)]
        colors[i] = color
        
    return colors

# Define vibrant colors for minerals and regions
commodity_colors = {
    'Gold': '#FFD700',
    'Diamond': '#B9F2FF',
    'Iron': '#FF5733',
    'Kimberlite': '#32CD32',
    'Copper': '#FFA500',
    'Base Metal': '#8B4513',
    'Polymetallic': '#9370DB',
    'Granite': '#A9A9A9',
    'Manganese': '#006400',
    'Tungsten': '#FF69B4',
    'Lead,Antimony,Sulphides': '#8A2BE2',
    'Rare Earth Elements': '#000080',
    'Tin-Tungsten': '#D3D3D3',
    'Platinum Group Elements': '#FFFF00',  # Bright yellow as in your image
    'Kimberlite,Diamond': '#00FF00'
}

# Generate random colors for geomorphology types
geomorph_types = sorted([g for g in geomorph_gdf['legend_sho'].dropna().unique() 
                         if isinstance(g, str)])
bright_colors = generate_bright_colors(len(geomorph_types))
geomorph_colors = {geomorph_types[i]: bright_colors[i] for i in range(len(geomorph_types))}

# Get unique mineral types
commodities = sorted([c for c in mineral_gdf['commodity'].dropna().unique() 
                      if isinstance(c, str)])

# Control panel
st.subheader("Select Region and Mineral")
col1, col2 = st.columns(2)

with col1:
    selected_geomorph = st.selectbox(
        "Select Geomorphology Type",
        ["Select a region..."] + geomorph_types
    )

with col2:
    selected_mineral = st.selectbox(
        "Select Mineral Type",
        ["Select a mineral..."] + commodities
    )

# Check if both selections are made
both_selected = (selected_geomorph != "Select a region..." and selected_mineral != "Select a mineral...")
only_geomorph = (selected_geomorph != "Select a region..." and selected_mineral == "Select a mineral...")
only_mineral = (selected_geomorph == "Select a region..." and selected_mineral != "Select a mineral...")
nothing_selected = (selected_geomorph == "Select a region..." and selected_mineral == "Select a mineral...")

# Show appropriate warnings/info
if only_geomorph:
    st.markdown('<div class="warning-box">Please select a mineral type to see its presence in this region.</div>', unsafe_allow_html=True)
elif only_mineral:
    st.markdown('<div class="warning-box">Please select a region to see the presence of this mineral.</div>', unsafe_allow_html=True)
elif nothing_selected:
    st.markdown('<div class="info-box">Please select both a region and a mineral to analyze their relationship.</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="info-box">Showing presence of <b>{selected_mineral}</b> in <b>{selected_geomorph}</b></div>', unsafe_allow_html=True)

# Filter data based on selection
if both_selected:
    # Get relevant data
    geom_plot = geomorph_gdf[geomorph_gdf['legend_sho'] == selected_geomorph]
    min_plot = mineral_gdf[mineral_gdf['commodity'] == selected_mineral]
    
    # Get count of selected mineral in selected region
    col_name = f"count_{selected_mineral.replace(' ', '_').replace(',', '_').replace('-', '_')}"
    if col_name in geom_plot.columns:
        mineral_count = geom_plot[col_name].sum()
        total_minerals = geom_plot['mineral_count'].sum()
        if total_minerals > 0:
            mineral_percentage = (mineral_count / total_minerals) * 100
        else:
            mineral_percentage = 0
    else:
        mineral_count = 0
        mineral_percentage = 0
    
    # Show metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Area", f"{geom_plot['area_sqkm'].sum():.1f} km²")
    with col2:
        st.metric(f"{selected_mineral} Count", f"{int(mineral_count)}")
    with col3:
        st.metric("Presence", f"{mineral_percentage:.1f}%")
elif only_geomorph:
    geom_plot = geomorph_gdf[geomorph_gdf['legend_sho'] == selected_geomorph]
    min_plot = pd.DataFrame()  # Empty DataFrame
elif only_mineral:
    geom_plot = pd.DataFrame()  # Empty DataFrame
    min_plot = mineral_gdf[mineral_gdf['commodity'] == selected_mineral]
else:
    geom_plot = pd.DataFrame()  # Empty DataFrame
    min_plot = pd.DataFrame()  # Empty DataFrame

# Maps section
map_col, plot_col = st.columns([3, 2])

with map_col:
    st.subheader("Region and Mineral Distribution")
    
    # Create a map using matplotlib instead of folium to match your example
    if nothing_selected:
        # Create a colorful map of all regions similar to the example
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot all geomorphology regions with random colors
        geomorph_gdf.plot(
            column='legend_sho',
            ax=ax,
            categorical=True,
            cmap='viridis',
            edgecolor='black',
            linewidth=0.5,
            legend=False
        )
        
        # Create legend for a sample of regions to avoid overcrowding
        # Use a selection of the most prevalent regions
        top_regions = geomorph_gdf['legend_sho'].value_counts().head(10).index
        handles = []
        for region in top_regions:
            if isinstance(region, str):
                color = geomorph_colors.get(region, '#CCCCCC')
                patch = mpatches.Patch(color=color, label=region)
                handles.append(patch)
        
        # Place legend to the right of the plot
        ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
        ax.set_title('Geomorphology Regions')
        
        # Remove axis ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        
        st.pyplot(fig)
