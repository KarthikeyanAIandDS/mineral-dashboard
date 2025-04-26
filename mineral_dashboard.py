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
from matplotlib.cm import ScalarMappable
from shapely.geometry import MultiPolygon, Polygon
import os

# Set page configuration must be the FIRST Streamlit command
st.set_page_config(
    page_title="Mineral Targeting Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS with more colorful styling
st.markdown("""
    <style>
        /* Main dashboard styling */
        .block-container {padding: 0.5rem;}
        .main > div {padding: 0.5rem;}
        
        /* Gradient header */
        .title-gradient {
            background: linear-gradient(90deg, #FF9A8B 0%, #FF6A88 55%, #FF99AC 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 20px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        
        /* Colorful info box */
        .info-box {
            background: linear-gradient(120deg, #84fab0 0%, #8fd3f4 100%);
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
            color: black;
            font-weight: bold;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* Warning box with warm gradient */
        .warning-box {
            background: linear-gradient(120deg, #f6d365 0%, #fda085 100%);
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
            color: black;
            font-weight: bold;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* Section headers */
        .section-header {
            background: linear-gradient(90deg, #8EC5FC 0%, #E0C3FC 100%);
            padding: 10px 15px;
            border-radius: 8px;
            color: black;
            font-weight: bold;
            margin-top: 10px;
            margin-bottom: 15px;
            text-align: center;
        }
        
        /* Metric cards */
        .metric-container {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        
        .metric-card {
            background: linear-gradient(135deg, #43cea2 0%, #185a9d 100%);
            padding: 15px;
            border-radius: 10px;
            flex: 1;
            color: white;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .metric-card.blue {
            background: linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%);
        }
        
        .metric-card.purple {
            background: linear-gradient(135deg, #834d9b 0%, #d04ed6 100%);
        }
        
        .metric-card h3 {
            margin: 0;
            font-size: 24px;
            font-weight: bold;
        }
        
        .metric-card p {
            margin: 5px 0 0 0;
            font-size: 14px;
            opacity: 0.9;
        }
        
        /* Footer styling */
        .footer {
            background: linear-gradient(90deg, #a1c4fd 0%, #c2e9fb 100%);
            padding: 10px;
            border-radius: 8px;
            margin-top: 20px;
            font-size: 12px;
            color: #333;
            text-align: center;
            box-shadow: 0 -2px 4px rgba(0,0,0,0.05);
        }
    </style>
""", unsafe_allow_html=True)

# Gradient title instead of plain title
st.markdown('<div class="title-gradient"><h1>🌍 Mineral Targeting Dashboard</h1></div>', unsafe_allow_html=True)

# Function to plot polygons properly (handling both Polygon and MultiPolygon)
def plot_polygon(ax, geometry, color, alpha=0.7, edgecolor='black', linewidth=1):
    """Safely plot a polygon, handling both Polygon and MultiPolygon geometries"""
    if geometry.is_empty:
        return
        
    if isinstance(geometry, MultiPolygon):
        for poly in geometry.geoms:
            if not poly.is_empty and hasattr(poly, 'exterior'):
                x, y = poly.exterior.xy
                ax.fill(x, y, color=color, alpha=alpha, edgecolor=edgecolor, linewidth=linewidth)
                
                # Also plot any interior holes
                for interior in poly.interiors:
                    x, y = interior.xy
                    ax.fill(x, y, color='white', alpha=1, edgecolor=edgecolor, linewidth=linewidth)
    elif isinstance(geometry, Polygon) and hasattr(geometry, 'exterior'):
        x, y = geometry.exterior.xy
        ax.fill(x, y, color=color, alpha=alpha, edgecolor=edgecolor, linewidth=linewidth)
        
        # Also plot any interior holes
        for interior in geometry.interiors:
            x, y = interior.xy
            ax.fill(x, y, color='white', alpha=1, edgecolor=edgecolor, linewidth=linewidth)

# File path helper function
def get_file_path(relative_path):
    """Get absolute file path from relative path."""
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Join with the relative path
    return os.path.join(script_dir, relative_path)

# Load data with better path handling
def load_data():
    try:
        # Use helper function to get correct file paths
        geomorph_path = get_file_path("datasets/multi_layer_geological_map_of_karnataka_and_andhra_pradesh_25k_scale_v1/25K/lithology_25k_ngdr_20250224140917945/lithology_25k_ngdr.shp")
        mineral_path = get_file_path("datasets/multi_layer_geological_map_of_karnataka_and_andhra_pradesh_25k_scale_v1/25K/mineralization_25k_ngdr_20250224141143411/mineralization_25k_ngdr_20250224141143411.shp")
        
        # Show debugging information about data loading
        with st.expander("Debug Information"):
            st.write(f"Loading geomorphology from: {geomorph_path}")
            st.write(f"Loading minerals from: {mineral_path}")
            st.write(f"File exists (geomorph): {os.path.exists(geomorph_path)}")
            st.write(f"File exists (mineral): {os.path.exists(mineral_path)}")
        
        # Read the files
        geomorph_gdf = gpd.read_file(geomorph_path)
        mineral_gdf = gpd.read_file(mineral_path)
        
        # Display data columns for debugging
        with st.expander("Data Columns"):
            st.write("Geomorphology columns:", geomorph_gdf.columns.tolist())
            st.write("Mineral columns:", mineral_gdf.columns.tolist())
        
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
        
        # Find suitable region and mineral column names based on available data
        if 'legend_sho' in geomorph_gdf.columns:
            region_column = 'legend_sho'
        elif 'lith_unit' in geomorph_gdf.columns:
            region_column = 'lith_unit'
        elif 'name' in geomorph_gdf.columns:
            region_column = 'name'
        elif 'lithology' in geomorph_gdf.columns:
            region_column = 'lithology'
        else:
            # Use the first string column as fallback
            for col in geomorph_gdf.columns:
                if geomorph_gdf[col].dtype == 'object':
                    region_column = col
                    break
            else:
                region_column = geomorph_gdf.columns[0]
                
        if 'commodity' in mineral_gdf.columns:
            mineral_column = 'commodity'
        elif 'mineral' in mineral_gdf.columns:
            mineral_column = 'mineral'
        elif 'name' in mineral_gdf.columns:
            mineral_column = 'name'
        else:
            # Use the first string column as fallback
            for col in mineral_gdf.columns:
                if mineral_gdf[col].dtype == 'object':
                    mineral_column = col
                    break
            else:
                mineral_column = mineral_gdf.columns[0]
        
        # Store the column names in session state
        st.session_state.region_column = region_column
        st.session_state.mineral_column = mineral_column
        
        # Spatial join to calculate mineral counts
        joined = gpd.sjoin(geomorph_utm, mineral_utm, how="left", predicate="intersects")
        mineral_counts = joined.groupby(joined.index)[mineral_column].count()
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
        for commodity in mineral_gdf[mineral_column].unique():
            if pd.notnull(commodity):
                col_name = f"count_{str(commodity).replace(' ', '_').replace(',', '_').replace('-', '_')}"
                # Filter for this commodity
                this_mineral = mineral_utm[mineral_utm[mineral_column] == commodity]
                if not this_mineral.empty:
                    # Join with geomorphology
                    joined_specific = gpd.sjoin(geomorph_utm, this_mineral, how="left", predicate="intersects")
                    # Count occurrences
                    counts = joined_specific.groupby(joined_specific.index)[mineral_column].count()
                    # Create column name based on commodity
                    geomorph_gdf[col_name] = counts
                    geomorph_gdf[col_name] = geomorph_gdf[col_name].fillna(0).astype(int)
        
        return geomorph_gdf, mineral_gdf
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# Load the data only once - use session state
if 'data_loaded' not in st.session_state:
    with st.spinner('Loading geospatial data...'):
        geomorph_gdf, mineral_gdf = load_data()
        if geomorph_gdf is not None and mineral_gdf is not None:
            st.session_state.data_loaded = True
            st.session_state.geomorph_gdf = geomorph_gdf
            st.session_state.mineral_gdf = mineral_gdf
else:
    geomorph_gdf = st.session_state.geomorph_gdf
    mineral_gdf = st.session_state.mineral_gdf

if geomorph_gdf is None or mineral_gdf is None:
    st.warning("Please provide the correct path to the shapefile data.")
    st.stop()

# Get region and mineral columns from session state
region_column = st.session_state.get('region_column', 'legend_sho')
mineral_column = st.session_state.get('mineral_column', 'commodity')

# Generate vibrant, distinct colors
def generate_bright_colors(n, saturation=0.9, value=0.95):
    colors = []
    golden_ratio_conjugate = 0.618033988749895
    h = random.random()  # Starting hue
    
    for i in range(n):
        h += golden_ratio_conjugate
        h %= 1
        r, g, b = [x for x in mcolors.hsv_to_rgb([h, saturation, value])]
        colors.append(f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}')
    
    return colors

# Define vibrant colors for minerals and regions
commodity_colors = {
    'Gold': '#FFD700',
    'Diamond': '#00FFFF',  # More vibrant cyan
    'Iron': '#FF2D00',     # Brighter red
    'Kimberlite': '#32CD32',
    'Copper': '#FF7F00',   # Brighter orange
    'Base Metal': '#A52A2A',
    'Polymetallic': '#8A2BE2', # Brighter purple
    'Granite': '#A9A9A9',
    'Manganese': '#008000', # Brighter green
    'Tungsten': '#FF1493',  # Deep pink
    'Lead,Antimony,Sulphides': '#9400D3', # Brighter violet
    'Rare Earth Elements': '#0000CD', # Medium blue
    'Tin-Tungsten': '#D3D3D3',
    'Platinum Group Elements': '#FFFF00', # Bright yellow
    'Kimberlite,Diamond': '#7CFC00'  # Brighter lawn green
}

# Generate random colors for geomorphology types
geomorph_types = sorted([g for g in geomorph_gdf[region_column].dropna().unique() 
                         if isinstance(g, str)])
                         
# Generate a colorful palette for geomorphology types
bright_colors = generate_bright_colors(len(geomorph_types))
geomorph_colors = {geomorph_types[i]: bright_colors[i] for i in range(len(geomorph_types))}

# Get unique mineral types
commodities = sorted([c for c in mineral_gdf[mineral_column].dropna().unique() 
                      if isinstance(c, str)])

# Control panel with styled headers
st.markdown('<div class="section-header">🔍 Select Region and Mineral</div>', unsafe_allow_html=True)
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

# Show appropriate warnings/info with styled boxes
if only_geomorph:
    st.markdown('<div class="warning-box">⚠️ Please select a mineral type to see its presence in this region.</div>', unsafe_allow_html=True)
elif only_mineral:
    st.markdown('<div class="warning-box">⚠️ Please select a region to see the presence of this mineral.</div>', unsafe_allow_html=True)
elif nothing_selected:
    st.markdown('<div class="info-box">ℹ️ Please select both a region and a mineral to analyze their relationship.</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="info-box">🔍 Showing presence of <b>{selected_mineral}</b> in <b>{selected_geomorph}</b></div>', unsafe_allow_html=True)

# Filter data based on selection
if both_selected:
    # Get relevant data
    geom_plot = geomorph_gdf[geomorph_gdf[region_column] == selected_geomorph]
    min_plot = mineral_gdf[mineral_gdf[mineral_column] == selected_mineral]
    
    # Get count of selected mineral in selected region
    col_name = f"count_{str(selected_mineral).replace(' ', '_').replace(',', '_').replace('-', '_')}"
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
    
    # Show metrics with styled cards
    st.markdown('''
    <div class="metric-container">
        <div class="metric-card">
            <p>Area</p>
            <h3>''' + f"{geom_plot['area_sqkm'].sum():.1f} km²" + '''</h3>
        </div>
        <div class="metric-card blue">
            <p>''' + f"{selected_mineral} Count" + '''</p>
            <h3>''' + f"{int(mineral_count)}" + '''</h3>
        </div>
        <div class="metric-card purple">
            <p>Presence</p>
            <h3>''' + f"{mineral_percentage:.1f}%" + '''</h3>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
elif only_geomorph:
    geom_plot = geomorph_gdf[geomorph_gdf[region_column] == selected_geomorph]
    min_plot = pd.DataFrame()  # Empty DataFrame
elif only_mineral:
    geom_plot = pd.DataFrame()  # Empty DataFrame
    min_plot = mineral_gdf[mineral_gdf[mineral_column] == selected_mineral]
else:
    geom_plot = pd.DataFrame()  # Empty DataFrame
    min_plot = pd.DataFrame()  # Empty DataFrame

# Maps section with styled header
map_col, plot_col = st.columns([3, 2])

with map_col:
    st.markdown('<div class="section-header">🗺️ Region and Mineral Distribution</div>', unsafe_allow_html=True)
    
    # Create a colorful map
    if nothing_selected:
        # Create a colorful map of all regions
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='#f0f2f6')
        
        # Set a colorful background
        ax.set_facecolor('#f0f2f6')
        
        # Create a custom colormap for regions
        cmap = plt.cm.get_cmap('viridis', len(geomorph_types))
        
        # Dictionary to map region types to colormap indices
        region_to_idx = {region: i for i, region in enumerate(geomorph_types)}
        
        # Add colors to each region
        for idx, row in geomorph_gdf.iterrows():
            region_type = row[region_column]
            if isinstance(region_type, str):
                color_idx = region_to_idx.get(region_type, 0)
                color = cmap(color_idx)
                
                # Plot with a black edge
                plot_polygon(ax, row.geometry, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Create a legend with a selection of top regions
        top_regions = geomorph_gdf[region_column].value_counts().head(10).index
        handles = []
        for i, region in enumerate(top_regions):
            if isinstance(region, str):
                color_idx = region_to_idx.get(region, 0)
                patch = mpatches.Patch(color=cmap(color_idx), label=region)
                handles.append(patch)
        
        # Add a fancy legend
        ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8, 
                  facecolor='white', edgecolor='gray', framealpha=0.8, shadow=True)
        
        # Remove axis ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add a title with fancy font
        ax.set_title('Geomorphology Region Map', fontsize=16, fontweight='bold', pad=20)
        
        # Add a grid
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Add a border around the plot
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(2)
        
        # Show the plot
        st.pyplot(fig)
    
    elif both_selected:
        # Create a colorful map showing the selected region and mineral distribution
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='#f0f2f6')
        
        # Set a nice background color
        ax.set_facecolor('#f0f2f6')
        
        # First plot all regions with a light color
        for idx, row in geomorph_gdf.iterrows():
            plot_polygon(ax, row.geometry, color='#f5f5f5', alpha=0.4, edgecolor='#aaaaaa', linewidth=0.5)
        
        # Then highlight the selected region with the mineral's color
        col_name = f"count_{str(selected_mineral).replace(' ', '_').replace(',', '_').replace('-', '_')}"
        has_mineral = False
        
        if col_name in geom_plot.columns and geom_plot[col_name].sum() > 0:
            has_mineral = True
            color = commodity_colors.get(selected_mineral, '#3388ff')
        else:
            color = '#ff3333'  # Red for no mineral
        
        for idx, row in geom_plot.iterrows():
            plot_polygon(ax, row.geometry, color=color, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add mineral markers with a nice glow effect
        if has_mineral:
            for idx, row in min_plot.iterrows():
                try:
                    # Get the mineral color
                    marker_color = commodity_colors.get(selected_mineral, '#3388ff')
                    
                    # Plot the marker with a glow effect
                    ax.plot(row.geometry.centroid.x, row.geometry.centroid.y, 
                            marker='o', markersize=12, color='white', alpha=0.5)  # Glow
                    ax.plot(row.geometry.centroid.x, row.geometry.centroid.y, 
                            marker='o', color=marker_color, markersize=8, alpha=0.9,
                            markeredgecolor='white', markeredgewidth=1)
                except:
                    continue
        
        # Add a fancy legend
        legend_elements = []
        
        if has_mineral:
            legend_elements.append(mpatches.Patch(
                facecolor=color, edgecolor='black', linewidth=1,
                label=f'{selected_geomorph} with {selected_mineral}'))
            
            legend_elements.append(plt.Line2D(
                [0], [0], marker='o', color='w', markerfacecolor=commodity_colors.get(selected_mineral, '#3388ff'),
                markersize=10, markeredgecolor='white', markeredgewidth=1,
                label=f'{selected_mineral} occurrence'))
        else:
            legend_elements.append(mpatches.Patch(
                facecolor=color, edgecolor='black', linewidth=1,
                label=f'{selected_geomorph} (no {selected_mineral})'))
        
        ax.legend(handles=legend_elements, loc='upper right', frameon=True, 
                  facecolor='white', edgecolor='black', framealpha=0.8, shadow=True)
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add a title with fancy font
        ax.set_title(f'{selected_mineral} in {selected_geomorph}', fontsize=16, fontweight='bold', pad=20)
        
        # Add a border around the plot
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(2)
        
        # Show the plot
        st.pyplot(fig)
