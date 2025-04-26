import streamlit as st
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import random
from matplotlib.cm import ScalarMappable
from shapely.geometry import MultiPolygon, Polygon

# Set page configuration
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

# Load data
def load_data():
    # Replace with your actual file paths
    geomorph_path = "geomorphology_250k_gcs_ngdr.shp"
    mineral_path = "exploration_data_gis_view.shp"
    
    try:
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
geomorph_types = sorted([g for g in geomorph_gdf['legend_sho'].dropna().unique() 
                         if isinstance(g, str)])
                         
# Generate a colorful palette for geomorphology types
bright_colors = generate_bright_colors(len(geomorph_types))
geomorph_colors = {geomorph_types[i]: bright_colors[i] for i in range(len(geomorph_types))}

# Get unique mineral types
commodities = sorted([c for c in mineral_gdf['commodity'].dropna().unique() 
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
    geom_plot = geomorph_gdf[geomorph_gdf['legend_sho'] == selected_geomorph]
    min_plot = pd.DataFrame()  # Empty DataFrame
elif only_mineral:
    geom_plot = pd.DataFrame()  # Empty DataFrame
    min_plot = mineral_gdf[mineral_gdf['commodity'] == selected_mineral]
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
            region_type = row['legend_sho']
            if isinstance(region_type, str):
                color_idx = region_to_idx.get(region_type, 0)
                color = cmap(color_idx)
                
                # Plot with a black edge
                plot_polygon(ax, row.geometry, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Create a legend with a selection of top regions
        top_regions = geomorph_gdf['legend_sho'].value_counts().head(10).index
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
        col_name = f"count_{selected_mineral.replace(' ', '_').replace(',', '_').replace('-', '_')}"
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
    
    elif only_geomorph:
        # Create a colorful map highlighting the selected region
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='#f0f2f6')
        
        # Set a nice background
        ax.set_facecolor('#f0f2f6')
        
        # First plot all regions with a light color
        for idx, row in geomorph_gdf.iterrows():
            if row['legend_sho'] != selected_geomorph:
                plot_polygon(ax, row.geometry, color='#f5f5f5', alpha=0.4, edgecolor='#aaaaaa', linewidth=0.5)
        
        # Then highlight the selected region with a vibrant color
        region_color = geomorph_colors.get(selected_geomorph, '#3388ff')
        
        for idx, row in geom_plot.iterrows():
            # Create a gradient effect
            plot_polygon(ax, row.geometry, color=region_color, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add a fancy legend
        legend_elements = [
            mpatches.Patch(facecolor=region_color, edgecolor='black', linewidth=1, label=selected_geomorph),
            mpatches.Patch(facecolor='#f5f5f5', edgecolor='#aaaaaa', linewidth=0.5, label='Other regions')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', frameon=True,
                  facecolor='white', edgecolor='black', framealpha=0.8, shadow=True)
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add a title with fancy font
        ax.set_title(f'Region: {selected_geomorph}', fontsize=16, fontweight='bold', pad=20)
        
        # Add a border around the plot
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(2)
        
        # Show the plot
        st.pyplot(fig)
    
    elif only_mineral:
        # Create a colorful map showing regions with the selected mineral
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='#f0f2f6')
        
        # Set a nice background
        ax.set_facecolor('#f0f2f6')
        
        # First plot all regions with a light color
        for idx, row in geomorph_gdf.iterrows():
            plot_polygon(ax, row.geometry, color='#f5f5f5', alpha=0.4, edgecolor='#aaaaaa', linewidth=0.5)
        
        # Then highlight regions with this mineral
        col_name = f"count_{selected_mineral.replace(' ', '_').replace(',', '_').replace('-', '_')}"
        
        if col_name in geomorph_gdf.columns:
            regions_with_mineral = geomorph_gdf[geomorph_gdf[col_name] > 0]
            
            if not regions_with_mineral.empty:
                # Get the mineral color
                mineral_color = commodity_colors.get(selected_mineral, '#3388ff')
                
                # Create a color gradient based on the count
                max_count = regions_with_mineral[col_name].max()
                min_count = regions_with_mineral[col_name].min()
                
                # Plot each region with color intensity based on count
                for idx, row in regions_with_mineral.iterrows():
                    count = row[col_name]
                    # If max_count equals min_count, then intensity is 1, otherwise scale it
                    if max_count > min_count:
                        intensity = 0.4 + 0.6 * (count - min_count) / (max_count - min_count)
                    else:
                        intensity = 0.8
                    
                    plot_polygon(ax, row.geometry, color=mineral_color, alpha=intensity, edgecolor='black', linewidth=0.5)
                
                # Add a colorbar
                if max_count > min_count:
                    sm = ScalarMappable(
                        norm=mcolors.Normalize(vmin=min_count, vmax=max_count),
                        cmap=mcolors.LinearSegmentedColormap.from_list('mineral_cmap', ['#ffffff', mineral_color])
                    )
                    sm.set_array([])
                    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', shrink=0.8)
                    cbar.set_label(f'{selected_mineral} Occurrence Count')
        
        # Add mineral locations with a nice effect
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
        legend_elements = [
            mpatches.Patch(facecolor=commodity_colors.get(selected_mineral, '#3388ff'), 
                          edgecolor='black', linewidth=1, label=f'Regions with {selected_mineral}'),
            mpatches.Patch(facecolor='#f5f5f5', edgecolor='#aaaaaa', 
                          linewidth=0.5, label='Regions without mineral'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=commodity_colors.get(selected_mineral, '#3388ff'),
                      markersize=10, markeredgecolor='white', markeredgewidth=1, label=f'{selected_mineral} occurrence')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', frameon=True,
                  facecolor='white', edgecolor='black', framealpha=0.8, shadow=True)
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add a title with fancy font
        ax.set_title(f'Distribution of {selected_mineral}', fontsize=16, fontweight='bold', pad=20)
        
        # Add a border around the plot
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(2)
        
        # Show the plot
        st.pyplot(fig)

with plot_col:
    # Show charts and analyses based on selection
    if both_selected:
        # Show analysis for the specific mineral in the region
        st.markdown('<div class="section-header">📊 Analysis: ' + f"{selected_mineral} in {selected_geomorph}</div>", unsafe_allow_html=True)
        
        # Check if the mineral is present in the region
        if mineral_count > 0:
            # Create colorful pie chart showing this mineral vs others
            fig, ax = plt.subplots(figsize=(6, 6), facecolor='#f0f2f6')
            ax.set_facecolor('#f0f2f6')
            
            other_minerals = total_minerals - mineral_count
            labels = [selected_mineral, 'Other Minerals']
            sizes = [mineral_count, other_minerals]
            colors = [commodity_colors.get(selected_mineral, '#3388ff'), '#CCCCCC']
            explode = (0.1, 0)  # explode the 1st slice (selected mineral)
            
            wedges, texts, autotexts = ax.pie(
                sizes, 
                explode=explode, 
                labels=labels, 
                colors=colors,
                autopct='%1.1f%%', 
                startangle=90, 
                shadow=True,
                wedgeprops={'edgecolor': 'black', 'linewidth': 1},
                textprops={'fontsize': 12, 'fontweight': 'bold'}
            )
            
            # Make the text more readable
            for text in texts:
                text.set_color('#333333')
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
            ax.set_title(f'Proportion of {selected_mineral} in {selected_geomorph}', 
                        fontsize=14, fontweight='bold', pad=20)
            
            # Add a border around the plot
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(2)
            
            st.pyplot(fig)
            
            # Show key findings with colorful icons
            st.markdown("""
            <h3 style="background: linear-gradient(90deg, #ffd3b6 0%, #ffaaa5 100%); padding: 10px; border-radius: 5px;">
                🔑 Key Findings
            </h3>
            """, unsafe_allow_html=True)
            
            st.markdown(f"- 💎 {selected_mineral} makes up **{mineral_percentage:.1f}%** of all minerals in this region")
            st.markdown(f"- 🔢 There are **{int(mineral_count)}** occurrences of {selected_mineral} in this region")
            
            # Calculate density
            region_area = geom_plot['area_sqkm'].sum()
            if region_area > 0:
                density = mineral_count / region_area
                st.markdown(f"- 📊 Density of {selected_mineral}: **{density:.4f}** occurrences per km²")
            
            # Check ranking
            if col_name in geomorph_gdf.columns:
                # Count occurrences by region
                region_counts = geomorph_gdf.groupby('legend_sho')[col_name].sum()
                region_counts = region_counts[region_counts > 0].sort_values(ascending=False)
                
                if selected_geomorph in region_counts.index:
                    rank = list(region_counts.index).index(selected_geomorph) + 1
                    total_regions = len(region_counts)
                    st.markdown(f"- 🏆 This region ranks **#{rank}** out of {total_regions} regions for {selected_mineral} occurrences")
            
            # Provide an interpretation with colorful background
            if mineral_percentage > 50:
                st.markdown("""
                <div style="background: linear-gradient(120deg, #d4fc79 0%, #96e6a1 100%); padding: 15px; border-radius: 10px; margin-top: 15px;">
                    <b>Interpretation:</b> This region is a <b>dominant source</b> for this mineral, with over half of all minerals being the selected type.
                </div>
                """, unsafe_allow_html=True)
            elif mineral_percentage > 25:
                st.markdown("""
                <div style="background: linear-gradient(120deg, #84fab0 0%, #8fd3f4 100%); padding: 15px; border-radius: 10px; margin-top: 15px;">
                    <b>Interpretation:</b> This region is a <b>significant source</b> for this mineral, with a substantial proportion of minerals being the selected type.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: linear-gradient(120deg, #e0c3fc 0%, #8ec5fc 100%); padding: 15px; border-radius: 10px; margin-top: 15px;">
                    <b>Interpretation:</b> This region contains the selected mineral, but it's not a dominant mineral in this region.
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning(f"No {selected_mineral} was found in {selected_geomorph}.")
            st.markdown("### Regions with this mineral:")
            
            # Show top regions for this mineral
            if col_name in geomorph_gdf.columns:
                region_counts = geomorph_gdf.groupby('legend_sho')[col_name].sum()
                region_counts = region_counts[region_counts > 0].sort_values(ascending=False).head(5)
                
                if not region_counts.empty:
                    fig, ax = plt.subplots(figsize=(6, 4), facecolor='#f0f2f6')
                    ax.set_facecolor('#f0f2f6')
                    
                    # Create a gradient of the mineral's color
                    cmap = mcolors.LinearSegmentedColormap.from_list(
                        'custom_cmap', 
                        ['#ffffff', commodity_colors.get(selected_mineral, '#3388ff')]
                    )
                    
                    bars = region_counts.plot.barh(ax=ax, color=cmap(np.linspace(0.5, 1, len(region_counts))))
                    
                    # Add count labels
                    for i, v in enumerate(region_counts):
                        ax.text(v + 0.1, i, str(int(v)), color='black', fontweight='bold', va='center')
                    
                    ax.set_xlabel('Number of Occurrences', fontweight='bold')
                    ax.set_title(f'Top 5 Regions for {selected_mineral}', fontsize=14, fontweight='bold', pad=20)
                    
                    # Remove the y-axis labels to avoid repetition with the legend
                    ax.set_yticks([])
                    
                    # Add a legend instead
                    ax.legend(region_counts.index, title='Regions', loc='lower right',
                             fontsize=8, title_fontsize=10)
                    
                    # Add a border
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_color('black')
                        spine.set_linewidth(1)
                    
                    st.pyplot(fig)
    
    elif only_geomorph:
        # Show mineral composition in this region
        st.markdown('<div class="section-header">📊 Mineral Composition in ' + f"{selected_geomorph}</div>", unsafe_allow_html=True)
        
        # Get all mineral columns
        mineral_cols = [col for col in geom_plot.columns if col.startswith('count_')]
        
        # Create a dictionary of mineral counts
        mineral_counts = {}
        for col in mineral_cols:
            mineral_name = col.replace('count_', '').replace('_', ' ')
            count = geom_plot[col].sum()
            if count > 0:
                mineral_counts[mineral_name] = count
        
        if mineral_counts:
            # Sort by count
            sorted_minerals = sorted(mineral_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Create a colorful pie chart of top minerals
            fig, ax = plt.subplots(figsize=(6, 6), facecolor='#f0f2f6')
            ax.set_facecolor('#f0f2f6')
            
            labels = []
            sizes = []
            colors = []
            explode = []
            total = sum(count for _, count in sorted_minerals)
            other_count = 0
            
            # Get top 6 minerals, group others
            for i, (mineral, count) in enumerate(sorted_minerals):
                if i < 6:
                    labels.append(mineral)
                    sizes.append(count)
                    colors.append(commodity_colors.get(mineral, '#999999'))
                    explode.append(0.05 if i == 0 else 0)  # Explode only the first slice
                else:
                    other_count += count
            
            if other_count > 0:
                labels.append('Other Minerals')
                sizes.append(other_count)
                colors.append('#CCCCCC')
                explode.append(0)
            
            wedges, texts, autotexts = ax.pie(
                sizes, 
                explode=explode,
                labels=labels, 
                colors=colors, 
                autopct='%1.1f%%', 
                startangle=90, 
                shadow=True,
                wedgeprops={'edgecolor': 'black', 'linewidth': 1},
                textprops={'fontsize': 9}
            )
            
            # Enhance text visibility
            for text in texts:
                text.set_color('#333333')
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(8)
            
            ax.axis('equal')
            ax.set_title(f'Mineral Composition in {selected_geomorph}', fontsize=14, fontweight='bold', pad=20)
            
            # Add a border
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(2)
            
            st.pyplot(fig)
            
            # Show top minerals in a list with colored badges
            st.markdown("""
            <h3 style="background: linear-gradient(90deg, #ffd3b6 0%, #ffaaa5 100%); padding: 10px; border-radius: 5px;">
                💎 Top Minerals in this Region
            </h3>
            """, unsafe_allow_html=True)
            
            for mineral, count in sorted_minerals[:8]:  # Show top 8
                percentage = (count / total) * 100
                color = commodity_colors.get(mineral, '#999999')
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 5px;">
                    <div style="background-color: {color}; width: 15px; height: 15px; border-radius: 50%; margin-right: 10px;"></div>
                    <b>{mineral}</b>: {int(count)} occurrences ({percentage:.1f}%)
                </div>
                """, unsafe_allow_html=True)
                
            # Show total minerals and density with a colorful box
            total_minerals = geom_plot['mineral_count'].sum()
            region_area = geom_plot['area_sqkm'].sum()
            density = total_minerals / region_area if region_area > 0 else 0
            
            st.markdown(f"""
            <div style="background: linear-gradient(120deg, #84fab0 0%, #8fd3f4 100%); padding: 15px; border-radius: 10px; margin-top: 15px;">
                <b>Total Minerals</b>: {int(total_minerals)}<br>
                <b>Mineral Density</b>: {density:.4f} per km²
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning(f"No minerals found in {selected_geomorph}.")
    
    elif only_mineral:
        # Show distribution of this mineral across regions
        st.markdown('<div class="section-header">📊 Distribution of ' + f"{selected_mineral}</div>", unsafe_allow_html=True)
        
        col_name = f"count_{selected_mineral.replace(' ', '_').replace(',', '_').replace('-', '_')}"
        if col_name in geomorph_gdf.columns:
            # Count occurrences by region
            region_counts = geomorph_gdf.groupby('legend_sho')[col_name].sum()
            region_counts = region_counts[region_counts > 0].sort_values(ascending=False)
            
            if not region_counts.empty:
                # Create colorful bar chart for top regions
                fig, ax = plt.subplots(figsize=(6, 6), facecolor='#f0f2f6')
                ax.set_facecolor('#f0f2f6')
                
                # Use a color gradient based on the mineral's color
                mineral_color = commodity_colors.get(selected_mineral, '#3388ff')
                cmap = mcolors.LinearSegmentedColormap.from_list(
                    'custom_cmap', 
                    ['#ffffff', mineral_color]
                )
                
                # Plot only top 10 for readability
                top_regions = region_counts.head(10)
                bars = top_regions.plot.barh(
                    ax=ax, 
                    color=cmap(np.linspace(0.5, 1, len(top_regions)))
                )
                
                # Add count labels
                for i, v in enumerate(top_regions):
                    ax.text(v + 0.1, i, str(int(v)), color='black', fontweight='bold', va='center')
                
                ax.set_xlabel('Number of Occurrences', fontweight='bold')
                ax.set_title(f'Top 10 Regions for {selected_mineral}', fontsize=14, fontweight='bold', pad=20)
                
                # Remove the y-axis labels to avoid repetition with the legend
                ax.set_yticks([])
                
                # Add a legend instead
                ax.legend(top_regions.index, title='Regions', loc='lower right',
                         fontsize=8, title_fontsize=10)
                
                # Add a border
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color('black')
                    spine.set_linewidth(1)
                
                st.pyplot(fig)
                
                # Show percentage distribution with a gradient progress bar
                st.markdown("""
                <h3 style="background: linear-gradient(90deg, #ffd3b6 0%, #ffaaa5 100%); padding: 10px; border-radius: 5px;">
                    🌍 Regional Distribution
                </h3>
                """, unsafe_allow_html=True)
                
                total = region_counts.sum()
                for region, count in region_counts.head(8).items():
                    percentage = (count / total) * 100
                    
                    # Create a gradient progress bar
                    st.markdown(f"""
                    <div style="margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <b>{region}</b>
                            <span>{int(count)} occurrences ({percentage:.1f}%)</span>
                        </div>
                        <div style="background-color: #eee; border-radius: 10px; height: 10px; width: 100%;">
                            <div style="background: linear-gradient(90deg, {mineral_color} 0%, {mineral_color}80 100%); width: {percentage}%; height: 10px; border-radius: 10px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Total occurrences in a colored box
                st.markdown(f"""
                <div style="background: linear-gradient(120deg, #84fab0 0%, #8fd3f4 100%); padding: 15px; border-radius: 10px; margin-top: 15px; text-align: center;">
                    <h3 style="margin: 0;">Total {selected_mineral} Occurrences: {int(total)}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Mineral concentration statistics
                st.markdown("""
                <h3 style="background: linear-gradient(90deg, #a1c4fd 0%, #c2e9fb 100%); padding: 10px; border-radius: 5px; margin-top: 15px;">
                    🔍 Highest Concentration Regions
                </h3>
                """, unsafe_allow_html=True)
                
                concentrations = []
                for region in region_counts.index:
                    region_minerals = geomorph_gdf[geomorph_gdf['legend_sho'] == region]['mineral_count'].sum()
                    if region_minerals > 0:
                        mineral_count = region_counts[region]
                        concentration = (mineral_count / region_minerals) * 100
                        concentrations.append((region, concentration))
                
                if concentrations:
                    concentrations.sort(key=lambda x: x[1], reverse=True)
                    for region, concentration in concentrations[:5]:
                        # Create color intensity based on concentration
                        intensity = min(100, concentration) / 100
                        bg_color = mcolors.to_hex(cmap(intensity))
                        
                        st.markdown(f"""
                        <div style="background-color: {bg_color}; padding: 8px; border-radius: 5px; margin-bottom: 5px; color: black;">
                            <b>{region}</b>: {concentration:.1f}% of minerals are {selected_mineral}
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning(f"No occurrences of {selected_mineral} found.")
        else:
            st.warning(f"No data available for {selected_mineral}.")
    
    else:
        # Show overall mineral distribution
        st.markdown('<div class="section-header">📊 Overall Mineral Distribution</div>', unsafe_allow_html=True)
        
        # Count by commodity
        mineral_counts = mineral_gdf['commodity'].value_counts().head(10)
        
        if not mineral_counts.empty:
            # Create colorful pie chart
            fig, ax = plt.subplots(figsize=(6, 6), facecolor='#f0f2f6')
            ax.set_facecolor('#f0f2f6')
            
            # Get colors for minerals
            colors = [commodity_colors.get(mineral, '#999999') for mineral in mineral_counts.index]
            
            wedges, texts, autotexts = ax.pie(
                mineral_counts, 
                labels=mineral_counts.index,
                colors=colors, 
                autopct='%1.1f%%', 
                startangle=90, 
                shadow=True,
                wedgeprops={'edgecolor': 'black', 'linewidth': 1},
                textprops={'fontsize': 9}
            )
            
            # Enhance text visibility
            for text in texts:
                text.set_color('#333333')
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(8)
            
            ax.axis('equal')
            ax.set_title('Overall Mineral Distribution', fontsize=14, fontweight='bold', pad=20)
            
            # Add a border
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(2)
            
            st.pyplot(fig)
            
            # Show counts in a colorful list
            st.markdown("""
            <h3 style="background: linear-gradient(90deg, #ffd3b6 0%, #ffaaa5 100%); padding: 10px; border-radius: 5px;">
                💎 Top Minerals
            </h3>
            """, unsafe_allow_html=True)
            
            total = mineral_counts.sum()
            for mineral, count in mineral_counts.items():
                percentage = (count / total) * 100
                color = commodity_colors.get(mineral, '#999999')
                
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 5px;">
                    <div style="background-color: {color}; width: 15px; height: 15px; border-radius: 50%; margin-right: 10px;"></div>
                    <b>{mineral}</b>: {count} occurrences ({percentage:.1f}%)
                </div>
                """, unsafe_allow_html=True)
            
            # Show total mineral count in a colorful box
            st.markdown(f"""
            <div style="background: linear-gradient(120deg, #84fab0 0%, #8fd3f4 100%); padding: 15px; border-radius: 10px; margin-top: 15px; text-align: center;">
                <h3 style="margin: 0;">Total Minerals: {int(total)}</h3>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("No mineral data available.")

# Add fancy footer
st.markdown("""
<div class="footer">
    <p><b>🌍 Mineral Targeting Dashboard</b> | A geospatial analysis tool for mineral exploration</p>
    <p>Visualizing the relationship between geomorphological regions and mineral occurrences</p>
</div>
""", unsafe_allow_html=True)
