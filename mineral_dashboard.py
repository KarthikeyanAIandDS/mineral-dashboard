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
# File path helper function
def get_file_path(relative_path):
    """Get absolute file path from relative path."""
    import os
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Join with the relative path
    return os.path.join(script_dir, relative_path)

# Add informative message about file paths
st.set_page_config(
    page_title="Mineral Targeting Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)
def load_data():
    try:
        # Use the helper function to get file paths
        # Replace these paths with your actual file paths
        geomorph_path = get_file_path("datasets/multi_layer_geological_map_of_karnataka_and_andhra_pradesh_25k_scale_v1/25K/lithology_25k_ngdr_20250224140917945/lithology_25k_ngdr.shp")
        mineral_path = get_file_path("datasets/multi_layer_geological_map_of_karnataka_and_andhra_pradesh_25k_scale_v1/25K/mineralization_25k_ngdr_20250224141143411/mineralization_25k_ngdr_20250224141143411.shp")
        
        # Show file paths to help with debugging
        st.write(f"Loading geomorphology from: {geomorph_path}")
        st.write(f"Loading minerals from: {mineral_path}")
        
        # Read the files
        geomorph_gdf = gpd.read_file(geomorph_path)
        mineral_gdf = gpd.read_file(mineral_path)
        
        # Continue with your existing code...
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.write("Please check if these files exist in your repository:")
        st.code(geomorph_path)
        st.code(mineral_path)
        return None, None

# Set page configuration
st.set_page_config(
    page_title="Mineral Targeting Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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
        ax.set_axis_off()
        
        # Add a grid
        ax.grid(True, linestyle='--', alpha=0.5)
        
        st.pyplot(fig)
    
    elif both_selected:
        # Create a map showing the selected region and mineral distribution
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot all regions with a light gray color
        geomorph_gdf.plot(
            ax=ax,
            color='lightgray',
            edgecolor='darkgray',
            linewidth=0.5
        )
        
        # Highlight the selected region
        col_name = f"count_{selected_mineral.replace(' ', '_').replace(',', '_').replace('-', '_')}"
        if col_name in geom_plot.columns and geom_plot[col_name].sum() > 0:
            # If the mineral is present in the region, color it with the mineral's color
            geom_plot.plot(
                ax=ax,
                color=commodity_colors.get(selected_mineral, '#3388ff'),
                edgecolor='black',
                linewidth=1
            )
        else:
            # If the mineral is not present, use a different color
            geom_plot.plot(
                ax=ax,
                color='#ff3333',  # Red color for regions without the mineral
                edgecolor='black',
                linewidth=1
            )
        
        # Plot mineral locations
        for idx, row in min_plot.iterrows():
            try:
                ax.plot(
                    row.geometry.centroid.x,
                    row.geometry.centroid.y,
                    marker='o',
                    color=commodity_colors.get(selected_mineral, '#999999'),
                    markersize=8,
                    markeredgecolor='white',
                    markeredgewidth=1
                )
            except:
                continue
        
        # Create a custom legend
        handles = [
            mpatches.Patch(color=commodity_colors.get(selected_mineral, '#3388ff'), 
                          label=f'Region with {selected_mineral}'),
            mpatches.Patch(color='#ff3333', label='Region without mineral'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=commodity_colors.get(selected_mineral, '#999999'),
                      markersize=8, label=f'{selected_mineral} occurrence')
        ]
        
        # Place legend to the right of the plot
        ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_title(f'{selected_mineral} in {selected_geomorph}')
        ax.set_axis_off()
        
        # Add a grid
        ax.grid(True, linestyle='--', alpha=0.5)
        
        st.pyplot(fig)
    
    elif only_geomorph:
        # Create a map highlighting the selected region
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot all regions with a light gray color
        geomorph_gdf.plot(
            ax=ax,
            color='lightgray',
            edgecolor='darkgray',
            linewidth=0.5
        )
        
        # Highlight the selected region with a distinctive color
        region_color = geomorph_colors.get(selected_geomorph, '#3388ff')
        geom_plot.plot(
            ax=ax,
            color=region_color,
            edgecolor='black',
            linewidth=1
        )
        
        # Create a custom legend
        handles = [
            mpatches.Patch(color=region_color, label=selected_geomorph),
            mpatches.Patch(color='lightgray', label='Other regions')
        ]
        
        # Place legend to the right of the plot
        ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_title(f'Region: {selected_geomorph}')
        ax.set_axis_off()
        
        # Add a grid
        ax.grid(True, linestyle='--', alpha=0.5)
        
        st.pyplot(fig)
    
    elif only_mineral:
        # Create a map showing all regions with the selected mineral
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot all regions with a light gray color
        geomorph_gdf.plot(
            ax=ax,
            color='lightgray',
            edgecolor='darkgray',
            linewidth=0.5
        )
        
        # Find regions that have this mineral
        col_name = f"count_{selected_mineral.replace(' ', '_').replace(',', '_').replace('-', '_')}"
        if col_name in geomorph_gdf.columns:
            regions_with_mineral = geomorph_gdf[geomorph_gdf[col_name] > 0]
            
            if not regions_with_mineral.empty:
                # Color regions that have this mineral
                regions_with_mineral.plot(
                    ax=ax,
                    color=commodity_colors.get(selected_mineral, '#3388ff'),
                    edgecolor='black',
                    linewidth=0.5
                )
        
        # Plot mineral locations
        for idx, row in min_plot.iterrows():
            try:
                ax.plot(
                    row.geometry.centroid.x,
                    row.geometry.centroid.y,
                    marker='o',
                    color=commodity_colors.get(selected_mineral, '#999999'),
                    markersize=8,
                    markeredgecolor='white',
                    markeredgewidth=1
                )
            except:
                continue
        
        # Create a custom legend
        handles = [
            mpatches.Patch(color=commodity_colors.get(selected_mineral, '#3388ff'), 
                          label=f'Regions with {selected_mineral}'),
            mpatches.Patch(color='lightgray', label='Regions without mineral'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=commodity_colors.get(selected_mineral, '#999999'),
                      markersize=8, label=f'{selected_mineral} occurrence')
        ]
        
        # Place legend to the right of the plot
        ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_title(f'Distribution of {selected_mineral}')
        ax.set_axis_off()
        
        # Add a grid
        ax.grid(True, linestyle='--', alpha=0.5)
        
        st.pyplot(fig)

with plot_col:
    # Show charts and analyses based on the selection
    if both_selected:
        # Show potential and analysis for the specific mineral in the region
        st.subheader(f"{selected_mineral} Analysis")
        
        # Check if the mineral is present in the region
        if mineral_count > 0:
            # Create pie chart showing this mineral vs others in the region
            fig, ax = plt.subplots(figsize=(6, 6))
            other_minerals = total_minerals - mineral_count
            labels = [selected_mineral, 'Other Minerals']
            sizes = [mineral_count, other_minerals]
            colors = [commodity_colors.get(selected_mineral, '#999999'), '#CCCCCC']
            explode = (0.1, 0)  # explode the 1st slice (selected mineral)
            
            ax.pie(sizes, explode=explode, labels=labels, colors=colors, 
                   autopct='%1.1f%%', startangle=90, shadow=True)
            ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
            ax.set_title(f'Proportion of {selected_mineral} in {selected_geomorph}')
            st.pyplot(fig)
            
            # Show some statistics and interpretation
            st.markdown(f"### Key Findings")
            st.markdown(f"- {selected_mineral} makes up **{mineral_percentage:.1f}%** of all minerals in this region")
            st.markdown(f"- There are **{int(mineral_count)}** occurrences of {selected_mineral} in this region")
            
            # Calculate density
            region_area = geom_plot['area_sqkm'].sum()
            if region_area > 0:
                density = mineral_count / region_area
                st.markdown(f"- Density of {selected_mineral}: **{density:.4f}** occurrences per km²")
            
            # Check if this is one of the top regions for this mineral
            if col_name in geomorph_gdf.columns:
                # Count occurrences by region
                region_counts = geomorph_gdf.groupby('legend_sho')[col_name].sum()
                region_counts = region_counts[region_counts > 0].sort_values(ascending=False)
                
                if selected_geomorph in region_counts.index:
                    rank = list(region_counts.index).index(selected_geomorph) + 1
                    total_regions = len(region_counts)
                    st.markdown(f"- This region ranks **#{rank}** out of {total_regions} regions for {selected_mineral} occurrences")
            
            # Provide an interpretation
            if mineral_percentage > 50:
                st.markdown(f"**Interpretation:** {selected_geomorph} is a **dominant region** for {selected_mineral}, with over half of all minerals in this region being {selected_mineral}.")
            elif mineral_percentage > 25:
                st.markdown(f"**Interpretation:** {selected_geomorph} is a **significant region** for {selected_mineral}, with a substantial proportion of minerals being {selected_mineral}.")
            else:
                st.markdown(f"**Interpretation:** {selected_geomorph} contains {selected_mineral}, but it's not a dominant mineral in this region.")
        else:
            st.warning(f"No {selected_mineral} was found in {selected_geomorph}.")
            st.markdown("### Regions with this mineral:")
            
            # Show top regions for this mineral
            if col_name in geomorph_gdf.columns:
                region_counts = geomorph_gdf.groupby('legend_sho')[col_name].sum()
                region_counts = region_counts[region_counts > 0].sort_values(ascending=False).head(5)
                
                if not region_counts.empty:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    region_counts.plot.barh(ax=ax, color=commodity_colors.get(selected_mineral, '#999999'))
                    ax.set_xlabel('Number of Occurrences')
                    ax.set_title(f'Top 5 Regions for {selected_mineral}')
                    st.pyplot(fig)
    
    elif only_geomorph:
        # Show general mineral composition in this region
        st.subheader(f"Minerals in {selected_geomorph}")
        
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
            
            # Create a pie chart of top minerals
            fig, ax = plt.subplots(figsize=(6, 6))
            
            labels = []
            sizes = []
            colors = []
            total = sum(count for _, count in sorted_minerals)
            other_count = 0
            
            # Get top 6 minerals, group others
            for i, (mineral, count) in enumerate(sorted_minerals):
                if i < 6:
                    labels.append(mineral)
                    sizes.append(count)
                    colors.append(commodity_colors.get(mineral, '#999999'))
                else:
                    other_count += count
            
            if other_count > 0:
                labels.append('Other Minerals')
                sizes.append(other_count)
                colors.append('#CCCCCC')
            
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, shadow=True)
            ax.axis('equal')
            ax.set_title(f'Mineral Composition in {selected_geomorph}')
            st.pyplot(fig)
            
            # Show top minerals in a list
            st.markdown("### Top Minerals in this Region:")
            for mineral, count in sorted_minerals[:8]:  # Show top 8
                percentage = (count / total) * 100
                st.markdown(f"- **{mineral}**: {int(count)} occurrences ({percentage:.1f}%)")
                
            # Show total minerals and density
            total_minerals = geom_plot['mineral_count'].sum()
            region_area = geom_plot['area_sqkm'].sum()
            if region_area > 0:
                density = total_minerals / region_area
                st.markdown(f"**Total Minerals**: {int(total_minerals)}")
                st.markdown(f"**Mineral Density**: {density:.4f} per km²")
        else:
            st.warning(f"No minerals found in {selected_geomorph}.")
    
    elif only_mineral:
        # Show distribution of this mineral across regions
        st.subheader(f"Distribution of {selected_mineral}")
        
        col_name = f"count_{selected_mineral.replace(' ', '_').replace(',', '_').replace('-', '_')}"
        if col_name in geomorph_gdf.columns:
            # Count occurrences by region
            region_counts = geomorph_gdf.groupby('legend_sho')[col_name].sum()
            region_counts = region_counts[region_counts > 0].sort_values(ascending=False)
            
            if not region_counts.empty:
                # Create bar chart for top regions
                fig, ax = plt.subplots(figsize=(6, 6))
                region_counts.head(10).plot.barh(ax=ax, color=commodity_colors.get(selected_mineral, '#999999'))
                ax.set_xlabel('Number of Occurrences')
                ax.set_title(f'Top 10 Regions for {selected_mineral}')
                st.pyplot(fig)
                
                # Show percentage distribution
                st.markdown("### Regional Distribution:")
                total = region_counts.sum()
                for region, count in region_counts.head(8).items():
                    percentage = (count / total) * 100
                    st.markdown(f"- **{region}**: {int(count)} occurrences ({percentage:.1f}%)")
                
                # Total occurrences
                st.markdown(f"**Total {selected_mineral} Occurrences**: {int(total)}")
                
                # Mineral concentration statistics
                concentrations = []
                for region in region_counts.index:
                    region_minerals = geomorph_gdf[geomorph_gdf['legend_sho'] == region]['mineral_count'].sum()
                    if region_minerals > 0:
                        mineral_count = region_counts[region]
                        concentration = (mineral_count / region_minerals) * 100
                        concentrations.append((region, concentration))
                
                if concentrations:
                    st.markdown("### Highest Concentration Regions:")
                    concentrations.sort(key=lambda x: x[1], reverse=True)
                    for region, concentration in concentrations[:5]:
                        st.markdown(f"- **{region}**: {concentration:.1f}% of minerals are {selected_mineral}")
            else:
                st.warning(f"No occurrences of {selected_mineral} found.")
        else:
            st.warning(f"No data available for {selected_mineral}.")
    
    else:
        # Show overall mineral distribution
        st.subheader("Overall Mineral Distribution")
        
        # Count by commodity
        mineral_counts = mineral_gdf['commodity'].value_counts().head(10)
        
        if not mineral_counts.empty:
            # Create pie chart
            fig, ax = plt.subplots(figsize=(6, 6))
            
            # Get colors for minerals
            colors = [commodity_colors.get(mineral, '#999999') for mineral in mineral_counts.index]
            
            mineral_counts.plot.pie(ax=ax, colors=colors, autopct='%1.1f%%', shadow=True)
            ax.set_ylabel('')  # Remove "commodity" label
            ax.set_title('Overall Mineral Distribution')
            st.pyplot(fig)
            
            # Show counts in a list
            st.markdown("### Top Minerals:")
            total = mineral_counts.sum()
            for mineral, count in mineral_counts.items():
                percentage = (count / total) * 100
                st.markdown(f"- **{mineral}**: {count} occurrences ({percentage:.1f}%)")
            
            # Show total mineral count
            st.markdown(f"**Total Minerals**: {int(total)}")
        else:
            st.warning("No mineral data available.")

# Add footer
st.markdown("""
<div style="background-color: #f8f9fa; padding: 5px; border-radius: 5px; margin-top: 10px; font-size: 11px; color: #666; text-align: center;">
    This dashboard visualizes the relationship between geomorphological regions and mineral occurrences.
</div>
""", unsafe_allow_html=True)
