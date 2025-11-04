from shiny import render
from shiny.express import input, ui, app
import osmnx as ox
import networkx as nx
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from shapely import MultiPolygon, Polygon
from shapely.ops import unary_union
import geopandas as gpd
from shapely.geometry import Point
from descartes import PolygonPatch
import matplotlib.colors as mcolors
from shiny import reactive
from dotenv import load_dotenv
import os

from load_static import DataLoader
from calculate_isochrones import calculate_isochrones
from weight_calculation import classic_weight_calculations, GNN_weight_calculations


# Global parameters
load_dotenv()


# Base directory from which to load weather data

BASE_DATA_DIR = os.getenv("BASE_DATA_DIR", r"C:\Users\elchi\Desktop\UIC_Chicago\Knowledge_graph\Backend\try_clean\good_candidates\20250706\t00z\outputs")


# Construct data carrying object
data_loader = DataLoader()

# Define paths to weather data files

data_loader.load_graph(graph_path=r"C:\Users\elchi\Desktop\UIC_Chicago\Knowledge_graph\Backend\Backtracking\chicago.graphml")


G = data_loader.G
rain_data = data_loader.rain_data


# --- UI ---
ui.page_opts(title="OSM Route Viewer", fillable=True)
with ui.layout_sidebar():
    # Sidebar - Controls
    with ui.sidebar():
        ui.input_text("origin", "Enter Origin (lat, lon)", "41.9814866, -87.8593659")
        ui.input_text("destination", "Enter Destination (lat, lon)", "41.7905674, -87.5831307")
        
        # @render.text
        # def txt():
        #     return f"Origin: {input.origin()}, Destination: {input.destination()}"
        
        # ui.input_checkbox("show_rain", "Show Rain Overlay", True)
        # ui.input_slider("rain_alpha", "Rain Transparency", 0.1, 1.0, 0.4)
        ui.input_checkbox_group(  
            "weight_type",  
            "Weights to Use:",  
            {  
                "rain": "Rain-aware",  
                "heat": "Heat-aware",  
                "wind": "Wind-aware",
                "humidity": "Humidity-aware",
            },  
        )
        
        ui.input_radio_buttons(  
        "weight_calculation_method",  
            "Weight Calculation Method:",  
            {"GNN": "GNN", "Classic": "Classic"},  
        )  
        
        ui.input_radio_buttons(
            "graph_show",
            "Route Visualization:",
            {"single": "Single map with single route",
             "single_m": "A single map with multiple routes",
             "multi": "A graph per weight type"},
            selected="single",
        ) 


        @render.ui
        def weight_adjustments():
            selected_weights = input.weight_type()
            
            if selected_weights:  
                sliders = []
                
                # Rain weight slider - only show if rain is selected
                if "rain" in selected_weights:
                    sliders.append(
                        ui.input_slider(
                            "rain_weight", 
                            "Rain Weight:", 
                            min=0.0, max=2.0, value=0.858, step=0.001
                        )
                    )

                if "heat" in selected_weights:
                    sliders.append(
                        ui.input_slider(
                            "heat_weight", 
                            "Heat Weight:", 
                            min=0.0, max=2.0, value=0.0285, step=0.001
                        )
                    )

                if "wind" in selected_weights:
                    sliders.append(
                        ui.input_slider(
                            "wind_weight", 
                            "Wind Weight:", 
                            min=0.0, max=2.0, value=0.0965, step=0.001
                        )
                    )
                

                if "humidity" in selected_weights:
                    sliders.append(
                        ui.input_slider(
                            "humidity_weight", 
                            "Humidity Weight:", 
                            min=0.0, max=2.0, value=0.0167, step=0.001
                        )
                    )
                
                if sliders:
                    return ui.div(
                        ui.h4("Weight Adjustments", style="margin-top: 20px;"),
                        ui.div(
                            *sliders,  # Unpack the list of sliders
                            style="border: 1px solid #ddd; padding: 15px; border-radius: 5px; background-color: #f9f9f9;"
                        )
                    )
            
            return ui.div() 

     

    @render.text
    def main_content():
        return "Map will be displayed here when ready. Click 'Generate Plot' to create visualization."
    
    ui.input_action_button("generate_plot", "Generate Plot")

    
    @reactive.calc
    @reactive.event(input.generate_plot)
    def generated_fig():
            try:
                orig_lat, orig_lon = map(float, input.origin().split(","))
                dest_lat, dest_lon = map(float, input.destination().split(","))

                orig_node = ox.distance.nearest_nodes(G, X=orig_lon, Y=orig_lat)
                dest_node = ox.distance.nearest_nodes(G, X=dest_lon, Y=dest_lat)

            

               
                route = nx.shortest_path(G, orig_node, dest_node, weight="travel_time")
                

                trip_times_seconds = calculate_isochrones(G, orig_node, dest_node, route)
                
                
                rain_data, lats, lons, rain_ds = data_loader.load_rain_data(
                    rain_data_path= BASE_DATA_DIR + "\RAIN.nc"
                )
                
                heat_data, lats, lons, heat_ds = data_loader.load_heat_index_data(
                    heat_index_path= BASE_DATA_DIR + "\T2.nc"
                )
                wind_speed_data, wind_dir_data, lats, lons, wind_speed_ds, wind_dir_ds = data_loader.load_wind_data(
                    wind_speed_path= BASE_DATA_DIR + "\WSPD10.nc",
                    wind_direction_path= BASE_DATA_DIR + "\WDIR10.nc"
                )
                humidity_data, lats, lons, humidity_ds = data_loader.load_relative_humidity_data(
                    rh_data_path= BASE_DATA_DIR + "\RH2.nc"
                )



                #Compute weights

                if input.weight_calculation_method() == "Classic":
                    classic_weight_calculations(
                        G, lats, lons,
                        rain_ds=rain_ds,
                        heat_ds=heat_ds,
                        wind_speed_ds=wind_speed_ds,
                        wind_dir_ds=wind_dir_ds,
                        humidity_ds=humidity_ds,
                        time=data_loader.time,
                        rain_weight=input.rain_weight() if "rain" in input.weight_type() else None,
                        heat_weight=input.heat_weight() if "heat" in input.weight_type() else None,
                        wind_weight=input.wind_weight() if "wind" in input.weight_type() else None,
                        humidity_weight=input.humidity_weight() if "humidity" in input.weight_type() else None,
                    )
                else:
                    selected_weights = input.weight_type()
                    optional_weights = {}
                    if "rain" in selected_weights:
                        optional_weights['rain_weight'] = input.rain_weight()
                    if "heat" in selected_weights:
                        optional_weights['heat_weight'] = input.heat_weight()
                    if "wind" in selected_weights:
                        optional_weights['wind_weight'] = input.wind_weight()
                    if "humidity" in selected_weights:
                        optional_weights['humidity_weight'] = input.humidity_weight()

                    rain_dataset = GNN_weight_calculations(G, lats, lons,
                        rain_ds=rain_ds,
                        heat_ds=heat_data,
                        wind_speed_ds=wind_speed_data,
                        wind_dir_ds=wind_dir_data,
                        humidity_ds=humidity_data,
                        time=data_loader.time,
                        trip_time_seconds=trip_times_seconds,
                        **optional_weights # **kwargs
                    )
                    
                    
                
                    
                
              
                fig, ax = plt.subplots(figsize=(10, 10))

                # Isochrone colors
                iso_colors = ox.plot.get_colors(n=len(trip_times_seconds), cmap='plasma', start=0)

                # Node coloring
                node_colors = {}
                for trip_time, color in zip(sorted(trip_times_seconds, reverse=True), iso_colors):
                    subgraph = nx.ego_graph(G, orig_node, radius=trip_time, distance='time')
                    for node in subgraph.nodes():
                        node_colors[node] = color

                nc = [node_colors.get(node, 'none') for node in G.nodes()]
                ns = [15 if node in node_colors else 0 for node in G.nodes()]

                # Plot graph with isochrone coloring
                ox.plot_graph(G, ax=ax, node_color=nc, node_size=ns, node_alpha=0.8,
                            edge_color='gray', edge_linewidth=1, show=False, close=False)

                if input.graph_show() == "single_m":
                    selected_weights = input.weight_type() or []
                    routes = []
                    route_colors = []
                    route_labels = []
                    
                    # Always show baseline fastest route for comparison
                    baseline_route = nx.shortest_path(G, orig_node, dest_node, weight="travel_time")
                    routes.append(baseline_route)
                    route_colors.append("black")
                    route_labels.append("Fastest (no weather)")
                    
                    if selected_weights:
                        if "rain" in selected_weights:
                            try:
                                rainy_route = nx.shortest_path(G, orig_node, dest_node, weight="rain_weight")
                                routes.append(rainy_route)
                                route_colors.append("blue")
                                route_labels.append("Rain-aware")
                            except (KeyError, nx.NetworkXNoPath) as e:
                                print(f"Rain route failed: {e}")
                        
                        if "heat" in selected_weights:
                            try:
                                heat_route = nx.shortest_path(G, orig_node, dest_node, weight="heat_weight")
                                routes.append(heat_route)
                                route_colors.append("orange") 
                                route_labels.append("Heat-aware")
                            except (KeyError, nx.NetworkXNoPath) as e:
                                print(f"Heat route failed: {e}")
                        
                        if "wind" in selected_weights:
                            try:
                                wind_route = nx.shortest_path(G, orig_node, dest_node, weight="wind_weight")
                                routes.append(wind_route)
                                route_colors.append("green")
                                route_labels.append("Wind-aware")
                            except (KeyError, nx.NetworkXNoPath) as e:
                                print(f"Wind route failed: {e}")
                        
                        if "humidity" in selected_weights:
                            try:
                                humidity_route = nx.shortest_path(G, orig_node, dest_node, weight="humidity_weight")
                                routes.append(humidity_route)
                                route_colors.append("purple")
                                route_labels.append("Humidity-aware")
                            except (KeyError, nx.NetworkXNoPath) as e:
                                print(f"Humidity route failed: {e}")
                                
                    ox.plot_graph_routes(G, routes=routes, route_colors=route_colors,
                                        route_linewidth=3, node_size=0, ax=ax, show=False, close=False)


                elif input.graph_show() == "single":
                    # Single route with combined weights
                    try:
                        route = nx.shortest_path(G, orig_node, dest_node, weight="total_weight")
                        routes = [route]
                        route_colors = ["red"]
                        route_labels = ["Combined weather-aware"]
                    except (KeyError, nx.NetworkXNoPath) as e:
                        print(f"Combined route failed: {e}, falling back to travel_time")
                        route = nx.shortest_path(G, orig_node, dest_node, weight="travel_time")
                        routes = [route]
                        route_colors = ["red"]
                        route_labels = ["Fastest (fallback)"]



                    # Plot route
                    ox.plot_graph_routes(G, routes=routes, route_colors=route_colors,
                                        route_linewidth=3, node_size=0, ax=ax, show=False, close=False)



                elif input.graph_show() == "multi":
                    selected_weights = input.weight_type()
                    
                    if not selected_weights:
                        
                        fig, ax = plt.subplots(figsize=(10, 10))
                        
                        # Plot base graph
                        ox.plot_graph(G, ax=ax, node_color=nc, node_size=ns, node_alpha=0.8,
                                    edge_color='gray', edge_linewidth=1, show=False, close=False)
                        
                        baseline_route = nx.shortest_path(G, orig_node, dest_node, weight="travel_time")
                        ox.plot_graph_routes(G, routes=[baseline_route], route_colors=["black"],
                                            route_linewidth=3, node_size=0, ax=ax, show=False, close=False)
                        ax.set_title("Fastest Route (no weather)")
                    else:
                        
                        n_plots = len(selected_weights)
                        n_cols = min(2, n_plots)  # Max 2 columns
                        n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division
                        
                        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 10 * n_rows))
                        if n_plots == 1:
                            axes = [axes]  # Make it iterable
                        else:
                            axes = axes.flatten()
                        
                        plot_idx = 0
                        
                        # Plot each selected weight on separate subplot
                        if "rain" in selected_weights:
                            ax = axes[plot_idx]
                            ox.plot_graph(G, ax=ax, node_color=nc, node_size=ns, node_alpha=0.8,
                                        edge_color='gray', edge_linewidth=1, show=False, close=False)
                            try:
                                rainy_route = nx.shortest_path(G, orig_node, dest_node, weight="rain_weight")
                                ox.plot_graph_routes(G, routes=[rainy_route], route_colors=["blue"],
                                                    route_linewidth=3, node_size=0, ax=ax, show=False, close=False)
                                ax.set_title("Rain-Aware Route", fontsize=14, fontweight='bold', color='blue')
                            except (KeyError, nx.NetworkXNoPath) as e:
                                ax.text(0.5, 0.5, f"Rain route failed:\n{e}", ha="center", va="center", 
                                       transform=ax.transAxes, fontsize=12, color='red')
                            plot_idx += 1
                        
                        if "heat" in selected_weights:
                            ax = axes[plot_idx]
                            ox.plot_graph(G, ax=ax, node_color=nc, node_size=ns, node_alpha=0.8,
                                        edge_color='gray', edge_linewidth=1, show=False, close=False)
                            try:
                                heat_route = nx.shortest_path(G, orig_node, dest_node, weight="heat_weight")
                                ox.plot_graph_routes(G, routes=[heat_route], route_colors=["orange"],
                                                    route_linewidth=3, node_size=0, ax=ax, show=False, close=False)
                                ax.set_title("Heat-Aware Route", fontsize=14, fontweight='bold', color='orange')
                            except (KeyError, nx.NetworkXNoPath) as e:
                                ax.text(0.5, 0.5, f"Heat route failed:\n{e}", ha="center", va="center",
                                       transform=ax.transAxes, fontsize=12, color='red')
                            plot_idx += 1
                        
                        if "wind" in selected_weights:
                            ax = axes[plot_idx]
                            ox.plot_graph(G, ax=ax, node_color=nc, node_size=ns, node_alpha=0.8,
                                        edge_color='gray', edge_linewidth=1, show=False, close=False)
                            try:
                                wind_route = nx.shortest_path(G, orig_node, dest_node, weight="wind_weight")
                                ox.plot_graph_routes(G, routes=[wind_route], route_colors=["green"],
                                                    route_linewidth=3, node_size=0, ax=ax, show=False, close=False)
                                ax.set_title("Wind-Aware Route", fontsize=14, fontweight='bold', color='green')
                            except (KeyError, nx.NetworkXNoPath) as e:
                                ax.text(0.5, 0.5, f"Wind route failed:\n{e}", ha="center", va="center",
                                       transform=ax.transAxes, fontsize=12, color='red')
                            plot_idx += 1
                        
                        if "humidity" in selected_weights:
                            ax = axes[plot_idx]
                            ox.plot_graph(G, ax=ax, node_color=nc, node_size=ns, node_alpha=0.8,
                                        edge_color='gray', edge_linewidth=1, show=False, close=False)
                            try:
                                humidity_route = nx.shortest_path(G, orig_node, dest_node, weight="humidity_weight")
                                ox.plot_graph_routes(G, routes=[humidity_route], route_colors=["purple"],
                                                    route_linewidth=3, node_size=0, ax=ax, show=False, close=False)
                                ax.set_title("Humidity-Aware Route", fontsize=14, fontweight='bold', color='purple')
                            except (KeyError, nx.NetworkXNoPath) as e:
                                ax.text(0.5, 0.5, f"Humidity route failed:\n{e}", ha="center", va="center",
                                       transform=ax.transAxes, fontsize=12, color='red')
                            plot_idx += 1
                        
                        # Hide unused subplots
                        for idx in range(plot_idx, len(axes)):
                            axes[idx].axis('off')
                        
                        fig.tight_layout()

                lat_min, lat_max = 41.61, 42.04
                lon_min, lon_max = -88.03, -87.30
                mask = (lats >= lat_min) & (lats <= lat_max) & (lons >= lon_min) & (lons <= lon_max)
                data_masked = np.where(mask, rain_dataset, np.nan)    

                # Rain overlay
                c = ax.pcolormesh(lons, lats, data_masked, cmap="Blues", shading="auto", alpha=0.4)
                fig.colorbar(c, ax=ax, label="Rainfall (mm)")
                c.set_clim(0, 10.5)

                return fig

            except Exception as e:
                print(f"Error: {e}")
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center")
                ax.axis("off")
                return fig


    @render.plot
    def plot():
        # Before clicking, show a placeholder
        if input.generate_plot() == 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "Click 'Generate Plot' to create visualization",
                    ha="center", va="center", fontsize=14)
            ax.axis("off")
            return fig
        return generated_fig()
        