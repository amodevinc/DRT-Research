import json
import folium
from folium import plugins
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any
import branca.colormap as cm

def parse_arguments():
    parser = argparse.ArgumentParser(description='Visualize insertion-based matching process.')
    parser.add_argument('json_file', type=str, help='Path to the JSON file containing frame data')
    parser.add_argument('--output', type=str, default='insertion_visualization.html', help='Output HTML file name')
    parser.add_argument('--center', type=str, default=None, help='Map center coordinates in format "lat,lon"')
    parser.add_argument('--zoom', type=int, default=14, help='Initial zoom level')
    return parser.parse_args()

def load_data(json_file: str) -> Dict:
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def create_service_area_polygon():
    service_area = [
        [37.198957, 126.827032],
        [37.203549, 126.822082], 
        [37.206509, 126.818869],
        [37.209793, 126.817734],
        [37.216226, 126.832442],
        [37.209711, 126.833297],
        [37.210003, 126.836499],
        [37.206861, 126.838809],
        [37.201823, 126.844007],
        [37.19876, 126.828538],
        [37.198957, 126.827032]  # Close the polygon
    ]
    return service_area

def get_map_center(service_area):
    lats = [p[0] for p in service_area[:-1]]
    lons = [p[1] for p in service_area[:-1]]
    return [sum(lats) / len(lats), sum(lons) / len(lons)]

def create_base_map(center=None, zoom_start=14):
    if center is None:
        service_area = create_service_area_polygon()
        center = get_map_center(service_area)
    
    m = folium.Map(location=center, zoom_start=zoom_start, tiles='OpenStreetMap')
    
    # Add service area polygon
    service_area = create_service_area_polygon()
    folium.Polygon(
        locations=service_area,
        color='blue',
        fill=True,
        fill_color='blue',
        fill_opacity=0.1,
        popup='Service Area'
    ).add_to(m)
    
    return m

def convert_timestamp(timestamp_str):
    dt = datetime.fromisoformat(timestamp_str)
    return dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

def create_cost_colormap():
    return cm.LinearColormap(
        colors=['green', 'yellow', 'red'],
        vmin=0,
        vmax=1,
        caption='Insertion Cost'
    )

def visualize_compatible_vehicles(m: folium.Map, frame_data: Dict, fg: folium.FeatureGroup):
    """Visualize compatible vehicles and request locations"""
    # Add request origin and destination
    folium.CircleMarker(
        location=frame_data['request_origin'],
        radius=8,
        color='green',
        fill=True,
        popup='Request Origin',
        weight=2
    ).add_to(fg)
    
    folium.CircleMarker(
        location=frame_data['request_destination'],
        radius=8,
        color='red',
        fill=True,
        popup='Request Destination',
        weight=2
    ).add_to(fg)
    
    # Add vehicles
    for vehicle in frame_data['compatible_vehicles']:
        folium.CircleMarker(
            location=vehicle['current_location'],
            radius=6,
            color='blue',
            fill=True,
            popup=f"""
                Vehicle: {vehicle['vehicle_id']}<br>
                Status: {vehicle['status']}<br>
                Occupancy: {vehicle['current_occupancy']}/{vehicle['capacity']}
            """,
            weight=2
        ).add_to(fg)

def visualize_route(m: folium.Map, route_data: Dict, color: str, fg: folium.FeatureGroup, popup_prefix: str = ""):
    """Visualize a route with stops and segments"""
    if not route_data:
        return
        
    # Draw route segments
    for segment in route_data['segments']:
        folium.PolyLine(
            locations=[segment['origin'], segment['destination']],
            color=color,
            weight=2,
            opacity=0.8
        ).add_to(fg)
    
    # Add stops
    for i, stop in enumerate(route_data['stops']):
        icon_color = 'green' if stop['pickup_passengers'] else 'red' if stop['dropoff_passengers'] else 'gray'
        folium.CircleMarker(
            location=stop['location'],
            radius=5,
            color=icon_color,
            fill=True,
            popup=f"{popup_prefix}Stop {i+1}<br>Arrival: {stop['planned_arrival_time']}<br>Pickups: {stop['pickup_passengers']}<br>Dropoffs: {stop['dropoff_passengers']}",
            weight=2
        ).add_to(fg)

def visualize_insertion_option(m: folium.Map, frame_data: Dict, fg: folium.FeatureGroup, cost_colormap: cm.LinearColormap):
    """Visualize a single insertion option"""
    # Normalize cost for color mapping
    normalized_cost = min(1.0, frame_data['total_cost'])
    route_color = cost_colormap(normalized_cost)
    
    # Visualize the proposed route
    visualize_route(m, frame_data['new_route'], route_color, fg, 
                   popup_prefix=f"Cost: {frame_data['total_cost']:.2f}<br>")

def visualize_final_assignment(m: folium.Map, frame_data: Dict, fg: folium.FeatureGroup):
    """Visualize the final selected assignment"""
    # Visualize the final route
    visualize_route(m, frame_data['route'], 'green', fg, "Final ")
    
    # Add pickup and dropoff markers with timing information
    folium.CircleMarker(
        location=frame_data['pickup_stop']['location'],
        radius=8,
        color='darkgreen',
        fill=True,
        popup=f"""
            Pickup Stop<br>
            Planned Arrival: {frame_data['pickup_stop']['planned_arrival']}<br>
            Vehicle: {frame_data['vehicle_id']}
        """,
        weight=3
    ).add_to(fg)
    
    folium.CircleMarker(
        location=frame_data['dropoff_stop']['location'],
        radius=8,
        color='darkred',
        fill=True,
        popup=f"""
            Dropoff Stop<br>
            Planned Arrival: {frame_data['dropoff_stop']['planned_arrival']}
        """,
        weight=3
    ).add_to(fg)
    
    # Add computation time info
    fg.add_child(folium.Popup(
        f"""
        Final Assignment Details:<br>
        Vehicle: {frame_data['vehicle_id']}<br>
        Computation Time: {frame_data['computation_time']:.3f}s
        """,
        max_width=300
    ))

def create_visualization(frames: List[Dict], output_file: str = 'insertion_visualization.html', center=None, zoom_start=14):
    # Create base map
    m = create_base_map(center, zoom_start)
    
    # Create cost colormap
    cost_colormap = create_cost_colormap()
    m.add_child(cost_colormap)
    
    # Sort frames by timestamp
    frames.sort(key=lambda x: x['timestamp'])
    
    # Create feature groups for each frame
    frame_groups = []
    frame_group_names = []
    timestamps = []
    
    for i, frame in enumerate(frames):
        fg = folium.FeatureGroup(name=f"Frame {i+1} - {convert_timestamp(frame['timestamp'])}")
        frame_name = f"frame_{i}"
        frame_group_names.append(frame_name)
        timestamps.append(convert_timestamp(frame['timestamp']))
        
        if frame['frame_type'] == 'compatible_vehicles':
            visualize_compatible_vehicles(m, frame['data'], fg)
        elif frame['frame_type'] == 'route_evaluation_start':
            if frame['data']['current_route']:
                visualize_route(m, frame['data']['current_route'], 'blue', fg, "Current ")
        elif frame['frame_type'] == 'insertion_option':
            visualize_insertion_option(m, frame['data'], fg, cost_colormap)
        elif frame['frame_type'] == 'insertion_results':
            # Visualize best insertion if available
            if frame['data']['best_insertion']:
                best_data = frame['data']['best_insertion']
                fg.add_child(folium.Popup(
                    f"""
                    Best Insertion:<br>
                    Vehicle: {best_data['vehicle_id']}<br>
                    Total Cost: {best_data['total_cost']:.2f}<br>
                    Pickup Index: {best_data['pickup_index']}<br>
                    Dropoff Index: {best_data['dropoff_index']}<br>
                    """,
                    max_width=300
                ))
        elif frame['frame_type'] == 'final_assignment':
            visualize_final_assignment(m, frame['data'], fg)
        
        frame_groups.append(fg)
        fg.add_to(m)  # Add each feature group to the map immediately
    
    # Create a time slider for frame navigation
    slider_html = f'''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 50%; height: auto;
                background-color: white; border:2px solid grey; z-index:9999; font-size:14px;
                padding: 10px">
    <input type="range" id="timeSlider" min="0" max="{len(frame_groups) - 1}" value="0" style="width:100%">
    <div id="timeLabel"></div>
    <div id="frameInfo"></div>
    </div>
    
    <script>
        // Store map reference
        var mymap = document.querySelector('#map');
        
        // Store feature groups
        var featureGroups = {{
            {",".join([f'"{name}": {name}' for name in frame_group_names])}
        }};
        
        var timestamps = {json.dumps(timestamps)};
        var frameTypes = {json.dumps([frame['frame_type'] for frame in frames])};
        
        function showFrame(frameIndex) {{
            // Hide all frames
            Object.values(featureGroups).forEach(function(group) {{
                group.remove();
            }});
            
            // Show the selected frame
            featureGroups[`frame_${{frameIndex}}`].addTo(mymap);
            
            // Update the time label and frame info
            document.getElementById('timeLabel').innerHTML = 'Time: ' + timestamps[frameIndex];
            document.getElementById('frameInfo').innerHTML = 'Stage: ' + frameTypes[frameIndex].replace(/_/g, ' ');
        }}
        
        // Set up slider event
        var slider = document.getElementById('timeSlider');
        slider.max = {len(frame_groups) - 1};
        slider.addEventListener('input', function() {{
            showFrame(parseInt(this.value));
        }});
        
        // Initialize with the first frame when the map is ready
        window.addEventListener('load', function() {{
            setTimeout(function() {{
                showFrame(0);
            }}, 100);
        }});
    </script>
    '''
    
    # Add each feature group to the map with a unique variable name
    for i, fg in enumerate(frame_groups):
        m.get_root().script.add_child(folium.Element(f"var frame_{i} = {fg.get_name()};"))
    
    # Add the slider
    m.get_root().html.add_child(folium.Element(slider_html))
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Save the map
    m.save(output_file)
    print(f"Visualization saved to {output_file}")

def main():
    args = parse_arguments()
    
    try:
        # Load data
        data = load_data(args.json_file)
        if 'frames' not in data:
            print("Error: No frames found in the data")
            return
            
        frames = data['frames']
        if not frames:
            print("Error: Empty frames list")
            return
            
        print(f"Loaded {len(frames)} frames")
        print("Frame types:", set(frame['frame_type'] for frame in frames))
            
        # Set map center if provided
        center = None
        if args.center:
            try:
                lat, lon = map(float, args.center.split(','))
                center = [lat, lon]
            except ValueError:
                print("Error: Invalid center coordinates format. Using default.")
        
        # Create visualization
        create_visualization(frames, output_file=args.output, center=center, zoom_start=args.zoom)
        
    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 