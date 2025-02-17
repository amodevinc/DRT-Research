import json
import folium
import sys
import os
from datetime import datetime
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Visualize virtual stop selections over time.')
    parser.add_argument('json_file', type=str, help='Path to the JSON file containing frame data')
    parser.add_argument('--output', type=str, default='visualization.html', help='Output HTML file name')
    parser.add_argument('--center', type=str, default=None, help='Map center coordinates in format "lat,lon"')
    parser.add_argument('--zoom', type=int, default=14, help='Initial zoom level')
    return parser.parse_args()

def load_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data['frames']

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
    lats = [p[0] for p in service_area[:-1]]  # Exclude the last point which is a repeat
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

def get_unique_request_ids(frames):
    request_ids = set()
    for frame in frames:
        if 'metadata' in frame and 'request_id' in frame['metadata']:
            request_ids.add(frame['metadata']['request_id'])
    return sorted(list(request_ids))

def assign_colors_to_requests(request_ids):
    # Use a colormap to assign distinct colors to each request
    colors = {}
    cmap = plt.cm.get_cmap('tab10', len(request_ids))
    for i, req_id in enumerate(request_ids):
        rgba = cmap(i)
        hex_color = '#%02x%02x%02x' % (int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
        colors[req_id] = hex_color
    return colors

def create_visualization(frames, output_file='visualization.html', center=None, zoom_start=14):
    # Get unique request IDs and assign colors
    request_ids = get_unique_request_ids(frames)
    request_colors = assign_colors_to_requests(request_ids)
    
    # Create base map
    m = create_base_map(center, zoom_start)
    
    # Sort frames by timestamp
    frames.sort(key=lambda x: x['timestamp'])
    
    # Create a feature group for each frame
    frame_groups = []
    frame_group_names = []
    timestamps = []
    
    for i, frame in enumerate(frames):
        fg = folium.FeatureGroup(name=f"Frame {i+1} - {convert_timestamp(frame['timestamp'])}")
        frame_name = f"frame_{i}"
        frame_group_names.append(frame_name)
        timestamps.append(convert_timestamp(frame['timestamp']))
        
        if frame['frame_type'] == 'virtual_stop_candidates':
            request_id = frame['metadata']['request_id']
            color = request_colors[request_id]
            
            # Add request origin and destination
            folium.CircleMarker(
                location=[frame['data']['request_origin'][0], frame['data']['request_origin'][1]],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                popup=f"Request {request_id} Origin",
            ).add_to(fg)
            
            folium.CircleMarker(
                location=[frame['data']['request_destination'][0], frame['data']['request_destination'][1]],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                popup=f"Request {request_id} Destination",
            ).add_to(fg)
            
            # Add candidate stops for origin
            for j, candidate in enumerate(frame['data']['origin_candidates']):
                folium.CircleMarker(
                    location=[candidate[0], candidate[1]],
                    radius=3,
                    color=color,
                    fill=False,
                    popup=f"Request {request_id} Origin Candidate {j+1}",
                ).add_to(fg)
            
            # Add candidate stops for destination
            for j, candidate in enumerate(frame['data']['destination_candidates']):
                folium.CircleMarker(
                    location=[candidate[0], candidate[1]],
                    radius=3,
                    color=color,
                    fill=False,
                    popup=f"Request {request_id} Destination Candidate {j+1}",
                ).add_to(fg)
            
            # Add nearby stops
            for j, stop in enumerate(frame['data']['nearby_origin_stops']):
                folium.CircleMarker(
                    location=[stop[0], stop[1]],
                    radius=4,
                    color='gray',
                    fill=True,
                    fill_color='gray',
                    popup=f"Nearby Origin Stop {j+1}",
                ).add_to(fg)
            
            for j, stop in enumerate(frame['data']['nearby_dest_stops']):
                folium.CircleMarker(
                    location=[stop[0], stop[1]],
                    radius=4,
                    color='gray',
                    fill=True,
                    fill_color='gray',
                    popup=f"Nearby Destination Stop {j+1}",
                ).add_to(fg)
        
        elif frame['frame_type'] == 'virtual_stop_selection':
            request_id = frame['metadata']['request_id']
            color = request_colors[request_id]
            
            # Add selected origin and destination stops with bold highlighting
            folium.CircleMarker(
                location=[frame['data']['selected_origin'][0], frame['data']['selected_origin'][1]],
                radius=7,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=f"Selected Origin for Request {request_id}",
            ).add_to(fg)
            
            folium.CircleMarker(
                location=[frame['data']['selected_destination'][0], frame['data']['selected_destination'][1]],
                radius=7,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=f"Selected Destination for Request {request_id}",
            ).add_to(fg)
            
            # Draw a line connecting selected origin and destination
            folium.PolyLine(
                locations=[
                    [frame['data']['selected_origin'][0], frame['data']['selected_origin'][1]],
                    [frame['data']['selected_destination'][0], frame['data']['selected_destination'][1]]
                ],
                color=color,
                weight=2,
                opacity=0.7,
                popup=f"Route for Request {request_id}",
            ).add_to(fg)
        
        frame_groups.append(fg)
    
    # Add a legend for request IDs
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 150px; height: auto; 
                background-color: white; border:2px solid grey; z-index:9999; font-size:14px;
                padding: 10px">
    <b>Request IDs</b><br>
    '''
    
    for req_id, color in request_colors.items():
        legend_html += f'<span style="background-color: {color}; color: {color}; border: 1px solid black;">___</span> {req_id}<br>'
    
    legend_html += '''
    </div>
    '''
    
    # Create a time slider for frame navigation - Fixed the string formatting issue
    slider_html = f'''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 50%; height: auto;
                background-color: white; border:2px solid grey; z-index:9999; font-size:14px;
                padding: 10px">
    <input type="range" id="timeSlider" min="0" max="{len(frame_groups) - 1}" value="0" style="width:100%">
    <div id="timeLabel"></div>
    </div>
    
    <script>
        var frameGroups = {json.dumps(frame_group_names)};
        var timestamps = {json.dumps(timestamps)};
        
        function showFrame(frameIndex) {{
            // Hide all frames
            for (var i = 0; i < frameGroups.length; i++) {{
                map.removeLayer(window[frameGroups[i]]);
            }}
            
            // Show the selected frame
            map.addLayer(window[frameGroups[frameIndex]]);
            
            // Update the time label
            document.getElementById('timeLabel').innerHTML = 'Time: ' + timestamps[frameIndex];
        }}
        
        // Set up slider event
        document.getElementById('timeSlider').max = frameGroups.length - 1;
        document.getElementById('timeSlider').addEventListener('input', function() {{
            showFrame(parseInt(this.value));
        }});
        
        // Initialize with the first frame
        showFrame(0);
    </script>
    '''
    
    # Add each feature group to the map with a unique variable name
    for i, fg in enumerate(frame_groups):
        fg.add_to(m)
        m.get_root().script.add_child(folium.Element(f"var frame_{i} = {fg.get_name()};"))
    
    # Add the legend and slider to the map
    m.get_root().html.add_child(folium.Element(legend_html))
    m.get_root().html.add_child(folium.Element(slider_html))
    
    # Add layer control to the map
    folium.LayerControl().add_to(m)
    
    # Save the map to the output file
    m.save(output_file)
    print(f"Visualization saved to {output_file}")
    
    return m

def main():
    args = parse_arguments()
    
    # Load data from JSON file
    frames = load_data(args.json_file)
    
    # Set map center if provided
    center = None
    if args.center:
        lat, lon = map(float, args.center.split(','))
        center = [lat, lon]
    
    # Create visualization
    create_visualization(frames, output_file=args.output, center=center, zoom_start=args.zoom)

if __name__ == "__main__":
    main()