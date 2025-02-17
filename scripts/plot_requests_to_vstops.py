import json
import folium

# Load JSON file
with open('scripts/data/test.json', 'r') as f:
    data = json.load(f)

# Filter out only the "request.assigned" events
assigned_requests = [item for item in data if item.get("metric_name") == "request.assigned"]

# Define service area polygon points (provided as [lon, lat])
service_area_polygon = [
    [126.827032, 37.198957],
    [126.822082, 37.203549],
    [126.818869, 37.206509],
    [126.817734, 37.209793],
    [126.832442, 37.216226],
    [126.833297, 37.209711],
    [126.836499, 37.210003],
    [126.838809, 37.206861],
    [126.844007, 37.201823],
    [126.828538, 37.198760]
]

# Folium expects coordinates as [lat, lon]. Convert each polygon point.
converted_polygon = [[point[1], point[0]] for point in service_area_polygon]

# Use the first point of the polygon as the map center.
center_lat, center_lon = converted_polygon[0]
m = folium.Map(location=[center_lat, center_lon], zoom_start=14)

# Add the service area polygon to the map
folium.Polygon(
    locations=converted_polygon,
    color='blue',
    fill=True,
    fill_opacity=0.1,
    popup="Service Area"
).add_to(m)

# Define lists of colors for stops (so that each request gets a different shade)
origin_stop_colors = ["darkred", "lightred", "orange", "firebrick", "indianred"]
destination_stop_colors = ["darkblue", "lightblue", "cadetblue", "dodgerblue", "steelblue"]

# Process each assigned request (using enumerate to cycle through colors)
for idx, req in enumerate(assigned_requests):
    request_id = req.get("request_id")
    
    # Extract coordinate dictionaries
    req_origin = req.get("request_origin")
    req_destination = req.get("request_destination")
    origin_stop = req.get("origin_stop")
    destination_stop = req.get("destination_stop")
    
    # Select colors for this request's stops
    origin_stop_color = origin_stop_colors[idx % len(origin_stop_colors)]
    destination_stop_color = destination_stop_colors[idx % len(destination_stop_colors)]
    
    # Add marker for request origin (always red)
    if req_origin:
        folium.Marker(
            location=[req_origin["lat"], req_origin["lon"]],
            popup=f"Request {request_id} Origin",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)
    
    # Add marker for origin stop (a shade of red)
    if origin_stop:
        folium.Marker(
            location=[origin_stop["lat"], origin_stop["lon"]],
            popup=f"Request {request_id} Origin Stop",
            icon=folium.Icon(color=origin_stop_color, icon='info-sign')
        ).add_to(m)
    
    # Add marker for request destination (always blue)
    if req_destination:
        folium.Marker(
            location=[req_destination["lat"], req_destination["lon"]],
            popup=f"Request {request_id} Destination",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(m)
    
    # Add marker for destination stop (a shade of blue)
    if destination_stop:
        folium.Marker(
            location=[destination_stop["lat"], destination_stop["lon"]],
            popup=f"Request {request_id} Destination Stop",
            icon=folium.Icon(color=destination_stop_color, icon='info-sign')
        ).add_to(m)
    
    # Create a polyline connecting all four points for this request.
    # The order is: request origin -> origin stop -> destination stop -> request destination.
    polyline_points = []
    if req_origin:
        polyline_points.append([req_origin["lat"], req_origin["lon"]])
    if origin_stop:
        polyline_points.append([origin_stop["lat"], origin_stop["lon"]])
    if destination_stop:
        polyline_points.append([destination_stop["lat"], destination_stop["lon"]])
    if req_destination:
        polyline_points.append([req_destination["lat"], req_destination["lon"]])
    
    if polyline_points:
        folium.PolyLine(
            locations=polyline_points,
            color='black',
            weight=2,
            opacity=0.8,
            popup=f"Request {request_id} route"
        ).add_to(m)

# Save the map to an HTML file
m.save("scripts/output/map.html")
print("Map has been saved to map.html")
