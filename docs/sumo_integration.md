# SUMO Integration for DRT Research Platform

This document describes how to use the SUMO (Simulation of Urban MObility) integration with the DRT Research Platform.

## Overview

The SUMO integration allows you to:

1. Visualize DRT simulations in the SUMO GUI
2. Use realistic traffic simulation for vehicle movements
3. Convert existing network files to SUMO format
4. Analyze traffic patterns and congestion

## Prerequisites

Before using the SUMO integration, you need to install SUMO:

### Installing SUMO

#### On Linux:

```bash
sudo apt-get install sumo sumo-tools sumo-doc
```

#### On macOS:

```bash
brew install sumo
```

#### On Windows:

Download the installer from the [SUMO website](https://sumo.dlr.de/docs/Downloads.php) and follow the installation instructions.

### Setting up the SUMO environment

After installing SUMO, you need to set the `SUMO_HOME` environment variable:

#### On Linux/macOS:

```bash
export SUMO_HOME=/path/to/sumo
```

Add this line to your `.bashrc` or `.zshrc` file to make it permanent.

#### On Windows:

```
set SUMO_HOME=C:\path\to\sumo
```

Add this to your environment variables to make it permanent.

## Configuration

To enable SUMO integration, add the following to your study configuration YAML file:

```yaml
simulation:
  # ... other simulation settings ...
  sumo:
    enabled: true
    sumo_binary: "sumo-gui"  # Use "sumo" for headless mode
    step_length: 1.0
    use_geo_coordinates: true
    visualization: true
    seed: 42
```

### Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `enabled` | Enable SUMO integration | `false` |
| `sumo_binary` | SUMO binary to use ("sumo" or "sumo-gui") | `"sumo-gui"` |
| `network_file` | Path to SUMO network file (.net.xml) | `null` |
| `route_file` | Path to SUMO route file (.rou.xml) | `null` |
| `additional_files` | List of additional SUMO files | `[]` |
| `gui_settings_file` | Path to SUMO GUI settings file | `null` |
| `step_length` | SUMO simulation step length in seconds | `1.0` |
| `begin_time` | SUMO simulation begin time in seconds | `0.0` |
| `end_time` | SUMO simulation end time in seconds | `86400.0` |
| `use_geo_coordinates` | Whether to use geo-coordinates | `true` |
| `port` | TraCI port for SUMO connection | `8813` |
| `seed` | Random seed for SUMO | `42` |
| `auto_convert_network` | Whether to automatically convert network | `true` |
| `visualization` | Whether to enable visualization | `true` |

## Converting Networks to SUMO Format

You can convert existing network files to SUMO format using the provided conversion script:

```bash
python scripts/convert_network_to_sumo.py --input data/networks/my_network.graphml --output data/networks/sumo
```

This will create a SUMO network file (.net.xml) that can be used for simulation.

### Supported Input Formats

- GraphML (.graphml)
- GeoJSON (.geojson, .json)
- OpenStreetMap (.osm, .xml)

## Running Simulations with SUMO

To run a simulation with SUMO visualization:

```bash
python scripts/run_simulation.py sumo_integration_example --parameter-sets small_fleet
```

This will start the simulation and open the SUMO GUI for visualization.

## Customizing SUMO Visualization

You can customize the SUMO visualization by creating a GUI settings file:

```xml
<viewsettings>
    <scheme name="real world"/>
    <delay value="100"/>
    <viewport zoom="100" x="0" y="0"/>
    <decal file="background.gif" centerX="550" centerY="550" width="1000" height="1000" rotation="0.00"/>
    <scheme name="custom">
        <opengl antialiase="1" dither="1"/>
        <background backgroundColor="black" showGrid="0" gridXSize="100.00" gridYSize="100.00"/>
        <vehicles vehicleMode="9" vehicleQuality="2" minVehicleSize="1.00" vehicleExaggeration="1.00" showBlinker="1"
                 vehicleName_show="0" vehicleName_size="50.00" vehicleName_color="0,0,0" />
        <edges laneEdgeMode="0" scaleMode="0" laneShowBorders="1" showLinkDecals="1" showRails="1" hideConnectors="0"
              edgeName_show="0" edgeName_size="50.00" edgeName_color="orange" />
    </scheme>
</viewsettings>
```

Save this file as `gui-settings.xml` and specify it in your configuration:

```yaml
simulation:
  sumo:
    gui_settings_file: "path/to/gui-settings.xml"
```

## Troubleshooting

### SUMO not found

If you get an error like "SUMO Python modules not found", make sure:

1. SUMO is installed correctly
2. The `SUMO_HOME` environment variable is set
3. The SUMO tools directory is in your Python path

### Connection errors

If you get connection errors when starting SUMO:

1. Make sure no other SUMO instance is running
2. Try changing the port in the configuration
3. Check if there are firewall issues

### Visualization issues

If the visualization doesn't look right:

1. Check if the coordinate systems match between your network and SUMO
2. Try setting `use_geo_coordinates: false` if your network uses projected coordinates
3. Adjust the GUI settings file

## Advanced Usage

### Using SUMO Traffic Information

You can retrieve traffic information from SUMO during simulation:

```python
# Get traffic info for an edge
traffic_info = await sumo_integration.get_traffic_info("edge_id")
print(f"Mean speed: {traffic_info['mean_speed']} m/s")
print(f"Travel time: {traffic_info['travel_time']} s")
print(f"Vehicle count: {traffic_info['vehicle_count']}")
```

### Adding Background Traffic

You can add background traffic to make the simulation more realistic:

1. Create a route file with background traffic:

```xml
<routes>
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5.0" minGap="2.5" maxSpeed="13.89" color="1,1,0"/>
    <flow id="flow_0" type="car" begin="0" end="3600" number="100" from="edge_1" to="edge_2"/>
</routes>
```

2. Specify the route file in your configuration:

```yaml
simulation:
  sumo:
    route_file: "path/to/background_traffic.rou.xml"
```

## References

- [SUMO Documentation](https://sumo.dlr.de/docs/)
- [TraCI (Traffic Control Interface) Documentation](https://sumo.dlr.de/docs/TraCI.html)
- [SUMO Network Generation](https://sumo.dlr.de/docs/Networks/SUMO_Road_Networks.html) 