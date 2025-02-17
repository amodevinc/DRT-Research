from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import logging
from pathlib import Path
import numpy as np

from drt_sim.models.base import SimulationEncoder

logger = logging.getLogger(__name__)

@dataclass
class VisualizationFrame:
    """A single frame of visualization data"""
    timestamp: datetime
    component_id: str
    module_id: str
    frame_type: str
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    sequence_number: Optional[int] = None  # Track sequence of related frames
    parent_frame_id: Optional[str] = None  # Link related frames
    description: Optional[str] = None      # Human readable description

@dataclass
class ComponentBuffer:
    """Buffer for a specific component's visualization data"""
    frames: List[VisualizationFrame] = field(default_factory=list)
    max_buffer_size: int = 1000
    frame_counter: int = field(default=0)  # Track frame sequence
    
    def add_frame(self, frame: VisualizationFrame) -> None:
        """Add a frame to the buffer, maintaining size limit"""
        self.frame_counter += 1
        frame.sequence_number = self.frame_counter
        self.frames.append(frame)
        if len(self.frames) > self.max_buffer_size:
            self._flush_to_disk()
            
    def _flush_to_disk(self) -> None:
        """Placeholder for flushing data to disk when buffer is full"""
        # This will be implemented when we add disk storage
        self.frames = []

class VisualizationManager:
    """
    Manages collection and storage of visualization data from various simulation components.
    Provides interfaces for components to store their visualization data and handles
    efficient storage and retrieval.
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize the visualization manager.
        
        Args:
            output_dir: Directory to store visualization data
        """
        self.output_dir = output_dir
        self.component_buffers: Dict[str, ComponentBuffer] = {}
        self.metadata: Dict[str, Any] = {}
        
        # Create visualization directory
        self.viz_dir = output_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Track registered components
        self.registered_components: Dict[str, Dict[str, Any]] = {}
        
    def register_component(self, 
                         component_id: str,
                         component_type: str,
                         modules: List[str],
                         metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a component for visualization data collection.
        
        Args:
            component_id: Unique identifier for the component
            component_type: Type of component (e.g., 'stop_handler', 'vehicle_handler')
            modules: List of module IDs that will provide visualization data
            metadata: Optional metadata about the component
        """
        if component_id in self.registered_components:
            logger.warning(f"Component {component_id} already registered")
            return
            
        self.registered_components[component_id] = {
            'type': component_type,
            'modules': modules,
            'metadata': metadata or {}
        }
        
        # Initialize buffer for this component
        self.component_buffers[component_id] = ComponentBuffer()
        
        logger.info(f"Registered component {component_id} for visualization")
        
    def add_frame(self,
                 component_id: str,
                 module_id: str,
                 frame_type: str,
                 data: Dict[str, Any],
                 timestamp: Optional[datetime] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 parent_frame_id: Optional[str] = None,
                 description: Optional[str] = None) -> None:
        """
        Add a visualization frame for a component.
        
        Args:
            component_id: ID of the component adding the frame
            module_id: ID of the specific module (e.g., 'stop_selector', 'stop_assigner')
            frame_type: Type of visualization frame (e.g., 'cluster_update', 'stop_selection')
            data: The visualization data
            timestamp: Optional timestamp for the frame
            metadata: Optional metadata for the frame
            parent_frame_id: Optional ID of a related parent frame
            description: Optional human-readable description of what this frame represents
        """
        if component_id not in self.registered_components:
            logger.warning(f"Component {component_id} not registered for visualization")
            return
            
        if module_id not in self.registered_components[component_id]['modules']:
            logger.warning(f"Module {module_id} not registered for component {component_id}")
            return
            
        # Enhance metadata with context
        enhanced_metadata = metadata or {}
        if description:
            enhanced_metadata['description'] = description
            
        frame = VisualizationFrame(
            timestamp=timestamp or datetime.now(),
            component_id=component_id,
            module_id=module_id,
            frame_type=frame_type,
            data=data,
            metadata=enhanced_metadata,
            parent_frame_id=parent_frame_id,
            description=description
        )
        
        self.component_buffers[component_id].add_frame(frame)
        
    def flush_component(self, component_id: str) -> None:
        """
        Flush visualization data for a specific component to disk.
        
        Args:
            component_id: ID of component to flush
        """
        if component_id not in self.component_buffers:
            return
            
        buffer = self.component_buffers[component_id]
        if not buffer.frames:
            return
            
        # Create component directory
        component_dir = self.viz_dir / component_id
        component_dir.mkdir(exist_ok=True)
        
        # Group frames by module
        frames_by_module = {}
        for frame in buffer.frames:
            if frame.module_id not in frames_by_module:
                frames_by_module[frame.module_id] = []
            frames_by_module[frame.module_id].append(frame)
            
        # Save each module's data
        for module_id, frames in frames_by_module.items():
            module_data = {
                'component_id': component_id,
                'module_id': module_id,
                'component_type': self.registered_components[component_id]['type'],
                'frames': [
                    {
                        'timestamp': frame.timestamp.isoformat(),
                        'frame_type': frame.frame_type,
                        'data': frame.data,
                        'metadata': frame.metadata
                    }
                    for frame in frames
                ]
            }
            
            # Save to file
            output_file = component_dir / f"{module_id}_visualization.json"
            with open(output_file, 'w') as f:
                json.dump(module_data, f, cls=SimulationEncoder, indent=2)
                
        # Clear buffer
        buffer.frames = []
        
    def cleanup(self) -> None:
        """Clean up and save all remaining visualization data."""
        try:
            # Flush all component buffers
            for component_id in list(self.component_buffers.keys()):
                self.flush_component(component_id)
                
            # Save component registry
            registry_file = self.viz_dir / "component_registry.json"
            with open(registry_file, 'w') as f:
                json.dump(self.registered_components, f, cls=SimulationEncoder, indent=2)
                
            logger.info("Successfully saved all visualization data")
            
        except Exception as e:
            logger.error(f"Error during visualization cleanup: {str(e)}")
            raise 