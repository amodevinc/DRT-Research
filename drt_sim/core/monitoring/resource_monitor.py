'''Under Construction'''
from __future__ import annotations
from typing import Dict, List, Optional, Any
import threading
import time
import psutil
import logging
from datetime import datetime
from dataclasses import dataclass
from statistics import mean
from drt_sim.core.logging_config import setup_logger
@dataclass
class ResourceSnapshot:
    """Snapshot of resource usage at a point in time"""
    timestamp: datetime
    cpu_percent: float
    memory_used: float  # MB
    memory_percent: float
    disk_io_read: float  # MB/s
    disk_io_write: float  # MB/s
    thread_count: int

class ResourceMonitor:
    """Monitors system resource usage during simulation"""
    
    def __init__(self, sampling_interval: float = 1.0):
        self.logger = setup_logger(self.__class__.__name__)
        self.sampling_interval = sampling_interval
        self.process = psutil.Process()
        
        # Monitoring state
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._snapshots: List[ResourceSnapshot] = []
        
        # Peak values
        self.peak_cpu_percent: float = 0.0
        self.peak_memory_used: float = 0.0
        self.peak_thread_count: int = 0
        
        # Previous values for rate calculations
        self._prev_disk_read = 0
        self._prev_disk_write = 0
        self._prev_time = time.time()
    
    def start(self) -> None:
        """Start resource monitoring"""
        if self._monitoring:
            return
            
        self.logger.info("Starting resource monitoring")
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop(self) -> None:
        """Stop resource monitoring"""
        if not self._monitoring:
            return
            
        self.logger.info("Stopping resource monitoring")
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop"""
        while self._monitoring:
            try:
                snapshot = self._take_snapshot()
                self._snapshots.append(snapshot)
                self._update_peaks(snapshot)
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                self.logger.error(f"Error in resource monitoring: {str(e)}")
                continue
    
    def _take_snapshot(self) -> ResourceSnapshot:
        """Take a snapshot of current resource usage"""
        current_time = datetime.now()
        
        # Get CPU usage
        cpu_percent = self.process.cpu_percent()
        
        # Get memory usage
        memory_info = self.process.memory_info()
        memory_used = memory_info.rss / (1024 * 1024)  # Convert to MB
        memory_percent = self.process.memory_percent()
        
        # Get disk I/O rates - use psutil directly since Process doesn't have io_counters
        try:
            disk_io = psutil.disk_io_counters()
            current_time_s = time.time()
            time_delta = current_time_s - self._prev_time
            
            if time_delta > 0:
                disk_read_rate = (disk_io.read_bytes - self._prev_disk_read) / (1024 * 1024 * time_delta)  # MB/s
                disk_write_rate = (disk_io.write_bytes - self._prev_disk_write) / (1024 * 1024 * time_delta)  # MB/s
            else:
                disk_read_rate = 0
                disk_write_rate = 0
            
            # Update previous values
            self._prev_disk_read = disk_io.read_bytes
            self._prev_disk_write = disk_io.write_bytes
            self._prev_time = current_time_s
            
        except (AttributeError, OSError):
            # Handle case where disk I/O monitoring is not available
            disk_read_rate = 0
            disk_write_rate = 0
        
        # Get thread count
        thread_count = self.process.num_threads()
        
        return ResourceSnapshot(
            timestamp=current_time,
            cpu_percent=cpu_percent,
            memory_used=memory_used,
            memory_percent=memory_percent,
            disk_io_read=disk_read_rate,
            disk_io_write=disk_write_rate,
            thread_count=thread_count
        )
    
    def _update_peaks(self, snapshot: ResourceSnapshot) -> None:
        """Update peak values"""
        self.peak_cpu_percent = max(self.peak_cpu_percent, snapshot.cpu_percent)
        self.peak_memory_used = max(self.peak_memory_used, snapshot.memory_used)
        self.peak_thread_count = max(self.peak_thread_count, snapshot.thread_count)
    
    def get_current_usage(self) -> Dict[str, float]:
        """Get current resource usage"""
        if not self._snapshots:
            return {
                'cpu_percent': 0.0,
                'memory_used_mb': 0.0,
                'memory_percent': 0.0,
                'disk_read_mbs': 0.0,
                'disk_write_mbs': 0.0,
                'thread_count': 0
            }
            
        latest = self._snapshots[-1]
        return {
            'cpu_percent': latest.cpu_percent,
            'memory_used_mb': latest.memory_used,
            'memory_percent': latest.memory_percent,
            'disk_read_mbs': latest.disk_io_read,
            'disk_write_mbs': latest.disk_io_write,
            'thread_count': latest.thread_count
        }
    
    def get_average_usage(self) -> Dict[str, float]:
        """Get average resource usage"""
        if not self._snapshots:
            return {
                'avg_cpu_percent': 0.0,
                'avg_memory_used_mb': 0.0,
                'avg_memory_percent': 0.0,
                'avg_disk_read_mbs': 0.0,
                'avg_disk_write_mbs': 0.0,
                'avg_thread_count': 0
            }
            
        return {
            'avg_cpu_percent': mean(s.cpu_percent for s in self._snapshots),
            'avg_memory_used_mb': mean(s.memory_used for s in self._snapshots),
            'avg_memory_percent': mean(s.memory_percent for s in self._snapshots),
            'avg_disk_read_mbs': mean(s.disk_io_read for s in self._snapshots),
            'avg_disk_write_mbs': mean(s.disk_io_write for s in self._snapshots),
            'avg_thread_count': mean(s.thread_count for s in self._snapshots)
        }
    
    def get_peak_usage(self) -> Dict[str, float]:
        """Get peak resource usage"""
        return {
            'peak_cpu_percent': self.peak_cpu_percent,
            'peak_memory_used_mb': self.peak_memory_used,
            'peak_thread_count': self.peak_thread_count
        }
    
    def get_usage_history(self) -> List[Dict[str, Any]]:
        """Get complete usage history"""
        return [
            {
                'timestamp': s.timestamp.isoformat(),
                'cpu_percent': s.cpu_percent,
                'memory_used_mb': s.memory_used,
                'memory_percent': s.memory_percent,
                'disk_read_mbs': s.disk_io_read,
                'disk_write_mbs': s.disk_io_write,
                'thread_count': s.thread_count
            }
            for s in self._snapshots
        ]
    
    def cleanup(self) -> None:
        """Clean up monitoring resources"""
        self.stop()
        self._snapshots.clear()
        self.peak_cpu_percent = 0.0
        self.peak_memory_used = 0.0
        self.peak_thread_count = 0