from drt_sim.models.base import SimulationEncoder
from drt_sim.core.monitoring.types.metrics import MetricPoint
import json
import pandas as pd
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union, Iterator
import queue
import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

logger = logging.getLogger(__name__)

class MetricsStorage:
    """Handles persistent storage of metrics with efficient chunking."""
    
    def __init__(self, 
                 base_path: Union[str, Path],
                 replication_id: str,
                 chunk_size: int = 1000,
                 max_queue_size: int = 5000,
                 compaction_threshold: int = 10):  # Number of chunks before triggering compaction
        self.base_path = Path(base_path)
        self.replication_id = replication_id
        self.chunk_size = chunk_size
        self.current_chunk: List[MetricPoint] = []
        self.chunk_count = 0
        self.metrics_queue = queue.Queue(maxsize=max_queue_size)
        self.lock = Lock()
        self.shutdown = False
        self.compaction_threshold = compaction_threshold
        
        # Create directory structure with subdirectories for better organization
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.compacted_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Start background worker
        self.worker = ThreadPoolExecutor(max_workers=1)
        self.worker.submit(self._process_queue)
        
    @property
    def metrics_dir(self) -> Path:
        # Create hierarchical directory structure based on date and replication
        date_str = datetime.now().strftime("%Y-%m-%d")
        return self.base_path / "metrics" / date_str / self.replication_id
        
    @property
    def temp_dir(self) -> Path:
        return self.base_path / "temp" / self.replication_id
        
    @property
    def compacted_dir(self) -> Path:
        return self.base_path / "compacted" / self.replication_id
    
    @property
    def archive_dir(self) -> Path:
        return self.base_path / "archive" / self.replication_id
    
    def _get_chunk_path(self, chunk_num: int) -> Path:
        return self.metrics_dir / f"chunk_{chunk_num:06d}.json"
        
    def _get_compacted_path(self, start_chunk: int, end_chunk: int) -> Path:
        return self.compacted_dir / f"compacted_{start_chunk:06d}_{end_chunk:06d}.json"
    
    def add_metric(self, metric: MetricPoint):
        """Add a metric to the queue for processing."""
        try:
            self.metrics_queue.put(metric, timeout=1)
        except queue.Full:
            logging.warning("Metrics queue full, processing immediately")
            self._process_metric(metric)
    
    def _process_queue(self):
        """Background worker to process metrics queue."""
        while not self.shutdown:
            try:
                metric = self.metrics_queue.get(timeout=1)
                self._process_metric(metric)
                self.metrics_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error processing metric: {e}")
    
    def _process_metric(self, metric: MetricPoint):
        """Process a single metric point."""
        with self.lock:
            self.current_chunk.append(metric)
            
            if len(self.current_chunk) >= self.chunk_size:
                self._write_chunk()
    
    def _write_chunk(self):
        """Write current chunk to disk and trigger compaction if needed."""
        if not self.current_chunk:
            return
            
        chunk_path = self._get_chunk_path(self.chunk_count)
        temp_path = self.temp_dir / f"temp_chunk_{self.chunk_count:06d}.json"
        
        # Write to temporary file first
        with open(temp_path, 'w') as f:
            json.dump([m.dict() for m in self.current_chunk], f, cls=SimulationEncoder)
        
        # Atomic move to final location
        temp_path.rename(chunk_path)
        
        self.chunk_count += 1
        self.current_chunk = []
        
        # Check if compaction is needed
        if self.chunk_count % self.compaction_threshold == 0:
            self._compact_chunks()
            
    def _compact_chunks(self):
        """Merge multiple small chunk files into a larger compacted file."""
        try:
            # Find chunks to compact
            chunk_files = sorted(self.metrics_dir.glob("chunk_*.json"))
            if len(chunk_files) < self.compaction_threshold:
                return
                
            # Read and merge chunks
            merged_metrics = []
            start_chunk = None
            end_chunk = None
            
            for chunk_file in chunk_files[:self.compaction_threshold]:
                chunk_num = int(chunk_file.stem.split('_')[1])
                if start_chunk is None:
                    start_chunk = chunk_num
                end_chunk = chunk_num
                
                with open(chunk_file, 'r') as f:
                    chunk_data = json.load(f)
                    merged_metrics.extend(chunk_data)
            
            if not merged_metrics:
                return
                
            # Write compacted file
            compacted_path = self._get_compacted_path(start_chunk, end_chunk)
            temp_compacted = self.temp_dir / f"temp_compacted_{start_chunk:06d}_{end_chunk:06d}.json"
            
            with open(temp_compacted, 'w') as f:
                json.dump(merged_metrics, f, cls=SimulationEncoder)
                
            # Atomic move
            temp_compacted.rename(compacted_path)
            
            # Remove original chunks
            for chunk_file in chunk_files[:self.compaction_threshold]:
                chunk_file.unlink()
                
            logger.info(f"Compacted chunks {start_chunk}-{end_chunk} into {compacted_path}")
            
        except Exception as e:
            logger.error(f"Error during chunk compaction: {e}")
            
    def flush(self):
        """Flush any remaining metrics to disk."""
        with self.lock:
            self._write_chunk()
    
    def shutdown_storage(self):
        """Shutdown the storage system."""
        self.shutdown = True
        self.worker.shutdown(wait=True)
        self.flush()
        
    def iter_chunks(self) -> Iterator[List[MetricPoint]]:
        """Iterate through all stored chunks, including compacted files."""
        self.flush()  # Ensure all metrics are written
        
        # First read compacted files
        compacted_files = sorted(self.compacted_dir.glob("compacted_*.json"))
        for compacted_file in compacted_files:
            if compacted_file.exists():
                with open(compacted_file, 'r') as f:
                    data = json.load(f)
                    metrics = []
                    for d in data:
                        # Ensure timestamp is datetime
                        if isinstance(d.get('timestamp'), str):
                            d['timestamp'] = datetime.fromisoformat(d['timestamp'])
                        metrics.append(MetricPoint.parse_obj(d))
                    yield metrics
        
        # Then read remaining chunk files
        chunk_files = sorted(self.metrics_dir.glob("chunk_*.json"))
        for chunk_file in chunk_files:
            if chunk_file.exists():
                with open(chunk_file, 'r') as f:
                    data = json.load(f)
                    metrics = []
                    for d in data:
                        # Ensure timestamp is datetime
                        if isinstance(d.get('timestamp'), str):
                            d['timestamp'] = datetime.fromisoformat(d['timestamp'])
                        metrics.append(MetricPoint.parse_obj(d))
                    yield metrics
                    
    def save_consolidated_data(self) -> Optional[tuple[Path, Path]]:
        """Save all metrics to consolidated parquet and JSON files before cleanup.
        
        Returns:
            Tuple of (parquet_path, json_path) if successful, None otherwise.
        """
        try:
            # Ensure all metrics are written
            self.flush()
            
            # Create a list to store all metrics
            all_metrics = []
            
            # Iterate through all chunks and collect metrics
            for chunk in self.iter_chunks():
                all_metrics.extend([{
                    'metric_name': m.name,
                    'value': m.value,
                    'timestamp': m.timestamp.isoformat() if isinstance(m.timestamp, datetime) else m.timestamp,
                    **m.tags
                } for m in chunk])
            
            if not all_metrics:
                logger.warning("No metrics found to archive")
                return None
            
            # Convert to DataFrame with explicit timestamp parsing
            df = pd.DataFrame(all_metrics)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Generate archive filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_path = self.archive_dir / f"metrics_{self.replication_id}_{timestamp}.parquet"
            json_path = self.archive_dir / f"metrics_{self.replication_id}_{timestamp}.json"
            
            # Save as parquet
            df.to_parquet(archive_path, index=False)
            
            # Save as JSON for human readability
            with open(json_path, 'w') as f:
                json.dump(all_metrics, f, cls=SimulationEncoder, indent=2)
            
            logger.info(f"Saved consolidated metrics to {archive_path} and {json_path}")
            return archive_path, json_path
            
        except Exception as e:
            logger.error(f"Error saving consolidated data: {e}")
            return None
                    
    def cleanup(self):
        """Clean up all stored metrics after saving consolidated data."""
        shutil.rmtree(self.metrics_dir, ignore_errors=True)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        shutil.rmtree(self.compacted_dir, ignore_errors=True)