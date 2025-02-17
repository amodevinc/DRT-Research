import queue
from dataclasses import dataclass
from datetime import datetime
from threading import Thread, Event
import time
from typing import Dict, Optional
import logging
import mlflow
import traceback

logger = logging.getLogger(__name__)

@dataclass
class MetricEntry:
    name: str
    value: float
    timestamp: datetime
    tags: Optional[Dict[str, str]] = None

class MLflowAdapter:
    def __init__(self, flush_interval: float = 5.0, max_queue_size: int = 10000):
        """Initialize the MLflow adapter with batched logging.
        
        Args:
            flush_interval: How often to flush queues to MLflow (in seconds)
            max_queue_size: Maximum number of entries to queue before blocking
        """
        self.metrics_queue = queue.Queue(maxsize=max_queue_size)
        self.tags_queue = queue.Queue(maxsize=max_queue_size)
        self.flush_interval = flush_interval
        self.shutdown_event = Event()
        
        # Start background worker
        self.worker_thread = Thread(target=self._process_queues, daemon=True)
        self.worker_thread.start()
        
    def log_metric(self, name: str, value: float, timestamp: datetime, tags: Optional[Dict[str, str]] = None):
        """Queue a metric for logging to MLflow."""
        try:
            entry = MetricEntry(name=name, value=value, timestamp=timestamp, tags=tags)
            self.metrics_queue.put(entry, timeout=1)
        except queue.Full:
            logger.warning(f"Metrics queue full, dropping metric: {name}")
            
    def set_tag(self, key: str, value: str):
        """Queue a tag for logging to MLflow."""
        try:
            self.tags_queue.put((key, value), timeout=1)
        except queue.Full:
            logger.warning(f"Tags queue full, dropping tag: {key}")
            
    def _process_queues(self):
        """Background worker that processes queued metrics and tags."""
        while not self.shutdown_event.is_set():
            try:
                # Process metrics in batches
                metrics_batch = []
                while not self.metrics_queue.empty() and len(metrics_batch) < 100:
                    try:
                        metric = self.metrics_queue.get_nowait()
                        metrics_batch.append(metric)
                    except queue.Empty:
                        break
                        
                # Log metrics batch
                for metric in metrics_batch:
                    try:
                        mlflow.log_metric(
                            metric.name,
                            metric.value,
                            int(metric.timestamp.timestamp() * 1000)
                        )
                        if metric.tags:
                            for tag_key, tag_value in metric.tags.items():
                                mlflow.set_tag(f"{metric.name}.{tag_key}", tag_value)
                    except Exception as e:
                        logger.error(f"Error logging metric to MLflow: {e}")
                    finally:
                        self.metrics_queue.task_done()
                        
                # Process tags
                while not self.tags_queue.empty():
                    try:
                        key, value = self.tags_queue.get_nowait()
                        mlflow.set_tag(key, value)
                        self.tags_queue.task_done()
                    except queue.Empty:
                        break
                    except Exception as e:
                        logger.error(f"Error setting tag in MLflow: {e}")
                        
                # Sleep until next flush interval
                time.sleep(self.flush_interval)
                
            except Exception as e:
                logger.error(f"Error in MLflow processing thread: {e}")
                
    def shutdown(self):
        """Shutdown the adapter and process any remaining items."""
        logger.info("Starting MLflow adapter shutdown")
        try:
            # Signal shutdown
            self.shutdown_event.set()
            
            # Process remaining items
            timeout = 30  # 30 seconds timeout
            start_time = time.time()
            
            while (not self.metrics_queue.empty() or not self.tags_queue.empty()) and (time.time() - start_time < timeout):
                # Process metrics
                while not self.metrics_queue.empty():
                    try:
                        metric = self.metrics_queue.get_nowait()
                        try:
                            mlflow.log_metric(
                                metric.name,
                                metric.value,
                                int(metric.timestamp.timestamp() * 1000)
                            )
                            if metric.tags:
                                for tag_key, tag_value in metric.tags.items():
                                    mlflow.set_tag(f"{metric.name}.{tag_key}", tag_value)
                        except Exception as e:
                            logger.error(f"Error logging final metric to MLflow: {e}")
                        finally:
                            self.metrics_queue.task_done()
                    except queue.Empty:
                        break
                
                # Process tags
                while not self.tags_queue.empty():
                    try:
                        key, value = self.tags_queue.get_nowait()
                        try:
                            mlflow.set_tag(key, value)
                        except Exception as e:
                            logger.error(f"Error setting final tag in MLflow: {e}")
                        finally:
                            self.tags_queue.task_done()
                    except queue.Empty:
                        break
                
                time.sleep(0.1)  # Small sleep to prevent tight loop
            
            if not self.metrics_queue.empty() or not self.tags_queue.empty():
                remaining_metrics = self.metrics_queue.qsize()
                remaining_tags = self.tags_queue.qsize()
                logger.warning(f"Shutdown timeout reached with items still in queue: {remaining_metrics} metrics, {remaining_tags} tags")
            else:
                logger.info("Successfully processed all queued items during shutdown")
            
            # Wait for worker thread to finish
            self.worker_thread.join(timeout=5)
            if self.worker_thread.is_alive():
                logger.warning("Worker thread did not terminate cleanly")
            
        except Exception as e:
            logger.error(f"Error during MLflow adapter shutdown: {str(e)}\nTraceback: {traceback.format_exc()}") 