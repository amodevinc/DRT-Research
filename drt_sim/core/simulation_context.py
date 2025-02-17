class SimulationContext:
    def _sync_metrics_collector_time(self) -> None:
        """Synchronize the metrics collector with the current simulation time."""
        if self.metrics_collector:
            self.metrics_collector.current_simulation_time = self.get_elapsed_time().total_seconds()
            self.metrics_collector.current_datetime = self.current_time 