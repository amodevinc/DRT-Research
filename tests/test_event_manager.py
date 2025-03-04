import unittest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import threading
import time

from drt_sim.core.events.manager import EventManager
from drt_sim.models.event import Event, EventType, EventPriority, EventStatus

class TestEventManagerSync(unittest.TestCase):
    """Synchronous tests for EventManager"""
    
    def setUp(self):
        self.event_manager = EventManager(max_history_size=100, continue_on_handler_error=False)
        self.current_time = datetime.now()
    
    def test_register_handler(self):
        # Test handler registration
        mock_handler = MagicMock()
        self.event_manager.register_handler(EventType.SIMULATION_START, mock_handler)
        
        self.assertIn(EventType.SIMULATION_START, self.event_manager.handlers)
        self.assertEqual(len(self.event_manager.handlers[EventType.SIMULATION_START]), 1)
        self.assertEqual(self.event_manager.handlers[EventType.SIMULATION_START][0], mock_handler)
    
    def test_register_error_handler(self):
        # Test error handler registration
        mock_error_handler = MagicMock()
        self.event_manager.register_error_handler(EventType.SIMULATION_ERROR, mock_error_handler)
        
        self.assertIn(EventType.SIMULATION_ERROR, self.event_manager.error_handlers)
        self.assertEqual(self.event_manager.error_handlers[EventType.SIMULATION_ERROR], mock_error_handler)
    
    def test_event_validation(self):
        # Test valid event
        valid_event = Event(
            event_type=EventType.SIMULATION_START,
            priority=EventPriority.NORMAL,
            timestamp=self.current_time
        )
        is_valid, _ = self.event_manager._validate_event(valid_event)
        self.assertTrue(is_valid)
        
        # Test invalid event (missing timestamp)
        invalid_event = Event(
            event_type=EventType.SIMULATION_START,
            priority=EventPriority.NORMAL,
            timestamp=None  # type: ignore
        )
        is_valid, error_msg = self.event_manager._validate_event(invalid_event)
        self.assertFalse(is_valid)
        self.assertEqual(error_msg, "Missing timestamp")
        
        # Test custom validation rule
        def custom_rule(event):
            return event.priority == EventPriority.HIGH
        
        self.event_manager.register_handler(
            EventType.SIMULATION_START,
            MagicMock(),
            validation_rules=[custom_rule]
        )
        
        normal_priority_event = Event(
            event_type=EventType.SIMULATION_START,
            priority=EventPriority.NORMAL,
            timestamp=self.current_time
        )
        
        is_valid, error_msg = self.event_manager._validate_event(normal_priority_event)
        self.assertFalse(is_valid)
        self.assertEqual(error_msg, "Failed custom validation rule")
        
        high_priority_event = Event(
            event_type=EventType.SIMULATION_START,
            priority=EventPriority.HIGH,
            timestamp=self.current_time
        )
        
        is_valid, _ = self.event_manager._validate_event(high_priority_event)
        self.assertTrue(is_valid)
    
    def test_recurring_event(self):
        # Test recurring event scheduling
        mock_handler = MagicMock()
        self.event_manager.register_handler(EventType.METRICS_COLLECTED, mock_handler)
        
        # Schedule recurring event
        recurring_event = self.event_manager.schedule_recurring_event(
            event_type=EventType.METRICS_COLLECTED,
            start_time=self.current_time,
            interval_seconds=60,
            end_time=self.current_time + timedelta(minutes=5)
        )
        
        self.assertTrue(recurring_event.is_recurring)
        self.assertEqual(recurring_event.recurrence_interval, 60)
        self.assertEqual(self.event_manager.event_queue.qsize(), 1)
    
    def test_cancel_event(self):
        # Test event cancellation
        event1 = Event(
            event_type=EventType.SIMULATION_START,
            priority=EventPriority.NORMAL,
            timestamp=self.current_time
        )
        
        event2 = Event(
            event_type=EventType.SIMULATION_END,
            priority=EventPriority.NORMAL,
            timestamp=self.current_time + timedelta(minutes=1)
        )
        
        self.event_manager.publish_event(event1)
        self.event_manager.publish_event(event2)
        
        # Cancel the first event
        result = self.event_manager.cancel_event(event1.id)
        self.assertTrue(result)
        
        # Queue should only have the second event
        self.assertEqual(self.event_manager.event_queue.qsize(), 1)
        
        # History should have the cancelled event
        self.assertEqual(len(self.event_manager.event_history), 1)
        self.assertEqual(self.event_manager.event_history[0].status, EventStatus.CANCELLED)
        
        # Try to cancel non-existent event
        result = self.event_manager.cancel_event("non-existent-id")
        self.assertFalse(result)
    
    def test_history_size_limit(self):
        # Test history size limit
        small_manager = EventManager(max_history_size=3)
        
        # Add 5 events to history
        for i in range(5):
            event = Event(
                event_type=EventType.SIMULATION_START,
                priority=EventPriority.NORMAL,
                timestamp=self.current_time + timedelta(minutes=i)
            )
            small_manager._add_to_history(event)
        
        # History should be limited to 3 most recent events
        self.assertEqual(len(small_manager.event_history), 3)
        self.assertEqual(
            small_manager.event_history[0].timestamp,
            self.current_time + timedelta(minutes=2)
        )
    
    def test_thread_safety(self):
        # Test thread safety with concurrent access
        def add_events():
            for i in range(50):
                event = Event(
                    event_type=EventType.SIMULATION_START,
                    priority=EventPriority.NORMAL,
                    timestamp=self.current_time + timedelta(seconds=i)
                )
                self.event_manager.publish_event(event)
                time.sleep(0.01)  # Small delay to increase chance of race conditions
        
        # Create and start threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=add_events)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Should have 250 events (50 events * 5 threads)
        self.assertEqual(self.event_manager.event_queue.qsize(), 250)
    
    def test_get_event_history_filtering(self):
        # Test event history filtering
        # Add events of different types
        event1 = Event(
            event_type=EventType.SIMULATION_START,
            priority=EventPriority.NORMAL,
            timestamp=self.current_time - timedelta(minutes=5)
        )
        
        event2 = Event(
            event_type=EventType.SIMULATION_END,
            priority=EventPriority.NORMAL,
            timestamp=self.current_time
        )
        
        event3 = Event(
            event_type=EventType.SIMULATION_START,
            priority=EventPriority.NORMAL,
            timestamp=self.current_time + timedelta(minutes=5)
        )
        
        self.event_manager._add_to_history(event1)
        self.event_manager._add_to_history(event2)
        self.event_manager._add_to_history(event3)
        
        # Filter by event type
        filtered_by_type = self.event_manager.get_event_history(event_type=EventType.SIMULATION_START)
        self.assertEqual(len(filtered_by_type), 2)
        
        # Filter by time range
        filtered_by_time = self.event_manager.get_event_history(
            start_time=self.current_time - timedelta(minutes=1),
            end_time=self.current_time + timedelta(minutes=1)
        )
        self.assertEqual(len(filtered_by_time), 1)
        self.assertEqual(filtered_by_time[0].event_type, EventType.SIMULATION_END)
    
    def test_serializable_history(self):
        # Test serializable history
        event = Event(
            event_type=EventType.SIMULATION_START,
            priority=EventPriority.NORMAL,
            timestamp=self.current_time,
            data={"test_key": "test_value"},
            service_metrics={"metric1": 1.0}
        )
        
        self.event_manager._add_to_history(event)
        
        serialized = self.event_manager.get_serializable_history()
        self.assertEqual(len(serialized), 1)
        self.assertEqual(serialized[0]["event_type"], "simulation.start")
        self.assertEqual(serialized[0]["data"]["test_key"], "test_value")
        self.assertEqual(serialized[0]["service_metrics"]["metric1"], 1.0)
    
    def test_cleanup(self):
        # Test cleanup
        self.event_manager.register_handler(EventType.SIMULATION_START, MagicMock())
        self.event_manager.publish_event(
            Event(
                event_type=EventType.SIMULATION_START,
                priority=EventPriority.NORMAL,
                timestamp=self.current_time
            )
        )
        self.event_manager._add_to_history(
            Event(
                event_type=EventType.SIMULATION_START,
                priority=EventPriority.NORMAL,
                timestamp=self.current_time
            )
        )
        
        # Verify resources exist
        self.assertGreater(len(self.event_manager.handlers), 0)
        self.assertEqual(self.event_manager.event_queue.qsize(), 1)
        self.assertEqual(len(self.event_manager.event_history), 1)
        
        # Cleanup
        self.event_manager.cleanup()
        
        # Verify resources are cleared
        self.assertEqual(len(self.event_manager.handlers), 0)
        self.assertEqual(self.event_manager.event_queue.qsize(), 0)
        self.assertEqual(len(self.event_manager.event_history), 0)


class TestEventManagerAsync(unittest.IsolatedAsyncioTestCase):
    """Asynchronous tests for EventManager"""
    
    def setUp(self):
        self.event_manager = EventManager(max_history_size=100, continue_on_handler_error=False)
        self.current_time = datetime.now()
    
    async def test_process_event(self):
        # Test successful event processing
        mock_handler = MagicMock()
        self.event_manager.register_handler(EventType.SIMULATION_START, mock_handler)
        
        event = Event(
            event_type=EventType.SIMULATION_START,
            priority=EventPriority.NORMAL,
            timestamp=self.current_time
        )
        
        result = await self.event_manager._process_event(event)
        self.assertTrue(result)
        mock_handler.assert_called_once_with(event)
        
        # Check event history
        self.assertEqual(len(self.event_manager.event_history), 2)  # PROCESSING and COMPLETED events
        self.assertEqual(self.event_manager.event_history[0].status, EventStatus.PROCESSING)
        self.assertEqual(self.event_manager.event_history[1].status, EventStatus.COMPLETED)
    
    async def test_process_event_with_error(self):
        # Test event processing with error
        mock_handler = MagicMock(side_effect=Exception("Test error"))
        self.event_manager.register_handler(EventType.SIMULATION_ERROR, mock_handler)
        
        event = Event(
            event_type=EventType.SIMULATION_ERROR,
            priority=EventPriority.NORMAL,
            timestamp=self.current_time
        )
        
        result = await self.event_manager._process_event(event)
        self.assertFalse(result)
        mock_handler.assert_called_once_with(event)
        
        # Check event history
        self.assertEqual(len(self.event_manager.event_history), 2)  # PROCESSING and FAILED events
        self.assertEqual(self.event_manager.event_history[0].status, EventStatus.PROCESSING)
        self.assertEqual(self.event_manager.event_history[1].status, EventStatus.FAILED)
        self.assertIn("error_message", self.event_manager.event_history[1].data)
        self.assertEqual(self.event_manager.event_history[1].data["error_message"], "Test error")
    
    async def test_continue_on_handler_error(self):
        # Test continue_on_handler_error=True
        self.event_manager = EventManager(max_history_size=100, continue_on_handler_error=True)
        
        mock_handler1 = MagicMock(side_effect=Exception("Test error"))
        mock_handler2 = MagicMock()
        
        self.event_manager.register_handler(EventType.SIMULATION_START, mock_handler1)
        self.event_manager.register_handler(EventType.SIMULATION_START, mock_handler2)
        
        event = Event(
            event_type=EventType.SIMULATION_START,
            priority=EventPriority.NORMAL,
            timestamp=self.current_time
        )
        
        result = await self.event_manager._process_event(event)
        self.assertFalse(result)  # Still returns False because there was an error
        mock_handler1.assert_called_once_with(event)
        mock_handler2.assert_called_once_with(event)  # Second handler still called
    
    async def test_process_events(self):
        # Test processing multiple events
        mock_handler = MagicMock()
        self.event_manager.register_handler(EventType.SIMULATION_START, mock_handler)
        
        # Add events to queue
        event1 = Event(
            event_type=EventType.SIMULATION_START,
            priority=EventPriority.NORMAL,
            timestamp=self.current_time - timedelta(minutes=1)  # Past event
        )
        
        event2 = Event(
            event_type=EventType.SIMULATION_START,
            priority=EventPriority.NORMAL,
            timestamp=self.current_time - timedelta(seconds=30)  # Past event
        )
        
        event3 = Event(
            event_type=EventType.SIMULATION_START,
            priority=EventPriority.NORMAL,
            timestamp=self.current_time + timedelta(minutes=1)  # Future event
        )
        
        self.event_manager.publish_event(event1)
        self.event_manager.publish_event(event2)
        self.event_manager.publish_event(event3)
        
        # Process events up to current time
        processed_events = await self.event_manager.process_events(self.current_time)
        
        # Should process 2 past events but not the future event
        self.assertEqual(len(processed_events), 2)
        self.assertEqual(mock_handler.call_count, 2)
        self.assertEqual(self.event_manager.event_queue.qsize(), 1)  # Future event still in queue


# Run the tests
if __name__ == "__main__":
    unittest.main() 