"""
Timing utilities for performance debugging.
"""

import time
from functools import wraps
from typing import Dict, List, Callable
import numpy as np


class TimingCollector:
    """Collects timing information for performance analysis."""
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
        self.call_counts: Dict[str, int] = {}
        
    def record(self, name: str, duration: float):
        """Record a timing measurement."""
        if name not in self.timings:
            self.timings[name] = []
            self.call_counts[name] = 0
        self.timings[name].append(duration)
        self.call_counts[name] += 1
        
    def get_summary(self) -> str:
        """Get a summary of all timings."""
        lines = ["Performance Summary:"]
        lines.append("-" * 80)
        lines.append(f"{'Function':<40} {'Calls':>10} {'Total(s)':>10} {'Avg(ms)':>10} {'Max(ms)':>10}")
        lines.append("-" * 80)
        
        # Sort by total time descending
        sorted_funcs = sorted(self.timings.items(), 
                            key=lambda x: sum(x[1]), 
                            reverse=True)
        
        for name, times in sorted_funcs:
            total_time = sum(times)
            avg_time = np.mean(times) * 1000  # Convert to ms
            max_time = max(times) * 1000
            calls = self.call_counts[name]
            
            lines.append(f"{name:<40} {calls:>10} {total_time:>10.2f} {avg_time:>10.2f} {max_time:>10.2f}")
            
        lines.append("-" * 80)
        return "\n".join(lines)


# Global timing collector
_timing_collector = TimingCollector()


def time_function(name: str = None):
    """Decorator to time function execution."""
    def decorator(func: Callable):
        func_name = name or f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                _timing_collector.record(func_name, duration)
                
        return wrapper
    return decorator


def get_timing_summary() -> str:
    """Get the timing summary from the global collector."""
    return _timing_collector.get_summary()


def reset_timing():
    """Reset the timing collector."""
    global _timing_collector
    _timing_collector = TimingCollector()


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        _timing_collector.record(self.name, duration)