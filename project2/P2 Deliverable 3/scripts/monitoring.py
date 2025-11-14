#!/usr/bin/env python3
"""
monitoring.py - System Monitoring and Metrics Collection

Provides comprehensive monitoring including:
- Performance metrics tracking
- Resource usage monitoring
- Error tracking and alerting
- Health checks
- Prometheus metrics export (optional)
"""

import os
import time
import psutil
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import deque, defaultdict
from dataclasses import dataclass, field
import threading


@dataclass
class MonitoringConfig:
    """Monitoring configuration settings."""
    enable_monitoring: bool = True
    enable_metrics_export: bool = False
    metrics_port: int = 9090
    health_check_interval: int = 60
    metrics_retention_minutes: int = 60
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'cpu_percent': 80.0,
        'memory_percent': 85.0,
        'error_rate_per_minute': 10.0,
        'avg_response_time_seconds': 5.0
    })


class SystemMonitor:
    """Monitors system health and performance metrics."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Metrics storage (time-series data)
        retention_size = config.metrics_retention_minutes * 60
        self.metrics_history = {
            'cpu_usage': deque(maxlen=retention_size),
            'memory_usage': deque(maxlen=retention_size),
            'response_times': deque(maxlen=1000),
            'error_counts': deque(maxlen=retention_size),
            'request_counts': deque(maxlen=retention_size),
        }
        
        # Real-time counters
        self.counters = defaultdict(int)
        self.gauges = {}
        
        # Error tracking
        self.recent_errors = deque(maxlen=100)
        
        # Health check thread
        self.health_check_thread = None
        self.stop_monitoring = threading.Event()
        
        if self.config.enable_monitoring:
            self.start_monitoring()
    
    def _setup_logging(self) -> logging.Logger:
        """Configure monitoring logging."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - MONITOR - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def record_request(self, response_time: float):
        """Record a request with its response time."""
        timestamp = time.time()
        self.metrics_history['response_times'].append({
            'timestamp': timestamp,
            'value': response_time
        })
        self.metrics_history['request_counts'].append({
            'timestamp': timestamp,
            'value': 1
        })
        self.counters['total_requests'] += 1
    
    def record_error(self, error_type: str, error_message: str, context: Optional[Dict] = None):
        """Record an error for tracking."""
        timestamp = time.time()
        error_info = {
            'timestamp': timestamp,
            'type': error_type,
            'message': error_message,
            'context': context or {}
        }
        
        self.recent_errors.append(error_info)
        self.metrics_history['error_counts'].append({
            'timestamp': timestamp,
            'value': 1
        })
        self.counters['total_errors'] += 1
        self.counters[f'errors_{error_type}'] += 1
        
        self.logger.error(f"Error recorded: {error_type} - {error_message}")
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system resource usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            metrics = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_mb': memory.used / (1024 * 1024),
                'memory_available_mb': memory.available / (1024 * 1024),
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024 * 1024 * 1024),
                'timestamp': time.time()
            }
            
            # Record metrics
            self.metrics_history['cpu_usage'].append({
                'timestamp': metrics['timestamp'],
                'value': cpu_percent
            })
            self.metrics_history['memory_usage'].append({
                'timestamp': metrics['timestamp'],
                'value': memory.percent
            })
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get system metrics: {e}")
            return {}
    
    def get_application_metrics(self) -> Dict[str, Any]:
        """Get application-specific metrics."""
        current_time = time.time()
        
        # Calculate average response time (last minute)
        recent_responses = [
            r['value'] for r in self.metrics_history['response_times']
            if current_time - r['timestamp'] < 60
        ]
        avg_response_time = sum(recent_responses) / len(recent_responses) if recent_responses else 0
        
        # Calculate error rate (last minute)
        recent_errors = [
            e for e in self.metrics_history['error_counts']
            if current_time - e['timestamp'] < 60
        ]
        error_rate = len(recent_errors)
        
        # Calculate request rate (last minute)
        recent_requests = [
            r for r in self.metrics_history['request_counts']
            if current_time - r['timestamp'] < 60
        ]
        request_rate = len(recent_requests)
        
        return {
            'total_requests': self.counters['total_requests'],
            'total_errors': self.counters['total_errors'],
            'requests_per_minute': request_rate,
            'errors_per_minute': error_rate,
            'avg_response_time_seconds': avg_response_time,
            'error_rate_percent': (error_rate / request_rate * 100) if request_rate > 0 else 0,
            'timestamp': current_time
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status."""
        system_metrics = self.get_system_metrics()
        app_metrics = self.get_application_metrics()
        
        # Check against thresholds
        alerts = []
        status = "healthy"
        
        if system_metrics.get('cpu_percent', 0) > self.config.alert_thresholds['cpu_percent']:
            alerts.append(f"High CPU usage: {system_metrics['cpu_percent']:.1f}%")
            status = "degraded"
        
        if system_metrics.get('memory_percent', 0) > self.config.alert_thresholds['memory_percent']:
            alerts.append(f"High memory usage: {system_metrics['memory_percent']:.1f}%")
            status = "degraded"
        
        if app_metrics['errors_per_minute'] > self.config.alert_thresholds['error_rate_per_minute']:
            alerts.append(f"High error rate: {app_metrics['errors_per_minute']} errors/min")
            status = "unhealthy"
        
        if app_metrics['avg_response_time_seconds'] > self.config.alert_thresholds['avg_response_time_seconds']:
            alerts.append(f"Slow response time: {app_metrics['avg_response_time_seconds']:.2f}s")
            status = "degraded"
        
        return {
            'status': status,
            'alerts': alerts,
            'system_metrics': system_metrics,
            'application_metrics': app_metrics,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_error_summary(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get summary of recent errors."""
        cutoff_time = time.time() - (time_window_minutes * 60)
        
        recent = [e for e in self.recent_errors if e['timestamp'] > cutoff_time]
        
        # Group by type
        by_type = defaultdict(list)
        for error in recent:
            by_type[error['type']].append(error)
        
        return {
            'total_errors': len(recent),
            'error_types': {
                error_type: len(errors)
                for error_type, errors in by_type.items()
            },
            'recent_errors': list(recent)[-10:],  # Last 10 errors
            'time_window_minutes': time_window_minutes
        }
    
    def start_monitoring(self):
        """Start background monitoring thread."""
        if self.health_check_thread and self.health_check_thread.is_alive():
            self.logger.warning("Monitoring already running")
            return
        
        self.stop_monitoring.clear()
        self.health_check_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.health_check_thread.start()
        self.logger.info("System monitoring started")
    
    def stop_monitoring_thread(self):
        """Stop background monitoring thread."""
        if self.health_check_thread:
            self.stop_monitoring.set()
            self.health_check_thread.join(timeout=5)
            self.logger.info("System monitoring stopped")
    
    def _monitoring_loop(self):
        """Background thread for periodic health checks."""
        while not self.stop_monitoring.is_set():
            try:
                # Collect metrics
                self.get_system_metrics()
                
                # Check health status
                health = self.get_health_status()
                
                # Log alerts
                if health['alerts']:
                    for alert in health['alerts']:
                        self.logger.warning(f"ALERT: {alert}")
                
                # Wait for next interval
                if self.stop_monitoring.wait(timeout=self.config.health_check_interval):
                    break
                    
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
    
    def export_metrics_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        app_metrics = self.get_application_metrics()
        system_metrics = self.get_system_metrics()
        
        lines = [
            "# HELP app_requests_total Total number of requests",
            "# TYPE app_requests_total counter",
            f"app_requests_total {self.counters['total_requests']}",
            "",
            "# HELP app_errors_total Total number of errors",
            "# TYPE app_errors_total counter",
            f"app_errors_total {self.counters['total_errors']}",
            "",
            "# HELP app_response_time_seconds Average response time",
            "# TYPE app_response_time_seconds gauge",
            f"app_response_time_seconds {app_metrics['avg_response_time_seconds']}",
            "",
            "# HELP system_cpu_percent CPU usage percentage",
            "# TYPE system_cpu_percent gauge",
            f"system_cpu_percent {system_metrics.get('cpu_percent', 0)}",
            "",
            "# HELP system_memory_percent Memory usage percentage",
            "# TYPE system_memory_percent gauge",
            f"system_memory_percent {system_metrics.get('memory_percent', 0)}",
        ]
        
        return "\n".join(lines)
    
    def reset_metrics(self):
        """Reset all metrics (useful for testing)."""
        self.counters.clear()
        for key in self.metrics_history:
            self.metrics_history[key].clear()
        self.recent_errors.clear()
        self.logger.info("All metrics reset")


# Optional: Prometheus metrics server
def start_metrics_server(monitor: SystemMonitor, port: int = 9090):
    """Start a simple HTTP server for Prometheus metrics."""
    try:
        from http.server import HTTPServer, BaseHTTPRequestHandler
        
        class MetricsHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/metrics':
                    metrics = monitor.export_metrics_prometheus()
                    self.send_response(200)
                    self.send_header('Content-type', 'text/plain')
                    self.end_headers()
                    self.wfile.write(metrics.encode())
                elif self.path == '/health':
                    health = monitor.get_health_status()
                    self.send_response(200 if health['status'] == 'healthy' else 503)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    import json
                    self.wfile.write(json.dumps(health, indent=2).encode())
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def log_message(self, format, *args):
                pass  # Suppress request logging
        
        server = HTTPServer(('', port), MetricsHandler)
        monitor.logger.info(f"Metrics server started on port {port}")
        server.serve_forever()
        
    except Exception as e:
        monitor.logger.error(f"Failed to start metrics server: {e}")
