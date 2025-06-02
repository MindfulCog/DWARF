import os
import time
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

class PerformanceLogger:
    """Track and log performance metrics"""
    
    def __init__(self, log_dir="logs/performance"):
        self.log_dir = log_dir
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = time.time()
        self.metrics = []
        self.sampling_interval = 5.0  # seconds
        self.last_sample_time = self.start_time
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Initial hardware info
        self.gpu_info = self._get_gpu_info()
        self.cpu_info = self._get_cpu_info()
        
        # Save initial hardware info
        self._save_system_info()
        
    def _get_gpu_info(self):
        """Get GPU information"""
        try:
            import cupy as cp
            device = cp.cuda.Device()
            props = cp.cuda.runtime.getDeviceProperties(device.id)
            memory = cp.cuda.runtime.memGetInfo()
            
            return {
                "name": props["name"].decode(),
                "total_memory": memory[1],
                "free_memory": memory[0],
                "compute_capability": f"{props['major']}.{props['minor']}"
            }
        except Exception as e:
            return {"error": str(e), "available": False}
    
    def _get_cpu_info(self):
        """Get CPU information"""
        cpu_info = {
            "cpu_count": psutil.cpu_count(logical=False),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available
        }
        return cpu_info
    
    def _save_system_info(self):
        """Save system information to file"""
        info_file = os.path.join(self.log_dir, f"system_info_{self.session_id}.txt")
        
        with open(info_file, 'w') as f:
            f.write("DWARF Physics Simulator Performance Info\n")
            f.write("=======================================\n\n")
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Start time: {datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # CPU info
            f.write("CPU Information:\n")
            f.write(f"  Physical cores: {self.cpu_info['cpu_count']}\n")
            f.write(f"  Logical cores: {self.cpu_info['cpu_count_logical']}\n")
            f.write(f"  Total memory: {self.cpu_info['memory_total'] / (1024**3):.2f} GB\n")
            f.write(f"  Available memory: {self.cpu_info['memory_available'] / (1024**3):.2f} GB\n\n")
            
            # GPU info
            f.write("GPU Information:\n")
            if self.gpu_info.get("available", True):
                f.write(f"  Name: {self.gpu_info.get('name', 'Unknown')}\n")
                f.write(f"  Compute capability: {self.gpu_info.get('compute_capability', 'Unknown')}\n")
                f.write(f"  Total memory: {self.gpu_info.get('total_memory', 0) / (1024**3):.2f} GB\n")
                f.write(f"  Free memory: {self.gpu_info.get('free_memory', 0) / (1024**3):.2f} GB\n")
            else:
                f.write(f"  GPU not available: {self.gpu_info.get('error', 'Unknown error')}\n")
    
    def sample_metrics(self, simulator):
        """Sample current performance metrics"""
        current_time = time.time()
        
        # Only sample at specified intervals
        if current_time - self.last_sample_time < self.sampling_interval:
            return
        
        self.last_sample_time = current_time
        
        # Get CPU usage
        cpu_percent = psutil.cpu_percent()
        memory_usage = psutil.Process(os.getpid()).memory_info().rss
        
        # Get GPU usage if available
        gpu_metrics = {}
        try:
            import cupy as cp
            memory_info = cp.cuda.runtime.memGetInfo()
            gpu_memory_used = self.gpu_info.get('total_memory', 0) - memory_info[0]
            gpu_memory_percent = gpu_memory_used / self.gpu_info.get('total_memory', 1) * 100
            
            gpu_metrics = {
                "gpu_memory_used": gpu_memory_used,
                "gpu_memory_percent": gpu_memory_percent
            }
        except Exception as e:
            gpu_metrics = {
                "gpu_memory_used": 0,
                "gpu_memory_percent": 0,
                "error": str(e)
            }
        
        # Simulator specific metrics
        sim_metrics = {}
        if simulator:
            # Particle count
            sim_metrics["particle_count"] = len(simulator.particle_system.particles) if hasattr(simulator, 'particle_system') else 0
            
            # Grid refinement info
            if hasattr(simulator, 'grid') and hasattr(simulator.grid, 'refinement_regions'):
                sim_metrics["refinement_regions"] = sum(len(regions) for regions in simulator.grid.refinement_regions)
            
            # Physics calculation time if available
            if hasattr(simulator, 'physics') and hasattr(simulator.physics, 'last_calculation_time'):
                sim_metrics["physics_calc_time"] = simulator.physics.last_calculation_time
        
        # Combine all metrics
        metrics = {
            "timestamp": current_time,
            "elapsed_time": current_time - self.start_time,
            "cpu_percent": cpu_percent,
            "memory_usage": memory_usage,
            "memory_usage_mb": memory_usage / (1024 * 1024),
            **gpu_metrics,
            **sim_metrics
        }
        
        self.metrics.append(metrics)
        
        # Log to console every 5 samples
        if len(self.metrics) % 5 == 0:
            print(f"Performance: CPU: {cpu_percent:.1f}%, "
                  f"Memory: {memory_usage / (1024 * 1024):.1f} MB, "
                  f"GPU Memory: {gpu_metrics.get('gpu_memory_percent', 0):.1f}%, "
                  f"Particles: {sim_metrics.get('particle_count', 0)}")
    
    def save_metrics(self):
        """Save collected metrics to file"""
        if not self.metrics:
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(self.metrics)
        
        # Save to CSV
        csv_file = os.path.join(self.log_dir, f"performance_{self.session_id}.csv")
        df.to_csv(csv_file, index=False)
        
        # Generate performance plots
        self._generate_plots()
        
        # Clear metrics after saving
        self.metrics = []
    
    def _generate_plots(self):
        """Generate performance plots"""
        if not self.metrics:
            return
            
        df = pd.DataFrame(self.metrics)
        plot_dir = os.path.join(self.log_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        
        # CPU & Memory usage plot
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(df["elapsed_time"], df["cpu_percent"], 'b-', label="CPU Usage %")
        plt.title("CPU Usage Over Time")
        plt.ylabel("Percent %")
        plt.grid(True)
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(df["elapsed_time"], df["memory_usage_mb"], 'g-', label="Memory Usage (MB)")
        if "gpu_memory_percent" in df.columns:
            plt.plot(df["elapsed_time"], df["gpu_memory_percent"], 'r-', label="GPU Memory %")
        plt.title("Memory Usage Over Time")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Usage")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(os.path.join(plot_dir, f"resource_usage_{self.session_id}.png"))
        plt.close()
        
        # Simulation-specific metrics
        if "particle_count" in df.columns:
            plt.figure(figsize=(10, 5))
            plt.plot(df["elapsed_time"], df["particle_count"], 'b-o', label="Particle Count")
            plt.title("Simulation Size Over Time")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Number of particles")
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(plot_dir, f"particles_{self.session_id}.png"))
            plt.close()
        
        if "physics_calc_time" in df.columns:
            plt.figure(figsize=(10, 5))
            plt.plot(df["elapsed_time"], df["physics_calc_time"] * 1000, 'r-o', label="Physics Calculation Time (ms)")
            plt.title("Physics Calculation Performance")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Calculation Time (ms)")
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(plot_dir, f"physics_perf_{self.session_id}.png"))
            plt.close()

    def finalize(self):
        """Finalize logging and save all metrics"""
        # Save final metrics
        self.save_metrics()
        
        # Update system info file with end time
        info_file = os.path.join(self.log_dir, f"system_info_{self.session_id}.txt")
        with open(info_file, 'a') as f:
            end_time = time.time()
            duration = end_time - self.start_time
            f.write(f"\nEnd time: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)\n")
            
            # Add final memory stats
            f.write("\nFinal Memory Usage:\n")
            f.write(f"  Process memory: {psutil.Process(os.getpid()).memory_info().rss / (1024**3):.3f} GB\n")
            
            # Add final GPU stats if available
            try:
                import cupy as cp
                memory_info = cp.cuda.runtime.memGetInfo()
                gpu_memory_used = self.gpu_info.get('total_memory', 0) - memory_info[0]
                f.write("\nFinal GPU Usage:\n")
                f.write(f"  GPU memory used: {gpu_memory_used / (1024**3):.3f} GB\n")
                f.write(f"  GPU memory free: {memory_info[0] / (1024**3):.3f} GB\n")
            except:
                pass