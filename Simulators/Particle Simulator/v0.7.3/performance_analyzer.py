import numpy as np
import time
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
import argparse

class PerformanceAnalyzer:
    """Analyze and visualize DWARF simulation performance data"""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.sessions = self._discover_sessions()
    
    def _discover_sessions(self) -> List[str]:
        """Discover available simulation sessions"""
        if not self.log_dir.exists():
            return []
        
        sessions = []
        for item in self.log_dir.iterdir():
            if item.is_dir() and item.name.replace('_', '').replace('-', '').isdigit():
                sessions.append(item.name)
        
        return sorted(sessions)
    
    def analyze_session(self, session_id: str) -> Optional[Dict]:
        """Analyze performance data for a specific session"""
        session_dir = self.log_dir / session_id
        
        if not session_dir.exists():
            print(f"Session {session_id} not found")
            return None
        
        # Load performance data
        perf_file = session_dir / "performance" / "performance_log.json"
        if not perf_file.exists():
            print(f"No performance data found for session {session_id}")
            return None
        
        with open(perf_file, 'r') as f:
            perf_data = json.load(f)
        
        # Load final statistics
        stats_file = session_dir / "final_statistics.json"
        final_stats = {}
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                final_stats = json.load(f)
        
        return self._process_performance_data(perf_data, final_stats)
    
    def _process_performance_data(self, perf_data: List[Dict], final_stats: Dict) -> Dict:
        """Process raw performance data into analysis results"""
        
        if not perf_data:
            return {}
        
        # Extract time series data
        steps = [entry["step"] for entry in perf_data]
        wall_times = [entry["stats"]["wall_time"] for entry in perf_data]
        physics_times = [entry["stats"]["physics_time"] for entry in perf_data]
        timesteps = [entry["stats"].get("timestep", 0.01) for entry in perf_data]
        
        # Calculate derived metrics
        efficiencies = [p/w if w > 0 else 0 for p, w in zip(physics_times, wall_times)]
        fps_values = [1.0/w if w > 0 else 0 for w in wall_times]
        
        # Statistical analysis
        analysis = {
            "session_info": {
                "total_steps": len(steps),
                "final_step": max(steps) if steps else 0,
                "duration": max(steps) * np.mean(timesteps) if steps else 0
            },
            "timing_stats": {
                "wall_time": {
                    "mean": np.mean(wall_times),
                    "std": np.std(wall_times),
                    "min": np.min(wall_times),
                    "max": np.max(wall_times),
                    "median": np.median(wall_times)
                },
                "physics_time": {
                    "mean": np.mean(physics_times),
                    "std": np.std(physics_times),
                    "min": np.min(physics_times),
                    "max": np.max(physics_times),
                    "median": np.median(physics_times)
                }
            },
            "performance_metrics": {
                "efficiency": {
                    "mean": np.mean(efficiencies),
                    "std": np.std(efficiencies),
                    "min": np.min(efficiencies),
                    "max": np.max(efficiencies)
                },
                "fps": {
                    "mean": np.mean(fps_values),
                    "std": np.std(fps_values),
                    "min": np.min(fps_values),
                    "max": np.max(fps_values)
                }
            },
            "time_series": {
                "steps": steps,
                "wall_times": wall_times,
                "physics_times": physics_times,
                "efficiencies": efficiencies,
                "fps": fps_values,
                "timesteps": timesteps
            },
            "final_stats": final_stats
        }
        
        return analysis
    
    def generate_performance_plots(self, session_id: str, output_dir: Optional[str] = None):
        """Generate performance visualization plots"""
        analysis = self.analyze_session(session_id)
        if not analysis:
            return
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = self.log_dir / session_id
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'DWARF Simulation Performance Analysis - Session {session_id}')
        
        ts = analysis["time_series"]
        
        # Plot 1: Wall time vs step
        axes[0, 0].plot(ts["steps"], np.array(ts["wall_times"]) * 1000, 'b-', alpha=0.7)
        axes[0, 0].set_xlabel('Simulation Step')
        axes[0, 0].set_ylabel('Wall Time (ms)')
        axes[0, 0].set_title('Wall Clock Time per Step')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Physics time vs step
        axes[0, 1].plot(ts["steps"], np.array(ts["physics_times"]) * 1000, 'r-', alpha=0.7)
        axes[0, 1].set_xlabel('Simulation Step')
        axes[0, 1].set_ylabel('Physics Time (ms)')
        axes[0, 1].set_title('Physics Computation Time per Step')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Efficiency over time
        axes[1, 0].plot(ts["steps"], ts["efficiencies"], 'g-', alpha=0.7)
        axes[1, 0].set_xlabel('Simulation Step')
        axes[1, 0].set_ylabel('Efficiency (Physics/Wall Time)')
        axes[1, 0].set_title('Computational Efficiency')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 1)
        
        # Plot 4: FPS over time
        axes[1, 1].plot(ts["steps"], ts["fps"], 'm-', alpha=0.7)
        axes[1, 1].set_xlabel('Simulation Step')
        axes[1, 1].set_ylabel('FPS')
        axes[1, 1].set_title('Simulation Frame Rate')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = output_path / f"performance_analysis_{session_id}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Performance plots saved to {plot_file}")
        
        plt.show()
    
    def print_session_summary(self, session_id: str):
        """Print a detailed summary of session performance"""
        analysis = self.analyze_session(session_id)
        if not analysis:
            return
        
        print(f"\n{'='*60}")
        print(f"PERFORMANCE ANALYSIS - Session {session_id}")
        print(f"{'='*60}")
        
        # Session info
        info = analysis["session_info"]
        print(f"\nSession Information:")
        print(f"  Total Steps: {info['total_steps']}")
        print(f"  Final Step: {info['final_step']}")
        print(f"  Simulation Duration: {info['duration']:.2f} seconds")
        
        # Timing statistics
        wall_stats = analysis["timing_stats"]["wall_time"]
        phys_stats = analysis["timing_stats"]["physics_time"]
        
        print(f"\nTiming Statistics:")
        print(f"  Wall Time (ms):")
        print(f"    Mean: {wall_stats['mean']*1000:.2f} ± {wall_stats['std']*1000:.2f}")
        print(f"    Range: [{wall_stats['min']*1000:.2f}, {wall_stats['max']*1000:.2f}]")
        print(f"  Physics Time (ms):")
        print(f"    Mean: {phys_stats['mean']*1000:.2f} ± {phys_stats['std']*1000:.2f}")
        print(f"    Range: [{phys_stats['min']*1000:.2f}, {phys_stats['max']*1000:.2f}]")
        
        # Performance metrics
        eff_stats = analysis["performance_metrics"]["efficiency"]
        fps_stats = analysis["performance_metrics"]["fps"]
        
        print(f"\nPerformance Metrics:")
        print(f"  Efficiency: {eff_stats['mean']:.1%} ± {eff_stats['std']:.1%}")
        print(f"  Average FPS: {fps_stats['mean']:.1f} ± {fps_stats['std']:.1f}")
        print(f"  Peak FPS: {fps_stats['max']:.1f}")
        
        # Configuration info
        final_stats = analysis["final_stats"]
        if "configuration" in final_stats:
            config = final_stats["configuration"]
            print(f"\nConfiguration:")
            print(f"  Particles: {config.get('num_particles', 'Unknown')}")
            print(f"  Grid Resolution: {config.get('grid_resolution', 'Unknown')}")
            print(f"  GPU Enabled: {config.get('use_gpu', 'Unknown')}")
    
    def compare_sessions(self, session_ids: List[str]):
        """Compare performance across multiple sessions"""
        print(f"\n{'='*80}")
        print(f"SESSION COMPARISON")
        print(f"{'='*80}")
        
        comparisons = []
        for session_id in session_ids:
            analysis = self.analyze_session(session_id)
            if analysis:
                comparisons.append((session_id, analysis))
        
        if not comparisons:
            print("No valid sessions found for comparison")
            return
        
        # Print comparison table
        print(f"\n{'Session':<15} {'Particles':<10} {'Resolution':<12} {'Avg FPS':<10} {'Efficiency':<12} {'GPU':<5}")
        print("-" * 80)
        
        for session_id, analysis in comparisons:
            final_stats = analysis["final_stats"]
            config = final_stats.get("configuration", {})
            fps_mean = analysis["performance_metrics"]["fps"]["mean"]
            eff_mean = analysis["performance_metrics"]["efficiency"]["mean"]
            
            particles = config.get("num_particles", "Unknown")
            resolution = str(config.get("grid_resolution", "Unknown")).replace(" ", "")
            gpu = "Yes" if config.get("use_gpu", False) else "No"
            
            print(f"{session_id:<15} {particles:<10} {resolution:<12} {fps_mean:<10.1f} {eff_mean:<12.1%} {gpu:<5}")

def main():
    parser = argparse.ArgumentParser(description="DWARF Simulation Performance Analyzer")
    parser.add_argument("--log-dir", type=str, default="logs_optimized",
                        help="Directory containing simulation logs")
    parser.add_argument("--session", type=str,
                        help="Specific session to analyze")
    parser.add_argument("--list", action="store_true",
                        help="List available sessions")
    parser.add_argument("--compare", nargs="+",
                        help="Compare multiple sessions")
    parser.add_argument("--plot", action="store_true",
                        help="Generate performance plots")
    parser.add_argument("--all", action="store_true",
                        help="Analyze all available sessions")
    
    args = parser.parse_args()
    
    analyzer = PerformanceAnalyzer(args.log_dir)
    
    if args.list or not any([args.session, args.compare, args.all]):
        print("Available sessions:")
        for session in analyzer.sessions:
            print(f"  {session}")
        return
    
    if args.compare:
        analyzer.compare_sessions(args.compare)
        return
    
    if args.all:
        for session in analyzer.sessions:
            analyzer.print_session_summary(session)
            if args.plot:
                analyzer.generate_performance_plots(session)
        return
    
    if args.session:
        analyzer.print_session_summary(args.session)
        if args.plot:
            analyzer.generate_performance_plots(args.session)

if __name__ == "__main__":
    main()