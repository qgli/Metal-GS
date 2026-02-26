"""
Metal-GS Kernel Profiler

Collects per-kernel timing from the Metal rendering pipeline.
The C++ layer already returns elapsed_ms for each kernel — this module
captures, accumulates, and reports those timings.

Usage:
    from metal_gs.profiler import kernel_profiler
    kernel_profiler.enable()
    # ... run training iterations ...
    kernel_profiler.print_summary()
"""

import time
import numpy as np
from collections import defaultdict


class KernelProfiler:
    """
    Global singleton that collects per-kernel GPU timing data.
    
    Forward pass stages:
      - numpy_convert:  CPU→numpy conversion (.detach().cpu().numpy())
      - sh_forward:     SH evaluation (Metal GPU)
      - preprocess:     3D→2D projection + frustum culling (Metal GPU)
      - depth_sort:     Radix sort by depth (Metal GPU)
      - tile_binning:   Assign Gaussians to screen tiles (Metal GPU + CPU prefix sum)
      - rasterize:      Per-tile alpha blending (Metal GPU)
      - torch_convert:  numpy→torch + device transfer
    
    Backward pass stages:
      - numpy_convert_bw:  grad→numpy conversion
      - rasterize_bw:      Rasterize backward (Metal GPU)
      - preprocess_bw:     Preprocess backward (Metal GPU)
      - sh_bw:             SH backward (Metal GPU)
      - torch_convert_bw:  numpy grads→torch + device transfer
    """
    
    def __init__(self):
        self._enabled = False
        self._records = []  # list of dicts, one per iteration
        self._current = None
        
    def enable(self):
        self._enabled = True
        self._records = []
        self._current = None
    
    def disable(self):
        self._enabled = False
        
    @property
    def enabled(self):
        return self._enabled
    
    def begin_forward(self):
        """Call at start of forward pass."""
        if not self._enabled:
            return
        self._current = {"pass": "forward"}
    
    def begin_backward(self):
        """Call at start of backward pass."""
        if not self._enabled:
            return
        self._current = {"pass": "backward"}
    
    def record(self, key: str, elapsed_ms: float):
        """Record a timing measurement."""
        if not self._enabled or self._current is None:
            return
        self._current[key] = elapsed_ms
    
    def end_pass(self):
        """Finalize and store the current pass record."""
        if not self._enabled or self._current is None:
            return
        self._records.append(self._current)
        self._current = None
    
    def record_forward_stats(self, num_gaussians: int, num_visible: int, 
                              num_intersections: int):
        """Record forward pass scene statistics."""
        if not self._enabled or self._current is None:
            return
        self._current["num_gaussians"] = num_gaussians
        self._current["num_visible"] = num_visible
        self._current["num_intersections"] = num_intersections
    
    @property
    def forward_records(self):
        return [r for r in self._records if r.get("pass") == "forward"]
    
    @property
    def backward_records(self):
        return [r for r in self._records if r.get("pass") == "backward"]
    
    def get_summary(self, warmup=10):
        """
        Compute summary statistics, skipping first `warmup` iterations.
        
        Returns dict with forward and backward timing breakdowns.
        """
        fwd = self.forward_records
        bwd = self.backward_records
        
        # Skip warmup
        fwd = fwd[warmup:] if len(fwd) > warmup else fwd
        bwd = bwd[warmup:] if len(bwd) > warmup else bwd
        
        summary = {}
        
        # Forward stages
        fwd_stages = [
            ("numpy_convert", "CPU→Numpy"),
            ("sh_forward", "SH Forward (GPU)"),
            ("preprocess", "Preprocess (GPU)"),
            ("depth_sort", "Depth Sort (GPU)"),
            ("tile_binning", "Tile Binning (GPU+CPU)"),
            ("rasterize", "Rasterize (GPU)"),
            ("torch_convert", "Numpy→Torch"),
        ]
        
        fwd_data = {}
        for key, label in fwd_stages:
            values = [r.get(key, 0.0) for r in fwd]
            if values:
                fwd_data[key] = {
                    "label": label,
                    "mean_ms": np.mean(values),
                    "std_ms": np.std(values),
                    "min_ms": np.min(values),
                    "max_ms": np.max(values),
                    "median_ms": np.median(values),
                }
        summary["forward"] = fwd_data
        
        # Backward stages
        bwd_stages = [
            ("numpy_convert_bw", "CPU→Numpy (grad)"),
            ("rasterize_bw", "Rasterize BW (GPU)"),
            ("preprocess_bw", "Preprocess BW (GPU)"),
            ("sh_bw", "SH Backward (GPU)"),
            ("torch_convert_bw", "Numpy→Torch (grad)"),
        ]
        
        bwd_data = {}
        for key, label in bwd_stages:
            values = [r.get(key, 0.0) for r in bwd]
            if values:
                bwd_data[key] = {
                    "label": label,
                    "mean_ms": np.mean(values),
                    "std_ms": np.std(values),
                    "min_ms": np.min(values),
                    "max_ms": np.max(values),
                    "median_ms": np.median(values),
                }
        summary["backward"] = bwd_data
        
        # Scene statistics
        if fwd:
            summary["avg_gaussians"] = np.mean([r.get("num_gaussians", 0) for r in fwd])
            summary["avg_visible"] = np.mean([r.get("num_visible", 0) for r in fwd])
            summary["avg_intersections"] = np.mean([r.get("num_intersections", 0) for r in fwd])
        
        summary["num_iterations"] = len(fwd)
        summary["warmup_skipped"] = warmup
        
        return summary
    
    def print_summary(self, warmup=10):
        """Print a formatted summary table to stdout."""
        s = self.get_summary(warmup=warmup)
        
        print(f"\n{'='*72}")
        print(f"  METAL KERNEL PROFILING SUMMARY")
        print(f"  ({s['num_iterations']} iterations, {s['warmup_skipped']} warmup skipped)")
        print(f"{'='*72}")
        
        if "avg_gaussians" in s:
            print(f"\n  Scene Stats:")
            print(f"    Avg Gaussians:     {s['avg_gaussians']:,.0f}")
            print(f"    Avg Visible:       {s['avg_visible']:,.0f} "
                  f"({100*s['avg_visible']/max(s['avg_gaussians'],1):.1f}%)")
            print(f"    Avg Intersections: {s['avg_intersections']:,.0f}")
        
        # Forward
        fwd = s.get("forward", {})
        if fwd:
            fwd_total = sum(v["mean_ms"] for v in fwd.values())
            print(f"\n  {'─'*68}")
            print(f"  FORWARD PASS  (total: {fwd_total:.2f} ms)")
            print(f"  {'─'*68}")
            print(f"  {'Stage':<28} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'%':>6}")
            print(f"  {'─'*68}")
            for key, v in fwd.items():
                pct = 100 * v["mean_ms"] / max(fwd_total, 0.001)
                bar = '█' * int(pct / 2.5)
                print(f"  {v['label']:<28} {v['mean_ms']:>7.2f}  {v['std_ms']:>7.2f}"
                      f"  {v['min_ms']:>7.2f}  {v['max_ms']:>7.2f}  {pct:>5.1f}%  {bar}")
            print(f"  {'─'*68}")
            print(f"  {'TOTAL':<28} {fwd_total:>7.2f} ms")
        
        # Backward
        bwd = s.get("backward", {})
        if bwd:
            bwd_total = sum(v["mean_ms"] for v in bwd.values())
            print(f"\n  {'─'*68}")
            print(f"  BACKWARD PASS  (total: {bwd_total:.2f} ms)")
            print(f"  {'─'*68}")
            print(f"  {'Stage':<28} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'%':>6}")
            print(f"  {'─'*68}")
            for key, v in bwd.items():
                pct = 100 * v["mean_ms"] / max(bwd_total, 0.001)
                bar = '█' * int(pct / 2.5)
                print(f"  {v['label']:<28} {v['mean_ms']:>7.2f}  {v['std_ms']:>7.2f}"
                      f"  {v['min_ms']:>7.2f}  {v['max_ms']:>7.2f}  {pct:>5.1f}%  {bar}")
            print(f"  {'─'*68}")
            print(f"  {'TOTAL':<28} {bwd_total:>7.2f} ms")
        
        # Combined
        if fwd and bwd:
            fwd_total = sum(v["mean_ms"] for v in fwd.values())
            bwd_total = sum(v["mean_ms"] for v in bwd.values())
            iteration_total = fwd_total + bwd_total
            print(f"\n  {'─'*68}")
            print(f"  ITERATION TOTAL: {iteration_total:.2f} ms  "
                  f"({1000/max(iteration_total,0.001):.1f} theoretical it/s)")
            print(f"    Forward:  {fwd_total:.2f} ms ({100*fwd_total/iteration_total:.1f}%)")
            print(f"    Backward: {bwd_total:.2f} ms ({100*bwd_total/iteration_total:.1f}%)")
            
            # Top bottlenecks across both passes
            all_stages = []
            for k, v in fwd.items():
                all_stages.append((v["label"], v["mean_ms"], "fwd"))
            for k, v in bwd.items():
                all_stages.append((v["label"], v["mean_ms"], "bwd"))
            all_stages.sort(key=lambda x: -x[1])
            
            print(f"\n  TOP BOTTLENECKS:")
            for i, (label, ms, pass_type) in enumerate(all_stages[:5]):
                pct = 100 * ms / max(iteration_total, 0.001)
                print(f"    #{i+1}  {label:<28} {ms:>7.2f} ms  ({pct:.1f}%)  [{pass_type}]")
        
        print(f"{'='*72}\n")
    
    def to_dict(self):
        """Export all raw records for further analysis."""
        return {
            "records": self._records,
            "summary": self.get_summary(),
        }


# Global singleton
kernel_profiler = KernelProfiler()
