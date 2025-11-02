"""
Performance Benchmark Suite for LEXA GPU Optimizations

Measures:
- Response time (mean, p95, p99)
- Throughput (tokens/sec)
- GPU memory efficiency
- Early-exit effectiveness
- Power consumption (if available)

File: tests/benchmark.py
"""

import torch
import time
import numpy as np
import argparse
import logging
from typing import List, Dict
import json

import sys
sys.path.append('..')
from inference.engine import OptimizedInferenceEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LEXABenchmark:
    """Comprehensive benchmarking suite for LEXA"""
    
    def __init__(
        self,
        device: str = "cuda:0",
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    ):
        self.device = device
        self.model_name = model_name
        
        # Test prompts of varying lengths
        self.prompts = [
            "What is machine learning?",
            "Explain deep learning in simple terms.",
            "How do neural networks work? Describe the process step by step.",
            "What are transformers in AI?",
            "Define reinforcement learning.",
            "Explain gradient descent optimization.",
            "What is backpropagation?",
            "How does attention mechanism work?",
            "Describe convolutional neural networks.",
            "What is transfer learning?"
        ]
    
    def warmup(self, engine: OptimizedInferenceEngine, runs: int = 3):
        """Warmup GPU and cache"""
        logger.info("Running warmup...")
        for i in range(runs):
            engine.generate(
                self.prompts[0],
                max_new_tokens=10,
                enable_early_exit=False,
                profile=False
            )
        logger.info("Warmup complete")
    
    def benchmark_latency(
        self,
        engine: OptimizedInferenceEngine,
        runs: int = 100
    ) -> Dict:
        """Benchmark response latency"""
        logger.info(f"Benchmarking latency ({runs} runs)...")
        
        latencies = []
        token_counts = []
        
        for i in range(runs):
            prompt = self.prompts[i % len(self.prompts)]
            
            start = time.time()
            result = engine.generate(
                prompt,
                max_new_tokens=50,
                temperature=0.7,
                enable_early_exit=True,
                profile=False
            )
            latency = time.time() - start
            
            latencies.append(latency)
            token_counts.append(result['stats']['generated_tokens'])
            
            if (i + 1) % 20 == 0:
                logger.info(f"Progress: {i+1}/{runs}")
        
        return {
            "mean_latency_s": np.mean(latencies),
            "std_latency_s": np.std(latencies),
            "min_latency_s": np.min(latencies),
            "max_latency_s": np.max(latencies),
            "p50_latency_s": np.percentile(latencies, 50),
            "p95_latency_s": np.percentile(latencies, 95),
            "p99_latency_s": np.percentile(latencies, 99),
            "mean_tokens": np.mean(token_counts),
            "total_runs": runs
        }
    
    def benchmark_throughput(
        self,
        engine: OptimizedInferenceEngine,
        runs: int = 50
    ) -> Dict:
        """Benchmark token generation throughput"""
        logger.info(f"Benchmarking throughput ({runs} runs)...")
        
        throughputs = []
        
        for i in range(runs):
            prompt = self.prompts[i % len(self.prompts)]
            
            result = engine.generate(
                prompt,
                max_new_tokens=50,
                temperature=0.7,
                enable_early_exit=True,
                profile=False
            )
            
            throughputs.append(result['stats']['tokens_per_sec'])
            
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i+1}/{runs}")
        
        return {
            "mean_tokens_per_sec": np.mean(throughputs),
            "std_tokens_per_sec": np.std(throughputs),
            "min_tokens_per_sec": np.min(throughputs),
            "max_tokens_per_sec": np.max(throughputs),
            "p95_tokens_per_sec": np.percentile(throughputs, 95)
        }
    
    def benchmark_early_exit(
        self,
        engine: OptimizedInferenceEngine,
        runs: int = 50
    ) -> Dict:
        """Benchmark early-exit effectiveness"""
        logger.info(f"Benchmarking early-exit ({runs} runs)...")
        
        # Reset stats
        engine.reset_stats()
        
        # Run with early-exit
        for i in range(runs):
            prompt = self.prompts[i % len(self.prompts)]
            engine.generate(
                prompt,
                max_new_tokens=50,
                enable_early_exit=True,
                profile=False
            )
        
        # Get early-exit stats
        stats = engine.get_stats()
        early_exit_stats = stats.get('early_exit', {})
        
        return {
            "exit_rate": early_exit_stats.get('exit_rate', 0),
            "tokens_saved_pct": early_exit_stats.get('tokens_saved_pct', 0),
            "avg_exit_step": early_exit_stats.get('avg_exit_step', 0),
            "estimated_speedup": early_exit_stats.get('estimated_speedup', 1.0)
        }
    
    def benchmark_memory(
        self,
        engine: OptimizedInferenceEngine,
        runs: int = 50
    ) -> Dict:
        """Benchmark memory efficiency"""
        logger.info(f"Benchmarking memory ({runs} runs)...")
        
        # Reset memory stats
        engine.memory_manager.reset_stats()
        
        # Run inference
        for i in range(runs):
            prompt = self.prompts[i % len(self.prompts)]
            engine.generate(
                prompt,
                max_new_tokens=50,
                enable_early_exit=True,
                profile=False
            )
        
        # Get memory stats
        mem_stats = engine.memory_manager.get_stats()
        gpu_mem = torch.cuda.memory_stats(engine.device)
        
        return {
            "cache_hit_rate": mem_stats['cache_hit_rate'],
            "avg_alloc_time_ms": mem_stats['avg_alloc_time_ms'],
            "peak_allocated_mb": mem_stats['peak_allocated_mb'],
            "utilization": mem_stats['utilization'],
            "total_blocks": mem_stats['total_blocks'],
            "cuda_peak_mb": torch.cuda.max_memory_allocated(engine.device) / 1024 / 1024
        }
    
    def benchmark_profiling(
        self,
        engine: OptimizedInferenceEngine,
        runs: int = 20
    ) -> Dict:
        """Detailed profiling benchmark"""
        logger.info(f"Profiling detailed metrics ({runs} runs)...")
        
        # Reset profiler
        engine.profiler.reset()
        engine.kernel_scheduler.reset_stats()
        
        # Run with profiling enabled
        for i in range(runs):
            prompt = self.prompts[i % len(self.prompts)]
            engine.generate(
                prompt,
                max_new_tokens=50,
                enable_early_exit=True,
                profile=True
            )
        
        # Get profiling data
        prof_stats = engine.profiler.get_summary()
        kernel_stats = engine.kernel_scheduler.get_kernel_stats()
        
        # Aggregate kernel times
        kernel_summary = {}
        for name, stats in kernel_stats.items():
            if 'forward_step' in name:
                if 'forward_passes' not in kernel_summary:
                    kernel_summary['forward_passes'] = {
                        'total_calls': 0,
                        'total_time_ms': 0,
                        'avg_time_ms': []
                    }
                kernel_summary['forward_passes']['total_calls'] += stats['calls']
                kernel_summary['forward_passes']['total_time_ms'] += stats['total_time_ms']
                kernel_summary['forward_passes']['avg_time_ms'].append(stats['avg_time_ms'])
        
        # Calculate averages
        if 'forward_passes' in kernel_summary:
            kernel_summary['forward_passes']['avg_time_ms'] = np.mean(
                kernel_summary['forward_passes']['avg_time_ms']
            )
        
        return {
            "avg_kernel_latency_ms": prof_stats.get('avg_kernel_latency_ms', 0),
            "total_time_ms": prof_stats.get('total_time_ms', 0),
            "gpu_memory_mb": prof_stats['gpu_memory'].get('allocated_mb', 0),
            "kernel_summary": kernel_summary
        }
    
    def compare_baseline_vs_optimized(
        self,
        runs: int = 50
    ) -> Dict:
        """Compare baseline vs optimized performance"""
        logger.info("Running baseline vs optimized comparison...")
        
        # Baseline (no optimizations)
        logger.info("Testing baseline (no early-exit, no profiling)...")
        baseline_engine = OptimizedInferenceEngine(
            model_name=self.model_name,
            device=self.device,
            enable_early_exit=False,
            enable_profiling=False
        )
        
        self.warmup(baseline_engine)
        
        baseline_times = []
        for i in range(runs):
            prompt = self.prompts[i % len(self.prompts)]
            start = time.time()
            baseline_engine.generate(
                prompt,
                max_new_tokens=50,
                enable_early_exit=False
            )
            baseline_times.append(time.time() - start)
        
        baseline_mean = np.mean(baseline_times)
        
        # Optimized (with all optimizations)
        logger.info("Testing optimized (early-exit + profiling)...")
        optimized_engine = OptimizedInferenceEngine(
            model_name=self.model_name,
            device=self.device,
            enable_early_exit=True,
            enable_profiling=True
        )
        
        self.warmup(optimized_engine)
        
        optimized_times = []
        for i in range(runs):
            prompt = self.prompts[i % len(self.prompts)]
            start = time.time()
            optimized_engine.generate(
                prompt,
                max_new_tokens=50,
                enable_early_exit=True
            )
            optimized_times.append(time.time() - start)
        
        optimized_mean = np.mean(optimized_times)
        
        improvement_pct = (baseline_mean - optimized_mean) / baseline_mean * 100
        
        return {
            "baseline_mean_s": baseline_mean,
            "optimized_mean_s": optimized_mean,
            "improvement_pct": improvement_pct,
            "baseline_p95_s": np.percentile(baseline_times, 95),
            "optimized_p95_s": np.percentile(optimized_times, 95),
            "speedup_factor": baseline_mean / optimized_mean
        }
    
    def run_full_benchmark(self, runs: int = 100) -> Dict:
        """Run complete benchmark suite"""
        logger.info("="*60)
        logger.info("LEXA GPU Optimization Benchmark Suite")
        logger.info("="*60)
        
        # Initialize engine
        engine = OptimizedInferenceEngine(
            model_name=self.model_name,
            device=self.device,
            enable_early_exit=True,
            enable_profiling=True
        )
        
        # Warmup
        self.warmup(engine)
        
        results = {}
        
        # 1. Latency benchmark
        results['latency'] = self.benchmark_latency(engine, runs)
        
        # 2. Throughput benchmark
        results['throughput'] = self.benchmark_throughput(engine, runs // 2)
        
        # 3. Early-exit benchmark
        results['early_exit'] = self.benchmark_early_exit(engine, runs // 2)
        
        # 4. Memory benchmark
        results['memory'] = self.benchmark_memory(engine, runs // 2)
        
        # 5. Profiling benchmark
        results['profiling'] = self.benchmark_profiling(engine, runs // 5)
        
        # 6. Baseline comparison
        results['comparison'] = self.compare_baseline_vs_optimized(runs // 2)
        
        # Cleanup
        engine.cleanup()
        
        return results
    
    def print_results(self, results: Dict):
        """Print formatted benchmark results"""
        logger.info("\n" + "="*60)
        logger.info("BENCHMARK RESULTS")
        logger.info("="*60)
        
        # Latency
        logger.info("\n--- Latency Metrics ---")
        lat = results['latency']
        logger.info(f"Mean latency:    {lat['mean_latency_s']:.3f}s")
        logger.info(f"Std latency:     {lat['std_latency_s']:.3f}s")
        logger.info(f"P50 latency:     {lat['p50_latency_s']:.3f}s")
        logger.info(f"P95 latency:     {lat['p95_latency_s']:.3f}s")
        logger.info(f"P99 latency:     {lat['p99_latency_s']:.3f}s")
        
        # Throughput
        logger.info("\n--- Throughput Metrics ---")
        thr = results['throughput']
        logger.info(f"Mean throughput: {thr['mean_tokens_per_sec']:.1f} tokens/s")
        logger.info(f"P95 throughput:  {thr['p95_tokens_per_sec']:.1f} tokens/s")
        
        # Early-exit
        logger.info("\n--- Early-Exit Effectiveness ---")
        ee = results['early_exit']
        logger.info(f"Exit rate:       {ee['exit_rate']:.2%}")
        logger.info(f"Tokens saved:    {ee['tokens_saved_pct']:.1%}")
        logger.info(f"Estimated speedup: {ee['estimated_speedup']:.2f}x")
        
        # Memory
        logger.info("\n--- Memory Efficiency ---")
        mem = results['memory']
        logger.info(f"Cache hit rate:  {mem['cache_hit_rate']:.2%}")
        logger.info(f"Peak memory:     {mem['peak_allocated_mb']:.1f} MB")
        logger.info(f"Utilization:     {mem['utilization']:.2%}")
        
        # Comparison
        logger.info("\n--- Baseline vs Optimized ---")
        comp = results['comparison']
        logger.info(f"Baseline mean:   {comp['baseline_mean_s']:.3f}s")
        logger.info(f"Optimized mean:  {comp['optimized_mean_s']:.3f}s")
        logger.info(f"Improvement:     {comp['improvement_pct']:.1f}%")
        logger.info(f"Speedup:         {comp['speedup_factor']:.2f}x")
        
        logger.info("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="LEXA GPU Optimization Benchmark")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device")
    parser.add_argument("--runs", type=int, default=100, help="Number of benchmark runs")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output file")
    
    args = parser.parse_args()
    
    # Check CUDA
    if not torch.cuda.is_available():
        logger.error("CUDA not available!")
        return
    
    logger.info(f"Using device: {args.device}")
    logger.info(f"GPU: {torch.cuda.get_device_name(args.device)}")
    
    # Run benchmark
    benchmark = LEXABenchmark(device=args.device)
    results = benchmark.run_full_benchmark(runs=args.runs)
    
    # Print results
    benchmark.print_results(results)
    
    # Save to file
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
