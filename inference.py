# inferece benchmarking script for CFM model. usage: python inference.py --model-path /path/to/model.pt --config-path /path/to/config.yaml --source-file /path/to/data --n-objects 1000 --batch-size 32 --device cuda --n-model-instances 5 --num-threads 4
import yaml
import time
import os
import sys
import numpy as np
import torch
from tqdm import tqdm
import argparse
from torchdiffeq import odeint
from data.data_preprocessing import TestDataPreprocessor
from src.create_cfm_model import build_cfm_model, resume_cfm_model
from src.modded_cfm import ModelWrapper
import psutil
import pynvml
import threading
from contextlib import contextmanager
import json  # Add this import at the top

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config-path', type=str, required=True, help='Path to config file')
    parser.add_argument('--source-file', type=str, required=True, help='Path to source data')
    parser.add_argument('--n-objects', type=int, required=True, help='Number of objects to process')
    parser.add_argument('--batch-size', type=int, required=True, help='Batch size for inference')
    parser.add_argument('--device', type=str, required=True, help='Device to run on (cpu/cuda)')
    parser.add_argument('--n-model-instances', type=int, required=True, help='Number of model instances to run')
    parser.add_argument('--gpu-memory-limit', type=float, help='GPU memory limit in GB')
    parser.add_argument('--num-threads', type=int, help='Number of PyTorch threads to use')
    parser.add_argument('--monitor-interval', type=float, default=1.0, help='Resource monitoring interval in seconds')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU device ID to use (default: 0)')
    parser.add_argument('--output-json', type=str, help='Path to save benchmark results as JSON')
    return parser.parse_args()

class ResourceMonitor:
    def __init__(self, interval=1.0, gpu_id=0):
        self.interval = interval
        self.running = False
        self.stats = {'cpu': [], 'ram': [], 'gpu_util': [], 'gpu_mem': []}
        self.gpu_id = gpu_id
        
        # Initialize NVML for GPU monitoring
        try:
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
            self.has_gpu = True
        except:
            self.has_gpu = False
    
    def start(self):
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor)
        self.monitor_thread.start()
    
    def stop(self):
        self.running = False
        self.monitor_thread.join()
        if self.has_gpu:
            pynvml.nvmlShutdown()
    
    def _monitor(self):
        while self.running:
            # CPU and RAM
            self.stats['cpu'].append(psutil.cpu_percent(interval=None))
            self.stats['ram'].append(psutil.virtual_memory().percent)
            
            # GPU
            if self.has_gpu:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                    self.stats['gpu_util'].append(util.gpu)
                    self.stats['gpu_mem'].append(mem.used / mem.total * 100)
                except:
                    pass
            
            time.sleep(self.interval)
    
    def get_summary(self):
        summary = {
            'cpu_mean': np.mean(self.stats['cpu']),
            'cpu_max': np.max(self.stats['cpu']),
            'ram_mean': np.mean(self.stats['ram']),
            'ram_max': np.max(self.stats['ram'])
        }
        if self.has_gpu and len(self.stats['gpu_util']) > 0:
            summary.update({
                'gpu_util_mean': np.mean(self.stats['gpu_util']),
                'gpu_util_max': np.max(self.stats['gpu_util']),
                'gpu_mem_mean': np.mean(self.stats['gpu_mem']),
                'gpu_mem_max': np.max(self.stats['gpu_mem'])
            })
        return summary

def setup_resource_limits(args):
    # Set PyTorch thread count if specified
    if args.num_threads:
        torch.set_num_threads(args.num_threads)
    
    # Set GPU device and memory limit if specified
    if args.device == 'cuda':
        if torch.cuda.is_available():
            torch.cuda.set_device(args.gpu_id)
            if args.gpu_memory_limit:
                torch.cuda.set_per_process_memory_fraction(args.gpu_memory_limit)
        else:
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            args.device = 'cpu'

def load_data(source_file, n_objects, data_kwargs):
    # Load only the required test data
    # change the data_kwargs["N_test"] to n_objects
    # change the data_kwargs["dataset_path"] to source_file
    data_kwargs["N_test"] = n_objects
    data_kwargs["dataset_path"] = source_file
    test_dataset = TestDataPreprocessor(data_kwargs, allow_nans=True)
    X_test, Y_test = test_dataset.get_dataset()
    
    # Take only n_objects
    X_test = X_test[:n_objects]
    Y_test = Y_test[:n_objects]
    
    return X_test, Y_test

def format_benchmark_results(total_objects, total_time, resource_stats, device, num_threads, gpu_memory_limit=None):
    results = {
        "performance": {
            "total_objects": total_objects,
            "total_time_seconds": total_time,
            "throughput_objects_per_second": total_objects / total_time
        },
        "resource_usage": {
            "cpu": {
                "average_percent": float(f"{resource_stats['cpu_mean']:.1f}"),
                "peak_percent": float(f"{resource_stats['cpu_max']:.1f}")
            },
            "ram": {
                "average_percent": float(f"{resource_stats['ram_mean']:.1f}"),
                "peak_percent": float(f"{resource_stats['ram_max']:.1f}")
            }
        },
        "configuration": {
            "device": str(device),
            "num_threads": num_threads
        }
    }
    
    if gpu_memory_limit:
        results["configuration"]["gpu_memory_limit_gb"] = gpu_memory_limit
    
    if 'gpu_util_mean' in resource_stats:
        results["resource_usage"]["gpu"] = {
            "utilization_average_percent": float(f"{resource_stats['gpu_util_mean']:.1f}"),
            "utilization_peak_percent": float(f"{resource_stats['gpu_util_max']:.1f}"),
            "memory_average_percent": float(f"{resource_stats['gpu_mem_mean']:.1f}"),
            "memory_peak_percent": float(f"{resource_stats['gpu_mem_max']:.1f}")
        }
    
    return results

def run_benchmark(args, config):
    # Setup resource limits
    setup_resource_limits(args)
    
    # Initialize resource monitor with specified GPU
    monitor = ResourceMonitor(interval=args.monitor_interval, gpu_id=args.gpu_id)
    monitor.start()
    
    device = torch.device(f"{args.device}:{args.gpu_id}" if args.device == 'cuda' else args.device)
    total_time = 0
    total_objects = 0
    
    print(f"Running benchmark with {args.n_model_instances} model instances on {device}")
    print(f"PyTorch threads: {torch.get_num_threads()}")
    if args.gpu_memory_limit:
        print(f"GPU memory limit: {args.gpu_memory_limit:.1f} GB")
    
    try:
        for instance in range(args.n_model_instances):
            # Clear cache
            if device.type == "cuda":
                torch.cuda.empty_cache()
            
            # Load model
            model, _, _ = resume_cfm_model(os.path.dirname(args.model_path), os.path.basename(args.model_path))
            model = model.to(device)
            
            # Load data
            X_test, Y_test = load_data(args.source_file, args.n_objects, config["data_kwargs"])
            X_test = torch.tensor(X_test).float().to(device)
            Y_test = torch.tensor(Y_test).float().to(device)
            
            # Setup sampler
            sampler = ModelWrapper(model, context_dim=config["context_dim"])
            t_span = torch.linspace(0, 1, config["base_kwargs"]["cfm"]["timesteps"]).to(device)
            
            # Run inference
            start_time = time.time()
            samples_list = []
            
            with torch.no_grad():
                for i in range(0, len(X_test), args.batch_size):
                    Y_batch = Y_test[i:i + args.batch_size]
                    x0_sample = torch.randn(len(Y_batch), X_test.shape[1]).to(device)
                    initial_conditions = torch.cat([x0_sample, Y_batch], dim=-1)
                    
                    samples = odeint(
                        sampler,
                        initial_conditions,
                        t_span,
                        atol=1e-5,
                        rtol=1e-5,
                        method="dopri5"
                    )[-1, :, :X_test.shape[1]]
                    
                    samples_list.append(samples.cpu().numpy())
            
            instance_time = time.time() - start_time
            total_time += instance_time
            total_objects += len(X_test)
            
            print(f"Instance {instance+1}: Processed {len(X_test)} objects in {instance_time:.2f}s")
            
            # Clean up
            del model, sampler, X_test, Y_test, samples_list
            if device.type == "cuda":
                torch.cuda.empty_cache()
        
        throughput = total_objects / total_time
        print(f"\nTotal throughput: {throughput:.2f} objects/second")
        print(f"Total time: {total_time:.2f}s")
        print(f"Total objects: {total_objects}")
        
        # Add resource monitoring summary
        resource_stats = monitor.get_summary()
        print("\nResource Usage Summary:")
        print(f"CPU Average: {resource_stats['cpu_mean']:.1f}%")
        print(f"CPU Peak: {resource_stats['cpu_max']:.1f}%")
        print(f"RAM Average: {resource_stats['ram_mean']:.1f}%")
        print(f"RAM Peak: {resource_stats['ram_max']:.1f}%")
        if 'gpu_util_mean' in resource_stats:
            print(f"GPU Utilization Average: {resource_stats['gpu_util_mean']:.1f}%")
            print(f"GPU Utilization Peak: {resource_stats['gpu_util_max']:.1f}%")
            print(f"GPU Memory Average: {resource_stats['gpu_mem_mean']:.1f}%")
            print(f"GPU Memory Peak: {resource_stats['gpu_mem_max']:.1f}%")
        
        # Format and save results as JSON if requested
        if args.output_json:
            results = format_benchmark_results(
                total_objects=total_objects,
                total_time=total_time,
                resource_stats=resource_stats,
                device=device,
                num_threads=torch.get_num_threads(),
                gpu_memory_limit=args.gpu_memory_limit
            )
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
            
            # Save to JSON file
            with open(args.output_json, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nBenchmark results saved to: {args.output_json}")
    
    finally:
        monitor.stop()

def main():
    args = parse_args()
    
    with open(args.config_path, "r") as stream:
        config = yaml.safe_load(stream)
    
    run_benchmark(args, config)

if __name__ == "__main__":
    main()
