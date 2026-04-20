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
import warnings
import json  # Add this import at the top
import torch.utils.collect_env
try:
    import rocm_smi  # For AMD GPUs
    HAS_ROCM = True
except ImportError:
    HAS_ROCM = False

def get_available_devices():
    """Return list of available compute devices with their indices"""
    devices = {'cpu': [0]}  # CPU is always available
    
    # Check CUDA devices
    if torch.cuda.is_available():
        devices['cuda'] = list(range(torch.cuda.device_count()))
    
    # Check ROCm devices
    if HAS_ROCM:
        try:
            devices['rocm'] = list(range(rocm_smi.getDeviceCount()))
        except:
            pass
    
    # Check Vulkan devices
    if hasattr(torch, 'vulkan') and torch.vulkan.is_available():
        try:
            devices['vulkan'] = list(range(torch.vulkan.device_count()))
        except:
            pass
            
    return devices

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
    parser.add_argument('--list-devices', action='store_true', help='List available compute devices and exit')
    parser.add_argument('--progress-bar', action='store_true', help='Show progress bar for sampling')
    return parser.parse_args()

class ResourceMonitor:
    def __init__(self, interval=1.0, gpu_id=0):
        self.interval = interval
        self.running = False
        self.stats = {'cpu': [], 'ram': [], 'gpu_util': [], 'gpu_mem': []}
        self.gpu_id = gpu_id
        
        # Platform-agnostic GPU detection
        self.gpu_backend = None
        if torch.cuda.is_available():
            self.gpu_backend = 'cuda'
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
                self.has_gpu = True
            except:
                self.has_gpu = False
        elif HAS_ROCM:
            self.gpu_backend = 'rocm'
            try:
                rocm_smi.rocm_init()
                self.has_gpu = True
            except:
                self.has_gpu = False
        else:
            self.has_gpu = False
    
    def start(self):
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor)
        self.monitor_thread.start()
    
    def stop(self):
        self.running = False
        self.monitor_thread.join()
        if self.has_gpu:
            if self.gpu_backend == 'cuda':
                pynvml.nvmlShutdown()
            elif self.gpu_backend == 'rocm':
                rocm_smi.rocm_shutdown()
    
    def _monitor(self):
        while self.running:
            # CPU and RAM
            self.stats['cpu'].append(psutil.cpu_percent(interval=None))
            self.stats['ram'].append(psutil.virtual_memory().percent)
            
            # GPU monitoring based on backend
            if self.has_gpu:
                try:
                    if self.gpu_backend == 'cuda':
                        util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                        mem = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                        self.stats['gpu_util'].append(util.gpu)
                        self.stats['gpu_mem'].append(mem.used / mem.total * 100)
                    elif self.gpu_backend == 'rocm':
                        util = rocm_smi.getGpuUse(self.gpu_id)
                        mem = rocm_smi.getMemInfo(self.gpu_id)
                        self.stats['gpu_util'].append(util)
                        self.stats['gpu_mem'].append(mem['used'] / mem['total'] * 100)
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

    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

def setup_resource_limits(args):
    # Set PyTorch thread count if specified
    if args.num_threads:
        torch.set_num_threads(args.num_threads)
    
    # Platform-agnostic device setup
    if args.device.startswith(('cuda', 'rocm', 'vulkan')):
        backend = args.device.split(':')[0] if ':' in args.device else args.device
        if not hasattr(torch, backend) or not getattr(torch, backend).is_available():
            print(f"Warning: {backend} requested but not available. Falling back to CPU.")
            args.device = 'cpu'
            return
        
        # Set device if supported
        if backend == 'cuda':
            torch.cuda.set_device(args.gpu_id)
            if args.gpu_memory_limit:
                torch.cuda.set_per_process_memory_fraction(args.gpu_memory_limit)
        # For other backends, device selection is handled through device string
        # e.g., 'vulkan:1' or 'rocm:0'

def load_data(source_file, n_objects, data_kwargs):
    # Load only the required test data
    # change the data_kwargs["N_test"] to n_objects
    # change the data_kwargs["dataset_path"] to source_file
    data_kwargs["N_test"] = n_objects
    data_kwargs["dataset_path"] = source_file
    
    # Suppress warnings when loading data
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        test_dataset = TestDataPreprocessor(data_kwargs, allow_nans=True)
        X_test, Y_test = test_dataset.get_dataset()
    
    # Take only n_objects, but if n_objects is greater than the dataset size, append the dataset to itself
    if n_objects > len(X_test):
        n_repeats = n_objects // len(X_test)
        X_test = np.tile(X_test, n_repeats + 1)[:n_objects]
        Y_test = np.tile(Y_test, n_repeats + 1)[:n_objects]
    else:
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
    
    total_time = 0
    total_objects = 0
    
    # Use context manager for ResourceMonitor
    with ResourceMonitor(interval=args.monitor_interval, gpu_id=args.gpu_id) as monitor:
        # Modified device selection for all backends
        device = torch.device(f"{args.device}:{args.gpu_id}" if args.device in ['cuda', 'rocm', 'vulkan'] else args.device)
        
        print(f"Running benchmark with {args.n_model_instances} model instances on {device}")
        print(f"PyTorch threads: {torch.get_num_threads()}")
        if args.gpu_memory_limit:
            print(f"GPU memory limit: {args.gpu_memory_limit:.1f} GB")
        
        # Rest of the benchmark code
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
                print(f"Loaded {len(X_test)} objects for inference")
                # X_test = torch.tensor(X_test).float().to(device)
                # Y_test = torch.tensor(Y_test).float().to(device)
                
                # Setup sampler
                sampler = ModelWrapper(model, context_dim=config["context_dim"])
                t_span = torch.linspace(0, 1, config["base_kwargs"]["cfm"]["timesteps"]).to(device)
                
                # Run inference
                start_time = time.time()
                samples_list = []
                
                with torch.no_grad():
                    n_batches = (len(X_test) + args.batch_size - 1) // args.batch_size
                    batch_iter = range(0, len(X_test), args.batch_size)
                    
                    if args.progress_bar:
                        batch_iter = tqdm(batch_iter, total=n_batches, desc=f"Instance {instance+1}", ascii=True)
                    
                    for i in batch_iter:
                        Y_batch = torch.tensor(Y_test[i:i + args.batch_size]).float()
                        x0_sample = torch.randn(len(Y_batch), X_test.shape[1])
                        initial_conditions = torch.cat([x0_sample, Y_batch], dim=-1).to(device)
                        
                        samples = odeint(
                            sampler,
                            initial_conditions,
                            t_span,
                            atol=1e-6,
                            rtol=1e-6,
                            method="euler"
                        )[-1, :, :X_test.shape[1]]
                        
                        samples_list.append(samples.cpu().numpy())
                        
                        if not args.progress_bar:  # Only print if progress bar is disabled
                            print(f"Batch {i // args.batch_size + 1}/{n_batches} processed")
                
                instance_time = time.time() - start_time
                total_time += instance_time
                total_objects += len(X_test)
                                
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
        
        except Exception as e:
            print(f"Error during benchmark: {str(e)}")
            raise
        
        finally:
            if 'model' in locals():
                del model
            if 'sampler' in locals():
                del sampler
            if device.type == "cuda":
                torch.cuda.empty_cache()

def main():
    args = parse_args()
    
    # Add device listing functionality
    if args.list_devices:
        devices = get_available_devices()
        print("\nAvailable compute devices:")
        for backend, indices in devices.items():
            print(f"{backend}: {indices}")
        sys.exit(0)
    
    # Validate device selection before proceeding
    devices = get_available_devices()
    if args.device != 'cpu':
        if args.device not in devices:
            print(f"Error: {args.device} backend not available")
            sys.exit(1)
        if args.gpu_id not in devices[args.device]:
            print(f"Error: Device index {args.gpu_id} not available for {args.device} backend")
            print(f"Available indices: {devices[args.device]}")
            sys.exit(1)
    
    with open(args.config_path, "r") as stream:
        config = yaml.safe_load(stream)
    
    run_benchmark(args, config)

if __name__ == "__main__":
    main()
