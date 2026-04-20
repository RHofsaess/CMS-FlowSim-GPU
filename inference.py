# Inference benchmarking script for CFM model.
# Usage:
# python inference.py --model-path /path/to/model.pt --config-path /path/to/config.yaml --source-file /path/to/data --n-objects 1000 --batch-size 32 --device cuda --n-model-instances 5 --num-threads 4

import argparse
import json
import logging
import os
import sys
import threading
import time
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import psutil
# pynvml API is provided by the `nvidia-ml-py` package (pip install nvidia-ml-py).
# The older `pynvml` package is deprecated — use nvidia-ml-py instead.
# Suppress FutureWarning from pynvml (often triggered even if nvidia-ml-py is used)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning, module="pynvml")
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="networkx")
    try:
        import pynvml
    except ImportError:
        pynvml = None

try:
    import amdsmi
except Exception:
    amdsmi = None
import torch
from torchdiffeq import odeint
from tqdm import tqdm

from data.data_preprocessing import TestDataPreprocessor
from src.create_cfm_model import resume_cfm_model
from src.modded_cfm import ModelWrapper

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark CFM model inference throughput."
    )
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config-path", type=str, required=True,
                        help="Path to config YAML file")
    parser.add_argument("--source-file", type=str, required=True,
                        help="Path to source data file")
    parser.add_argument("--n-objects", type=int, required=True,
                        help="Number of objects to process")
    parser.add_argument("--batch-size", type=int, required=True,
                        help="Batch size for inference")
    parser.add_argument("--device", type=str, required=True,
                        help="Compute device: 'cpu' or 'cuda' (supports NVIDIA/AMD)")
    parser.add_argument("--n-model-instances", type=int, required=True,
                        help="Number of model instances to run")
    parser.add_argument("--gpu-memory-limit", type=float,
                        help="GPU memory limit as a fraction [0, 1] (NVIDIA only)")
    parser.add_argument("--num-threads", type=int,
                        help="Number of PyTorch CPU threads")
    parser.add_argument("--monitor-interval", type=float, default=1.0,
                        help="Resource monitoring interval in seconds (default: 1.0)")
    parser.add_argument("--gpu-id", type=int, default=0,
                        help="GPU/ROCm device index to use (default: 0)")
    parser.add_argument("--output-json", type=str,
                        help="Path to save benchmark results as JSON")
    parser.add_argument("--list-devices", action="store_true",
                        help="List available compute devices and exit")
    parser.add_argument("--progress-bar", action="store_true",
                        help="Show a tqdm progress bar during sampling")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Resource monitoring
# ---------------------------------------------------------------------------

class ResourceMonitor:
    """Thread-based CPU / RAM / GPU resource monitor.

    Supports use as a context manager::

        with ResourceMonitor(interval=1.0, gpu_id=0) as monitor:
            ...
        stats = monitor.get_summary()
    """

    def __init__(self, interval: float = 1.0, gpu_id: int = 0) -> None:
        self.interval = interval
        self.gpu_id = gpu_id
        self.stats: Dict[str, List] = {
            "cpu": [], "ram": [], "gpu_util": [], "gpu_mem": []
        }
        self.running = False

        # Initialise GPU management library based on the backend.
        self.has_gpu = False
        self.gpu_backend = None
        if torch.cuda.is_available():
            backend_type = "ROCm/AMD" if getattr(torch.version, "hip", None) else "CUDA/NVIDIA"
            self.gpu_backend = "amd" if "AMD" in backend_type else "nvidia"

            if self.gpu_backend == "nvidia":
                try:
                    pynvml.nvmlInit()
                    self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
                    self.has_gpu = True
                except Exception as e:
                    log.warning("NVML initialisation failed — GPU stats may be limited: %s", e)
            elif self.gpu_backend == "amd":
                if amdsmi is not None:
                    try:
                        amdsmi.amdsmi_init()
                        handles = amdsmi.amdsmi_get_processor_handles()
                        if self.gpu_id < len(handles):
                            self.gpu_handle = handles[self.gpu_id]
                            self.has_gpu = True
                    except Exception as e:
                        log.warning("AMDSMI initialisation failed — GPU stats may be limited: %s", e)

    # ------------------------------------------------------------------
    # Context-manager interface
    # ------------------------------------------------------------------

    def __enter__(self) -> "ResourceMonitor":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        self.running = True
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.running = False
        self._thread.join()
        if self.gpu_backend == "nvidia":
            try:
                pynvml.nvmlShutdown()
            except Exception as e:
                log.warning("NVML shutdown failed: %s", e)
        elif self.gpu_backend == "amd" and amdsmi:
            try:
                amdsmi.amdsmi_shut_down()
            except Exception as e:
                log.warning("AMDSMI shutdown failed: %s", e)

    def get_summary(self) -> dict:
        summary = {
            "cpu_mean": float(np.mean(self.stats["cpu"])),
            "cpu_max":  float(np.max(self.stats["cpu"])),
            "ram_mean": float(np.mean(self.stats["ram"])),
            "ram_max":  float(np.max(self.stats["ram"])),
        }
        if self.has_gpu and self.stats["gpu_util"]:
            summary.update({
                "gpu_util_mean": float(np.mean(self.stats["gpu_util"])),
                "gpu_util_max":  float(np.max(self.stats["gpu_util"])),
                "gpu_mem_mean":  float(np.mean(self.stats["gpu_mem"])),
                "gpu_mem_max":   float(np.max(self.stats["gpu_mem"])),
            })
        return summary

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _monitor(self) -> None:
        while self.running:
            self.stats["cpu"].append(psutil.cpu_percent(interval=None))
            self.stats["ram"].append(psutil.virtual_memory().percent)

            if self.has_gpu:
                try:
                    if self.gpu_backend == "nvidia":
                        util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                        mem  = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                        self.stats["gpu_util"].append(util.gpu)
                        self.stats["gpu_mem"].append(mem.used / mem.total * 100)
                    elif self.gpu_backend == "amd" and amdsmi:
                        # amdsmi returns a dictionary or object depending on version
                        util = amdsmi.amdsmi_get_gpu_activity(self.gpu_handle)
                        vram = amdsmi.amdsmi_get_gpu_vram_usage(self.gpu_handle)
                        self.stats["gpu_util"].append(util.get("gfx_activity", 0))
                        self.stats["gpu_mem"].append(vram.get("vram_used", 0) / vram.get("vram_total", 1) * 100)
                except Exception as e:
                    log.debug("GPU management library collection error: %s", e)

            # Fallback for memory tracking using PyTorch (works on both CUDA and ROCm)
            # This is useful if management libraries (NVML/AMDSMI) are missing.
            if torch.cuda.is_available() and not self.stats["gpu_mem"]:
                try:
                    mem_used = torch.cuda.memory_reserved(self.gpu_id)
                    mem_total = torch.cuda.get_device_properties(self.gpu_id).total_memory
                    self.stats["gpu_mem"].append(mem_used / mem_total * 100)
                    # We can't easily get utilization from PyTorch alone.
                except Exception:
                    pass

            time.sleep(self.interval)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_resource_limits(args: argparse.Namespace) -> None:
    """Apply thread / memory limits requested via CLI args."""
    if args.num_threads:
        torch.set_num_threads(args.num_threads)
        log.info("PyTorch CPU threads set to %d", args.num_threads)

    if args.device == "cuda":
        if not torch.cuda.is_available():
            log.warning("CUDA requested but not available — falling back to CPU.")
            args.device = "cpu"
            return
        torch.cuda.set_device(args.gpu_id)
        if args.gpu_memory_limit:
            backend_type = "ROCm/AMD" if getattr(torch.version, "hip", None) else "CUDA/NVIDIA"
            if "ROCm" in backend_type:
                log.warning("GPU memory fraction limit is not formally supported on ROCm — setting anyway.")
            torch.cuda.set_per_process_memory_fraction(args.gpu_memory_limit)
            log.info("GPU (%s) memory fraction limit set to %.2f", backend_type, args.gpu_memory_limit)


def load_data(
    source_file: str,
    n_objects: int,
    data_kwargs: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load test data, tiling the dataset if fewer samples are available."""
    data_kwargs = dict(data_kwargs)  # avoid mutating caller's dict
    data_kwargs["N_test"] = n_objects
    data_kwargs["dataset_path"] = source_file

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        test_dataset = TestDataPreprocessor(data_kwargs, allow_nans=True)
        X_test, Y_test = test_dataset.get_dataset()

    if n_objects > len(X_test):
        n_repeats = n_objects // len(X_test) + 1
        X_test = np.tile(X_test, n_repeats)[:n_objects]
        Y_test = np.tile(Y_test, n_repeats)[:n_objects]
    else:
        X_test = X_test[:n_objects]
        Y_test = Y_test[:n_objects]

    return X_test, Y_test


def log_device_info(device: torch.device) -> None:
    """Log detailed information about the selected compute device."""
    if device.type == "cuda":
        backend = "ROCm/AMD" if getattr(torch.version, "hip", None) else "CUDA/NVIDIA"
        try:
            props = torch.cuda.get_device_properties(device)
            vram_gb = props.total_memory / 1024**3
            log.info("Compute backend : %s", backend)
            log.info("Device name     : %s", props.name)
            log.info("Total VRAM      : %.1f GB", vram_gb)
        except Exception as e:
            log.warning("Could not retrieve GPU properties: %s", e)
    else:
        log.info("Using CPU backend")


def format_benchmark_results(
    total_objects: int,
    total_time: float,
    resource_stats: dict,
    device: torch.device,
    num_threads: int,
    gpu_memory_limit: Optional[float] = None,
) -> dict:
    """Return a structured benchmark result dict suitable for JSON serialisation."""
    results: dict = {
        "performance": {
            "total_objects": total_objects,
            "total_time_seconds": total_time,
            "throughput_objects_per_second": total_objects / total_time,
        },
        "resource_usage": {
            "cpu": {
                "average_percent": round(resource_stats["cpu_mean"], 1),
                "peak_percent":    round(resource_stats["cpu_max"],  1),
            },
            "ram": {
                "average_percent": round(resource_stats["ram_mean"], 1),
                "peak_percent":    round(resource_stats["ram_max"],  1),
            },
        },
        "configuration": {
            "device": str(device),
            "num_threads": num_threads,
        },
    }

    if gpu_memory_limit is not None:
        results["configuration"]["gpu_memory_limit_fraction"] = gpu_memory_limit

    if "gpu_util_mean" in resource_stats:
        results["resource_usage"]["gpu"] = {
            "utilization_average_percent": round(resource_stats["gpu_util_mean"], 1),
            "utilization_peak_percent":    round(resource_stats["gpu_util_max"],  1),
            "memory_average_percent":      round(resource_stats["gpu_mem_mean"],  1),
            "memory_peak_percent":         round(resource_stats["gpu_mem_max"],   1),
        }

    return results


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(args: argparse.Namespace, config: dict) -> None:
    setup_resource_limits(args)

    # Build device once — setup_resource_limits may have fallen back to 'cpu'.
    device = torch.device(
        f"cuda:{args.gpu_id}" if args.device == "cuda" else args.device
    )

    log.info("-" * 60)
    log_device_info(device)
    log.info("Benchmark setup : %d instance(s)  |  PyTorch threads: %d",
             args.n_model_instances, torch.get_num_threads())
    if args.gpu_memory_limit:
        log.info("GPU memory limit: %.2f (fraction)", args.gpu_memory_limit)
    log.info("-" * 60)

    total_time = 0.0
    total_objects = 0

    with ResourceMonitor(interval=args.monitor_interval, gpu_id=args.gpu_id) as monitor:
        try:
            for instance in range(args.n_model_instances):
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                # Load model
                model, _, _ = resume_cfm_model(
                    os.path.dirname(args.model_path),
                    os.path.basename(args.model_path),
                )
                model = model.to(device)

                # Load data
                X_test, Y_test = load_data(
                    args.source_file, args.n_objects, config["data_kwargs"]
                )
                log.info("Instance %d/%d — %d objects loaded",
                         instance + 1, args.n_model_instances, len(X_test))

                sampler = ModelWrapper(model, context_dim=config["context_dim"])
                t_span = torch.linspace(
                    0, 1, config["base_kwargs"]["cfm"]["timesteps"]
                ).to(device)

                # Inference loop
                start_time = time.time()
                samples_list = []

                with torch.no_grad():
                    n_batches = (len(X_test) + args.batch_size - 1) // args.batch_size
                    batch_offsets = range(0, len(X_test), args.batch_size)

                    if args.progress_bar:
                        batch_offsets = tqdm(
                            batch_offsets,
                            total=n_batches,
                            desc=f"Instance {instance + 1}",
                            ascii=True,
                        )

                    for i in batch_offsets:
                        Y_batch = torch.tensor(
                            Y_test[i : i + args.batch_size]
                        ).float()
                        x0_sample = torch.randn(len(Y_batch), X_test.shape[1])
                        initial_conditions = torch.cat(
                            [x0_sample, Y_batch], dim=-1
                        ).to(device)

                        samples = odeint(
                            sampler,
                            initial_conditions,
                            t_span,
                            atol=1e-6,
                            rtol=1e-6,
                            method="euler",
                        )[-1, :, : X_test.shape[1]]

                        samples_list.append(samples.cpu().numpy())

                        if not args.progress_bar:
                            log.info(
                                "Batch %d/%d processed",
                                i // args.batch_size + 1, n_batches,
                            )

                instance_time = time.time() - start_time
                total_time += instance_time
                total_objects += len(X_test)

                del model, sampler, X_test, Y_test, samples_list
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        except Exception as e:
            log.error("Error during benchmark: %s", e)
            raise

        finally:
            # Belt-and-suspenders cleanup
            for var in ("model", "sampler"):
                if var in locals():
                    del locals()[var]
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # --- Summary ---
    throughput = total_objects / total_time
    log.info("Total throughput : %.2f objects/second", throughput)
    log.info("Total time       : %.2f s", total_time)
    log.info("Total objects    : %d", total_objects)

    resource_stats = monitor.get_summary()
    log.info("CPU   avg/peak   : %.1f%% / %.1f%%",
             resource_stats["cpu_mean"], resource_stats["cpu_max"])
    log.info("RAM   avg/peak   : %.1f%% / %.1f%%",
             resource_stats["ram_mean"], resource_stats["ram_max"])
    if "gpu_util_mean" in resource_stats:
        log.info("GPU util avg/peak: %.1f%% / %.1f%%",
                 resource_stats["gpu_util_mean"], resource_stats["gpu_util_max"])
        log.info("GPU mem  avg/peak: %.1f%% / %.1f%%",
                 resource_stats["gpu_mem_mean"], resource_stats["gpu_mem_max"])

    if args.output_json:
        results = format_benchmark_results(
            total_objects=total_objects,
            total_time=total_time,
            resource_stats=resource_stats,
            device=device,
            num_threads=torch.get_num_threads(),
            gpu_memory_limit=args.gpu_memory_limit,
        )
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        log.info("Benchmark results saved to: %s", args.output_json)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import yaml

    args = parse_args()

    if args.list_devices:
        log.info("Available compute devices:")
        log.info("  cpu : [0]")
        if torch.cuda.is_available():
            backend = "ROCm/AMD" if getattr(torch.version, "hip", None) else "CUDA/NVIDIA"
            indices = list(range(torch.cuda.device_count()))
            log.info("  cuda (%s): %s", backend, indices)
        sys.exit(0)

    # Validate device selection
    if args.device == "cuda":
        if not torch.cuda.is_available():
            log.error("'cuda' device requested but CUDA/ROCm is not available.")
            sys.exit(1)
        n_gpus = torch.cuda.device_count()
        if args.gpu_id >= n_gpus:
            log.error(
                "GPU index %d requested but only %d device(s) available (indices 0-%d).",
                args.gpu_id, n_gpus, n_gpus - 1,
            )
            sys.exit(1)
    elif args.device != "cpu":
        log.error("Unknown device '%s'. Use 'cpu' or 'cuda'.", args.device)
        sys.exit(1)

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    run_benchmark(args, config)


if __name__ == "__main__":
    main()
