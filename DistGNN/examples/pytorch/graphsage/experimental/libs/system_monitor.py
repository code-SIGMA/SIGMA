"""
system_monitor.py

Background system/GPU metrics sampler for accurate real-time monitoring
during training runs.

Fixes the classic psutil/pynvml pitfalls:

- psutil.cpu_percent() needs a warm-up call, otherwise the first sample
  is garbage (it has nothing to compare against yet). The same applies
  separately to cpu_percent(percpu=True) and cpu_times_percent() - psutil
  keeps an independent "last call" cache per variant, so each one needs
  its own warm-up.
- Sampling must run in a dedicated background thread with drift-corrected
  timing, not blocking your training loop and not drifting over time due
  to `time.sleep(interval)` ignoring the time the measurement itself took.
- NVML GPU utilization is an instantaneous snapshot (percent of time busy
  over the driver's internal sample window, roughly 1/6s-1s), not an
  average over your polling interval. Polling too infrequently makes it
  look noisy/misleading. This monitor samples frequently (default 1s,
  can go lower) and keeps every sample so you can aggregate afterwards.
- Disk and network counters are cumulative totals, not rates. They are
  converted to MB/s by diffing against the previous sample and dividing
  by the *actually elapsed* time (measured with a monotonic clock), not
  by the nominal interval - those differ whenever the loop falls behind.

Collected fields per sample:

    timestamp                        unix epoch seconds (float)
    cpu_percent_system               total CPU load, 0-100
    cpu_iowait_percent               share of time waiting on I/O (Linux)
    cpu_per_core                     list[float], one entry per core
    cpu_percent_process_raw          may exceed 100% across threads/cores
    cpu_percent_process_normalized   divided by core count, 0-100
    ram_percent / ram_used_gb / ram_total_gb / ram_available_gb
    swap_percent
    disk_read_mb_s / disk_write_mb_s      None in the very first sample
    net_recv_mb_s / net_sent_mb_s         None in the very first sample
    proc_rss_gb                      resident memory of the tracked process
    proc_threads                     thread count of the tracked process
    gpu{i}_util_percent / gpu{i}_mem_util_percent
    gpu{i}_mem_used_gb / gpu{i}_mem_total_gb
    gpu{i}_power_w / gpu{i}_temp_c

Rate fields (disk_*, net_*) are None in the first sample because a rate
needs two points in time. Downstream consumers should treat None as
"unknown" rather than zero.

Usage:

    from system_monitor import SystemMonitor

    with SystemMonitor(interval=1.0, csv_path="metrics.csv") as mon:
        for epoch in range(epochs):
            train_one_epoch(...)
        print(mon.summary())

    # or manually:
    mon = SystemMonitor(interval=0.5, gpu_indices=[0, 1])
    mon.start()
    ...
    mon.stop()
    print(mon.summary())
"""
from __future__ import annotations

import csv
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional, List

import re

import psutil

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Display metadata for the textual rendering of summary():
#   key -> (group, label, unit, decimal places)
# Unknown keys fall back to the "Other" group, so nothing ever breaks.
# --------------------------------------------------------------------------
_FIELD_META: dict[str, tuple[str, str, str, int]] = {
    "cpu_percent_system":             ("CPU",     "Total load",        "%",    1),
    "cpu_iowait_percent":             ("CPU",     "I/O wait",          "%",    1),
    "cpu_percent_process_raw":        ("CPU",     "Process (raw)",     "%",    1),
    "cpu_percent_process_normalized": ("CPU",     "Process (per core)","%",    1),
    "ram_percent":                    ("Memory",  "RAM used",          "%",    1),
    "ram_used_gb":                    ("Memory",  "RAM used",          "GB",   2),
    "ram_available_gb":               ("Memory",  "RAM available",     "GB",   2),
    "ram_total_gb":                   ("Memory",  "RAM total",         "GB",   2),
    "swap_percent":                   ("Memory",  "Swap used",         "%",    1),
    "disk_read_mb_s":                 ("I/O",     "Disk read",         "MB/s", 2),
    "disk_write_mb_s":                ("I/O",     "Disk write",        "MB/s", 2),
    "net_recv_mb_s":                  ("I/O",     "Net received",      "MB/s", 2),
    "net_sent_mb_s":                  ("I/O",     "Net sent",          "MB/s", 2),
    "proc_rss_gb":                    ("Process", "RSS",               "GB",   3),
    "proc_threads":                   ("Process", "Threads",           "",     0),
}

# Suffix after "gpu<N>_" -> (label, unit, decimal places)
_GPU_META: dict[str, tuple[str, str, int]] = {
    "util_percent":     ("Utilization",   "%",  1),
    "mem_util_percent": ("Memory bus",    "%",  1),
    "mem_used_gb":      ("VRAM used",     "GB", 2),
    "mem_total_gb":     ("VRAM total",    "GB", 2),
    "power_w":          ("Power draw",    "W",  1),
    "temp_c":           ("Temperature",   "°C", 0),
}

_GROUP_ORDER = ("CPU", "Memory", "I/O", "Process")
_GPU_KEY_RE = re.compile(r"^gpu(\d+)_(.+)$")
_CORE_KEY_RE = re.compile(r"^cpu_per_core\[(\d+)\]$")

try:
    import pynvml

    _PYNVML_IMPORTABLE = True
except ImportError:
    _PYNVML_IMPORTABLE = False


class SystemMonitor:
    """
    Samples CPU, RAM, swap, disk I/O, network I/O, per-process and
    (optionally) GPU metrics on a fixed interval in a background thread and
    optionally writes each sample as a CSV row.

    interval:     seconds between samples (default 1.0). Lower it (e.g. 0.2)
                  if you need finer resolution, e.g. to catch short GPU spikes.
    csv_path:     if given, each sample is appended as a CSV row (streamed,
                  not buffered) so you get live data even if the process
                  crashes. List-valued fields (cpu_per_core) are stored as
                  JSON so the column stays a single, parseable cell.
    gpu_indices:  which GPU indices to monitor (e.g. [0, 1]). None = all
                  visible GPUs. Ignored if pynvml/NVML is unavailable.
    process:      psutil.Process to track for per-process CPU/RAM. Defaults
                  to the current process.
    per_core:     collect per-core CPU load (default True). Set False to keep
                  samples small on machines with very many cores.
    """

    # fields that are lists and therefore need JSON encoding in CSV
    _LIST_FIELDS = ("cpu_per_core",)

    def __init__(
        self,
        interval: float = 1.0,
        csv_path: Optional[str | Path] = None,
        gpu_indices: Optional[list[int]] = None,
        process: Optional[psutil.Process] = None,
        per_core: bool = True,
    ):
        self.interval = interval
        self.csv_path = Path(csv_path) if csv_path else None
        self.process = process or psutil.Process(os.getpid())

        try:
            self.cores = sorted(os.sched_getaffinity(0))   # cpuset des Containers
        except AttributeError:
            self.cores = list(range(psutil.cpu_count() or 1))
        self.cpu_count = len(self.cores)
        self.per_core = per_core

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._samples: list[dict[str, Any]] = []
        self._lock = threading.Lock()

        self._csv_file = None
        self._csv_writer: Optional[csv.DictWriter] = None

        # baselines for rate calculation (disk / network)
        self._last_disk: Optional[Any] = None
        self._last_net: Optional[Any] = None
        self._last_rate_time: Optional[float] = None

        self._gpu_available = False
        self._gpu_handles: dict[int, Any] = {}
        self._init_gpu(gpu_indices)

    # ---------- GPU setup ----------

    def _init_gpu(self, gpu_indices: Optional[List[int]]) -> None:
        if not _PYNVML_IMPORTABLE:
            logger.info("pynvml not installed, GPU metrics disabled.")
            return
        if gpu_indices is not None and len(gpu_indices) == 0:
            return

        nvml_started = False
        try:
            pynvml.nvmlInit()
            nvml_started = True
            device_count = pynvml.nvmlDeviceGetCount()
            indices = gpu_indices if gpu_indices is not None else list(range(device_count))
            for idx in indices:
                self._gpu_handles[idx] = pynvml.nvmlDeviceGetHandleByIndex(idx)
            self._gpu_available = len(self._gpu_handles) > 0
        except Exception as exc:
            logger.info("NVML unavailable, GPU metrics disabled (%s)", exc)
            self._gpu_handles.clear()
            self._gpu_available = False
            if nvml_started:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass

    # ---------- Context manager ----------

    def __enter__(self) -> "SystemMonitor":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    # ---------- Public API ----------

    def start(self) -> None:
        if self._thread is not None:
            return  # already running

        # Warm-up calls. psutil keeps a separate "previous value" cache per
        # variant, so system-wide, per-core, cpu_times and per-process each
        # need their own priming call - otherwise their first real sample
        # is measured against process start instead of against the interval.
        psutil.cpu_percent(interval=None, percpu=True)
        psutil.cpu_times_percent(interval=None)
        try:
            self.process.cpu_percent(interval=None)
        except psutil.Error:
            logger.warning("Could not prime per-process CPU measurement.")

        # Baselines for disk/network rates, so the second sample already
        # has a valid rate instead of wasting the first two samples.
        self._last_disk = self._safe_disk_counters()
        self._last_net = self._safe_net_counters()
        self._last_rate_time = time.monotonic()

        if self.csv_path:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join()
        self._thread = None

        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

        if self._gpu_available:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def samples(self) -> list[dict[str, Any]]:
        """Returns a copy of all collected samples."""
        with self._lock:
            return list(self._samples)

    def summary(self) -> dict[str, dict[str, float]]:
        """
        Returns min/mean/max for every numeric field across all samples.

        List fields (cpu_per_core) are aggregated element-wise into
        cpu_per_core[i] entries. Fields that are None everywhere (e.g. rates
        on a system without disk counters) are skipped.
        """
        rows = self.samples()
        if not rows:
            return {}

        # Union of keys across all rows: GPU keys can appear late if a query
        # failed on the very first sample.
        keys: list[str] = []
        for row in rows:
            for k in row:
                if k != "timestamp" and k not in keys:
                    keys.append(k)

        result: dict[str, dict[str, float]] = {}
        for key in keys:
            if key in self._LIST_FIELDS:
                result.update(self._summarize_list_field(rows, key))
                continue
            values = [
                r[key] for r in rows
                if isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)
            ]
            if not values:
                continue
            result[key] = {
                "min": min(values),
                "mean": sum(values) / len(values),
                "max": max(values),
            }
        return result

    @staticmethod
    def _summarize_list_field(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
        lists = [r[key] for r in rows if isinstance(r.get(key), list)]
        if not lists:
            return {}
        width = max(len(v) for v in lists)
        out: dict[str, dict[str, float]] = {}
        for i in range(width):
            values = [
                v[i] for v in lists
                if len(v) > i and isinstance(v[i], (int, float))
            ]
            if not values:
                continue
            out[f"{key}[{i}]"] = {
                "min": min(values),
                "mean": sum(values) / len(values),
                "max": max(values),
            }
        return out

    # ---------- Text rendering ----------

    def summary_text(self, title: str = "System metrics",
                     per_core: str = "compact") -> str:
        """
        summary() rendered as a readable table.

        per_core: "compact" (one aggregate line), "full" (one bar per core)
                  or "off" (skip per-core data entirely).
        Sample count and wall-clock span are appended to the title.
        """
        rows = self.samples()
        if rows:
            span = rows[-1]["timestamp"] - rows[0]["timestamp"]
            title = (f"{title} | {len(rows)} samples | "
                     f"{self._fmt_duration(span)} | interval {self.interval:g}s")
        return self.format_summary(self.summary(), title=title, per_core=per_core)

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        seconds = int(round(seconds))
        hours, rest = divmod(seconds, 3600)
        minutes, secs = divmod(rest, 60)
        if hours:
            return f"{hours}h {minutes:02d}m {secs:02d}s"
        if minutes:
            return f"{minutes}m {secs:02d}s"
        return f"{secs}s"

    @staticmethod
    def _bar(percent: float, width: int = 20) -> str:
        """Simple ASCII gauge, only used for percentage fields."""
        percent = max(0.0, min(100.0, percent))
        filled = int(round(percent / 100.0 * width))
        return "#" * filled + "." * (width - filled)

    @classmethod
    def format_summary(cls, summary: dict[str, dict[str, float]],
                       title: str = "System metrics",
                       per_core: str = "compact") -> str:
        """
        Formats a summary() dict as a table. Classmethod on purpose: it takes
        a plain dict, so summaries reconstructed from CSV or from another
        rank's database rows can be rendered with it too.
        """
        if not summary:
            return f"\n{title}: no samples recorded.\n"

        cores: dict[int, dict[str, float]] = {}
        groups: dict[str, list[tuple[str, str, int, dict[str, float]]]] = {}

        for key, stats in summary.items():
            core_match = _CORE_KEY_RE.match(key)
            if core_match:
                cores[int(core_match.group(1))] = stats
                continue

            meta = _FIELD_META.get(key)
            if meta is None:
                gpu_match = _GPU_KEY_RE.match(key)
                if gpu_match:
                    # GPU keys are generated per device index, so they cannot
                    # live in a static table.
                    idx, suffix = gpu_match.group(1), gpu_match.group(2)
                    label, unit, prec = _GPU_META.get(suffix, (suffix, "", 2))
                    meta = (f"GPU {idx}", label, unit, prec)
                else:
                    meta = ("Other", key, "", 2)

            group, label, unit, prec = meta
            groups.setdefault(group, []).append((label, unit, prec, stats))

        def group_key(name: str) -> tuple[int, int]:
            """Fixed groups first, then GPUs by index, then the fallback."""
            if name in _GROUP_ORDER:
                return (0, _GROUP_ORDER.index(name))
            if name.startswith("GPU "):
                return (1, int(name.split()[1]))
            return (2, 0)

        ordered = sorted(groups.items(), key=lambda kv: group_key(kv[0]))

        def num(value: float, prec: int) -> str:
            return f"{value:,.{prec}f}".replace(",", " ")

        # Column widths are computed globally so every group lines up.
        lab_w = max(len(lab) for rows in groups.values() for lab, *_ in rows)
        num_w = max(
            len(num(stats[c], prec))
            for rows in groups.values()
            for _, _, prec, stats in rows
            for c in ("min", "mean", "max")
        )
        num_w = max(num_w, 4)
        unit_w = max(len(u) for rows in groups.values() for _, u, _, _ in rows)

        body: list[str] = [
            f"  {'Metric':<{lab_w + 2}}  {'min':>{num_w}}  "
            f"{'mean':>{num_w}}  {'max':>{num_w}}"
        ]

        for group, rows in ordered:
            body.append("")
            body.append(f"  {group}")
            for label, unit, prec, stats in rows:
                line = (f"    {label:<{lab_w}}  "
                        f"{num(stats['min'], prec):>{num_w}}  "
                        f"{num(stats['mean'], prec):>{num_w}}  "
                        f"{num(stats['max'], prec):>{num_w}}  "
                        f"{unit:<{unit_w}}")
                if unit == "%":
                    line += "  " + cls._bar(stats["mean"])
                body.append(line.rstrip())

        if cores and per_core != "off":
            means = {i: s["mean"] for i, s in cores.items()}
            lo_i = min(means, key=means.get)
            hi_i = max(means, key=means.get)
            avg = sum(means.values()) / len(means)
            body.append("")
            body.append(f"  CPU per core ({len(cores)} cores)")
            body.append(f"    avg {avg:.1f} %   |   quietest #{lo_i} {means[lo_i]:.1f} %"
                        f"   |   busiest #{hi_i} {means[hi_i]:.1f} %"
                        f"   |   spread {means[hi_i] - means[lo_i]:.1f} pp")
            if per_core == "full":
                for i in sorted(cores):
                    s = cores[i]
                    body.append(f"    #{i:<3} {cls._bar(s['mean'])} {s['mean']:>5.1f} %"
                                f"   (min {s['min']:.1f} / max {s['max']:.1f})")

        width = max(max(len(line) for line in body), len(title) + 2) + 2
        out = ["", "=" * width, f"  {title}", "=" * width, body[0],
               "  " + "-" * (width - 4)]
        out.extend(body[1:])
        out.append("=" * width)
        out.append("")
        return "\n".join(out)


    # ---------- Internals ----------

    def _run(self) -> None:
        next_time = time.monotonic()
        while not self._stop_event.is_set():
            try:
                row = self._sample_once()
                self._handle_sample(row)
            except Exception:
                logger.exception("Error while collecting a system metrics sample")

            next_time += self.interval
            sleep_time = next_time - time.monotonic()
            if sleep_time > 0:
                self._stop_event.wait(sleep_time)
            else:
                # We fell behind (sampling took longer than interval) -
                # reset the schedule instead of trying to "catch up" in a
                # tight loop.
                next_time = time.monotonic()

    @staticmethod
    def _safe_disk_counters() -> Optional[Any]:
        try:
            return psutil.disk_io_counters()
        except Exception:
            return None

    @staticmethod
    def _safe_net_counters():
        try:
            nics = psutil.net_io_counters(pernic=True)
            real = [c for name, c in nics.items() if name != "lo"]
            if not real:
                return None
            class _Sum: pass
            s = _Sum()
            s.bytes_recv = sum(c.bytes_recv for c in real)
            s.bytes_sent = sum(c.bytes_sent for c in real)
            return s
        except Exception:
            return None

    def _sample_once(self) -> dict[str, Any]:
        row: dict[str, Any] = {"timestamp": time.time()}

        # ---- CPU (only cores in the cpuset) ----
        try:
            all_cores = psutil.cpu_percent(interval=None, percpu=True)
            mine = [all_cores[i] for i in self.cores if i < len(all_cores)]
            row["cpu_percent_system"] = sum(mine) / len(mine) if mine else None
            if self.per_core:
                row["cpu_per_core"] = mine
        except Exception:
            row["cpu_percent_system"] = None
            if self.per_core:
                row["cpu_per_core"] = None

        row["cpu_percent_system"] = psutil.cpu_percent(interval=None)

        # process cpu_percent can exceed 100% with multiple threads/cores;
        # normalized version divides by core count for a 0-100% scale.
        try:
            proc_cpu_raw = self.process.cpu_percent(interval=None)
            row["cpu_percent_process_raw"] = proc_cpu_raw
            row["cpu_percent_process_normalized"] = proc_cpu_raw / self.cpu_count
        except psutil.Error:
            row["cpu_percent_process_raw"] = None
            row["cpu_percent_process_normalized"] = None

        # ---- RAM / swap ----
        vm = psutil.virtual_memory()
        row["ram_percent"] = vm.percent
        row["ram_used_gb"] = round(vm.used / 1e9, 3)
        row["ram_total_gb"] = round(vm.total / 1e9, 3)
        # `available` is not total-used: it accounts for reclaimable cache
        # and buffers, so it reflects what a new allocation could actually get.
        row["ram_available_gb"] = round(vm.available / 1e9, 3)

        try:
            row["swap_percent"] = psutil.swap_memory().percent
        except Exception:
            row["swap_percent"] = None

        # ---- Disk / network rates ----
        self._add_io_rates(row)

        # ---- Per-process ----
        try:
            row["proc_rss_gb"] = round(self.process.memory_info().rss / 1e9, 4)
        except psutil.Error:
            row["proc_rss_gb"] = None
        try:
            row["proc_threads"] = self.process.num_threads()
        except psutil.Error:
            row["proc_threads"] = None

        # ---- GPU ----
        if self._gpu_available:
            for idx, handle in self._gpu_handles.items():
                self._add_gpu_metrics(row, idx, handle)

        return row

    def _add_io_rates(self, row: dict[str, Any]) -> None:
        """
        Convert cumulative disk/net counters into MB/s.

        Divides by the actually elapsed monotonic time rather than by
        self.interval: if the loop falls behind, those diverge and the rate
        would be overstated.
        """
        now = time.monotonic()
        elapsed = (now - self._last_rate_time) if self._last_rate_time is not None else 0.0

        disk = self._safe_disk_counters()
        net = self._safe_net_counters()

        def rate(current: Optional[int], previous: Optional[int]) -> Optional[float]:
            if current is None or previous is None or elapsed <= 0:
                return None
            delta = current - previous
            if delta < 0:  # counter wrapped or device reset
                return None
            return round(delta / 1e6 / elapsed, 4)

        row["disk_read_mb_s"] = rate(
            getattr(disk, "read_bytes", None), getattr(self._last_disk, "read_bytes", None)
        )
        row["disk_write_mb_s"] = rate(
            getattr(disk, "write_bytes", None), getattr(self._last_disk, "write_bytes", None)
        )
        row["net_recv_mb_s"] = rate(
            getattr(net, "bytes_recv", None), getattr(self._last_net, "bytes_recv", None)
        )
        row["net_sent_mb_s"] = rate(
            getattr(net, "bytes_sent", None), getattr(self._last_net, "bytes_sent", None)
        )

        if disk is not None:
            self._last_disk = disk
        if net is not None:
            self._last_net = net
        self._last_rate_time = now

    def _add_gpu_metrics(self, row: dict[str, Any], idx: int, handle: Any) -> None:
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            row[f"gpu{idx}_util_percent"] = util.gpu
            row[f"gpu{idx}_mem_util_percent"] = util.memory
        except Exception:
            pass

        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            row[f"gpu{idx}_mem_used_gb"] = round(mem.used / 1e9, 3)
            row[f"gpu{idx}_mem_total_gb"] = round(mem.total / 1e9, 3)
        except Exception:
            pass

        try:
            row[f"gpu{idx}_power_w"] = round(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000, 2)
        except Exception:
            pass

        try:
            row[f"gpu{idx}_temp_c"] = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )
        except Exception:
            pass

    # ---------- Output ----------

    @classmethod
    def _csv_value(cls, key: str, value: Any) -> Any:
        """Lists become JSON so a CSV cell stays a single parseable value."""
        if isinstance(value, list):
            return json.dumps(value)
        return value

    def _handle_sample(self, row: dict[str, Any]) -> None:
        with self._lock:
            self._samples.append(row)

        if self._csv_file is not None:
            if self._csv_writer is None:
                self._csv_writer = csv.DictWriter(
                    self._csv_file,
                    fieldnames=list(row.keys()),
                    extrasaction="ignore",  # a late-appearing GPU key must not crash the run
                )
                if self._csv_file.tell() == 0:
                    self._csv_writer.writeheader()
            self._csv_writer.writerow({k: self._csv_value(k, v) for k, v in row.items()})
            self._csv_file.flush()  # so data survives a crash, not just a clean exit
