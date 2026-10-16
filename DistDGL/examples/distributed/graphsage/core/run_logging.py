"""Everything logged during a run: system monitor,
epoch CSV, database and console output."""
import os
import re
from datetime import datetime, timezone

from csv_writer import CsvWriter
from system_monitor import SystemMonitor


EPOCH_FIELDS = [
    "epoch", "train_loss", "total_time", "forward_time",
    "backward_time", "wait_time", "update_time",
    "train_accuracy", "val_accuracy", "test_accuracy",
]
DB_EXCLUDED_FIELDS = {"wait_time"}      # The DB table does not have this column


def nvml_indices(device):
    """CUDA-Device -> physischer NVML-Index (NVML ignoriert CUDA_VISIBLE_DEVICES)."""
    if device.type != "cuda":
        return []
    idx = device.index
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis:
        idx = int(vis.split(",")[idx])
    return [idx]


def send_system_metrics_to_db(database, samples, run_id, rank, batch_size=2000):
    gpu_re = re.compile(r"^gpu(\d+)_(util_percent|mem_util_percent|mem_used_gb)$")
    columns = [
        "run_id", "rank", "recorded_at",
        "cpu_total_percent", "cpu_iowait", "cpu_per_core",
        "ram_percent", "ram_available_gb", "swap_percent",
        "disk_read_mb_s", "disk_write_mb_s",
        "net_recv_mb_s", "net_sent_mb_s",
        "proc_rss_gb", "proc_threads", "proc_cpu_percent",
        "gpu_util_percent", "gpu_mem_used_gb", "gpu_mem_percent",
    ]

    def build(s):
        def mem_pct(i):
            u = s.get(f"gpu{i}_mem_used_gb")
            t = s.get(f"gpu{i}_mem_total_gb")
            return round(u / t * 100, 2) if u is not None and t else None

        gpus = sorted({int(m.group(1)) for k in s if (m := gpu_re.match(k))})

        def arr(metric):
            return [s.get(f"gpu{i}_{metric}") for i in gpus] if gpus else None

        return {
            "run_id": run_id, "rank": rank,
            "recorded_at": datetime.fromtimestamp(s["timestamp"], timezone.utc).replace(tzinfo=None),
            "cpu_total_percent": s.get("cpu_percent_system"),
            "cpu_iowait": s.get("cpu_iowait_percent"),
            "cpu_per_core": s.get("cpu_per_core"),
            "ram_percent": s.get("ram_percent"),
            "ram_available_gb": s.get("ram_available_gb"),
            "swap_percent": s.get("swap_percent"),
            "disk_read_mb_s": s.get("disk_read_mb_s"),
            "disk_write_mb_s": s.get("disk_write_mb_s"),
            "net_recv_mb_s": s.get("net_recv_mb_s"),
            "net_sent_mb_s": s.get("net_sent_mb_s"),
            "proc_rss_gb": s.get("proc_rss_gb"),
            "proc_threads": s.get("proc_threads"),
            "proc_cpu_percent": s.get("cpu_percent_process_normalized"),
            "gpu_util_percent": arr("util_percent"),
            "gpu_mem_used_gb": arr("mem_used_gb"),
            "gpu_mem_percent": [mem_pct(i) for i in gpus] if gpus else None,
        }

    rows = [build(s) for s in samples]
    sent = 0
    for i in range(0, len(rows), batch_size):
        sent += database.copy_rows("system_metric", rows[i:i + batch_size], columns=columns)
    return sent


class RunLogger:
    """Bundle the monitor, CSV and DB for one rank.

    Initialization is defensive: a directory that cannot be written to
    disables only logging, not training.
    """

    def __init__(self, args, rank, device, data_dirpath, run_id, database):
        self.rank = rank
        self.run_id = run_id
        self.database = database

        write_here = args.save_not_only_rank0 or rank == 0
        self.send_data = args.send_data_to_database and write_here

        self.mon = None
        if args.save_system_metrics_to_csv and write_here:
            try:
                self.mon = SystemMonitor(
                    interval=0.5,
                    csv_path=(data_dirpath / f"system_metrics_rank{rank}-{run_id}.csv").as_posix(),
                    gpu_indices=nvml_indices(device),
                )
                self.mon.start()
            except OSError as exc:
                self._say(f"System-Monitoring deaktiviert: {exc}")
                self.mon = None

        self.writer = None
        if args.save_data_to_csv and write_here:
            try:
                self.writer = CsvWriter(
                    path=data_dirpath / f"epoch_metrics_rank{rank}-{run_id}.csv",
                    fieldnames=EPOCH_FIELDS,
                )
            except OSError as exc:
                self._say(f"CSV-Logging deaktiviert: {exc}")
                self.writer = None

        self._closed = False

    # ------------------------------------------------------------------
    def _say(self, msg):
        print(f"[rank {self.rank}] {msg}", flush=True)

    def _send_epoch_row(self, data):
        try:
            self.database.send_data(table_name="epoch_metric", data=data)
        except Exception as exc:
            self._say(f"Epoch metric not sent: {exc}")

    # ------------------------------------------------------------------
    def log_epoch(self, row, *, do_eval, max_time, sample_time, data_time):
        """row: dict with keys from EPOCH_FIELDS."""
        if self.writer is not None:
            self.writer.append_row(row)

        if self.send_data:
            db_row = {k: v for k, v in row.items() if k not in DB_EXCLUDED_FIELDS}
            db_row.update(run_id=self.run_id, rank=self.rank)
            self._send_epoch_row(db_row)

        if self.rank == 0:
            if do_eval:
                print()
            print(f"[{datetime.now():%H:%M:%S}] Epoch {row['epoch']:05d} | "
                  f"Max Time(s) {max_time:.4f} | Total Time(s) {row['total_time']:.4f} | "
                  f"Sample {sample_time:.4f} | Data {data_time:.4f} | "
                  f"Fwd {row['forward_time']:.4f} | Bwd {row['backward_time']:.4f} | "
                  f"Upd {row['update_time']:.4f} | Loss {row['train_loss']:.4f}"
                  + (f" | Train {row['train_accuracy']:.2f} | Val {row['val_accuracy']:.2f}"
                     if do_eval else ""),
                  flush=True)
            if do_eval:
                print()

    def log_best(self, res, best_epoch):
        """Result of the best model; call only on rank 0."""
        if self.send_data:
            self._send_epoch_row({
                "run_id": self.run_id, "rank": self.rank, "epoch": -1,
                "train_loss": -1, "total_time": -1, "forward_time": -1,
                "backward_time": -1, "update_time": -1,
                "train_accuracy": res["train"],
                "val_accuracy": res["val"],
                "test_accuracy": res["test"],
            })
        print(f"Best Model: Train {res['train']:.4f} | Val {res['val']:.4f} | "
              f"Test {res['test']:.4f} | Epoch {best_epoch:05d}", flush=True)

    def close(self):
        """Stop the monitor and CSV, send system metrics. Safe to call repeatedly."""
        if self._closed:
            return
        self._closed = True

        if self.writer is not None and hasattr(self.writer, "close"):
            self.writer.close()

        if self.mon is not None:
            self.mon.stop()
            print(self.mon.summary_text(title=f"System metrics | rank {self.rank}"), flush=True)
            if self.send_data:
                try:
                    sent = send_system_metrics_to_db(
                        self.database, self.mon.samples(), self.run_id, self.rank
                    )
                    self._say(f"{sent} system metrics sent to the database.")
                except Exception as exc:
                    self._say(f"Could not send system metrics: {exc}")
