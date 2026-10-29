#!/usr/bin/env python3
"""仅服务器CPU运行：IO提前失败、原异常保留、日志定位与CSV配对回归。"""
import errno
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import harness_io as storage
import profile_plan_transfer as profile
from checks.check_full_cache_inputs import CsvChecks, InputChecks


def fs_stat(free=10 << 30, inodes=10000):
    return SimpleNamespace(f_bavail=free, f_frsize=1, f_favail=inodes, f_files=100000)


class StorageChecks(unittest.TestCase):
    def test_disk_and_inode_shortage(self):
        with tempfile.TemporaryDirectory() as folder:
            for stat in (fs_stat(free=0), fs_stat(inodes=0)):
                with patch.object(storage.os, "statvfs", return_value=stat):
                    with self.assertRaisesRegex(RuntimeError, "free_bytes=.*free_inodes="):
                        storage.inspect_directory(folder, 1024, 10)

    def test_unwritable_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(storage.os, "open", side_effect=PermissionError("read-only")):
                with self.assertRaisesRegex(RuntimeError, "read-only"):
                    storage.inspect_directory(folder, 0, 0)

    def test_shared_filesystem_reserves_combined_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(storage.os, "statvfs", return_value=fs_stat(free=1500)):
                with self.assertRaisesRegex(RuntimeError, "合计最低余量"):
                    storage.preflight(Path(folder) / "out", Path(folder) / "tmp", 1000, 1000, 0)

    def test_temp_is_explicit_and_source_space_is_not_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            out, tmp = Path(folder) / "out", Path(folder) / "tmp"
            old_temp = tempfile.tempdir
            try:
                with patch.dict(os.environ), patch.object(storage.os, "statvfs", return_value=fs_stat()) as stat:
                    info = storage.preflight(out, tmp, 1, 1, 0)
                    self.assertEqual(os.environ["TMPDIR"], str(tmp.resolve()))
                    self.assertEqual(tempfile.gettempdir(), str(tmp.resolve()))
                    self.assertEqual(info["output"]["path"], str(out.resolve()))
                    self.assertTrue({call.args[0] for call in stat.call_args_list}.issubset(
                        {Path(folder).resolve(), out.resolve(), tmp.resolve()}))
                    self.assertNotIn(ROOT, {call.args[0] for call in stat.call_args_list})
            finally:
                tempfile.tempdir = old_temp

    def test_preflight_failure_precedes_npu_and_host_allocation(self):
        args = ["profile_plan_transfer.py", "--rows", "1", "--topk", "1", "--capacity", "3",
                "--max-token", "16"]
        with patch.object(sys, "argv", args), patch.object(profile, "preflight_args",
                side_effect=RuntimeError("disk full")), patch.object(profile.common, "load_runtime") as runtime, \
                patch.object(profile.common, "prepare_sources") as host:
            with self.assertRaisesRegex(RuntimeError, "disk full"):
                profile.main()
            runtime.assert_not_called()
            host.assert_not_called()


class FailureChecks(unittest.TestCase):
    def test_secondary_enospc_preserves_original_exception(self):
        original = ValueError("original parser failure")
        def operation():
            try:
                raise original
            except BaseException as error:
                storage.record_failure(Path("unused"), {}, error)
                raise
        with patch.object(storage, "write_json", side_effect=OSError(errno.ENOSPC, "disk full")), \
                patch.object(sys, "stderr", io.StringIO()):
            with self.assertRaises(ValueError) as caught:
                operation()
        self.assertIs(caught.exception, original)

    def test_first_parser_exception_and_paths_are_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            log = directory / "profile.log"
            log.write_text("Traceback (most recent call last):\nOSError: [Errno 28] No space left\n")
            detail = storage.diagnostic_details(directory, log)
            self.assertIn("OSError", detail["first_locatable_exception"])
            message = str(storage.capture_error(directory, log, RuntimeError("missing CSV")))
            for text in (str(directory), str(log), "OSError", "missing CSV"):
                self.assertIn(text, message)

    def test_json_is_atomic_and_failed_write_is_not_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "capture_metadata.json"
            storage.write_json(path, {"status": "running"})
            original = path.read_bytes()
            with patch.object(storage.os, "fsync", side_effect=OSError(errno.ENOSPC, "disk full")):
                with self.assertRaises(OSError):
                    storage.write_json(path, {"status": "complete"})
            self.assertEqual(path.read_bytes(), original)
            self.assertFalse((Path(folder) / "COMPLETE.json").exists())

    def test_finalization_enospc_does_not_publish_success_report(self):
        for failed_file in ("capture_metadata.json", "COMPLETE.json"):
            with tempfile.TemporaryDirectory() as folder:
                session = Path(folder)
                report = {"runs": [], "launch_mode": "prepared"}
                def save(path, data):
                    if Path(path).name == failed_file and data.get("status") == "complete":
                        raise OSError(errno.ENOSPC, "disk full at finalization")
                    storage.write_json(path, data)
                with patch.object(profile, "write_json", side_effect=save):
                    with self.assertRaises(OSError):
                        profile.finalize_report(session, report)
                self.assertFalse((session / "report.md").exists())
                self.assertFalse((session / "COMPLETE.json").exists())


class GapChecks(CsvChecks):
    def test_gap_is_median_of_actual_pairs(self):
        rows = [(profile.common.PLAN_KERNEL, "AIV", 0, 5),
                (profile.common.TRANSFER_KERNEL, "AIV", 1005, 5),
                (profile.common.PLAN_KERNEL, "AIV", 1200, 50),
                (profile.common.TRANSFER_KERNEL, "AIV", 1350, 50),
                (profile.common.PLAN_KERNEL, "AIV", 2000, 500),
                (profile.common.TRANSFER_KERNEL, "AIV", 2500, 500)]
        stats = profile.summarize(self.parse(rows, count=3), "chain", 1024)
        self.assertEqual(stats["between_kernel_gap"]["samples_us"], [1000, 100, 0])
        self.assertEqual(stats["between_kernel_gap"]["p50_us"], 100)
        self.assertNotEqual(stats["pair_span"]["p50_us"] - stats["pair_busy"]["p50_us"], 100)

    def test_reject_cross_stream_pairs(self):
        rows = [(*row, "0" if index < 3 else "1") for index, row in enumerate(self.valid())]
        with self.assertRaisesRegex(RuntimeError, "stream_id"):
            self.parse(rows, fields=["Name", "Task Type", "Start Time(us)", "Duration(us)", "Stream ID"])


if __name__ == "__main__":
    unittest.main()
