"""采集IO门禁和失败诊断。只检查输出/临时文件系统，不检查源码分区空间。"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback
import uuid


def add_storage_arguments(parser):
    parser.add_argument("--temp-dir", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp")))
    parser.add_argument("--min-output-free-bytes", type=int, default=1 << 30)
    parser.add_argument("--min-temp-free-bytes", type=int, default=1 << 30)
    parser.add_argument("--min-free-inodes", type=int, default=1024)


def inspect_directory(path, minimum_bytes, minimum_inodes):
    path = Path(path).expanduser().resolve()
    free_bytes = free_inodes = None
    anchor = path
    try:
        # 父分区已满时mkdir也可能失败；先取得最近现存目录的容量供报错。
        while not anchor.exists() and anchor != anchor.parent:
            anchor = anchor.parent
        stat = os.statvfs(anchor)
        free_bytes = stat.f_bavail * stat.f_frsize
        free_inodes = stat.f_favail if stat.f_files else None
        path.mkdir(parents=True, exist_ok=True)
        stat = os.statvfs(path)
        free_bytes = stat.f_bavail * stat.f_frsize
        free_inodes = stat.f_favail if stat.f_files else None
        if free_bytes < minimum_bytes or (free_inodes is not None and free_inodes < minimum_inodes):
            raise OSError("可用空间/inode低于配置阈值")
        # 仅删除本函数创建的空探针，不删除任何已有文件或采集结果。
        probe = path / (".write_probe_" + uuid.uuid4().hex)
        fd = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(fd)
        probe.unlink()
    except OSError as exc:
        raise RuntimeError(
            f"采集目录不可用: path={path}, capacity_probe={anchor}, free_bytes={free_bytes}, free_inodes={free_inodes}, "
            f"required_bytes={minimum_bytes}, required_inodes={minimum_inodes}; {exc}. "
            "请释放空间/inode，或显式选择其他 --output/--temp-dir；不会自动删除或切换目录。"
        ) from exc
    return {"path": str(path), "free_bytes": free_bytes, "free_inodes": free_inodes,
            "minimum_bytes": minimum_bytes, "minimum_inodes": minimum_inodes,
            "filesystem_device": os.stat(path).st_dev}


def preflight(output, temp_dir, output_bytes=1 << 30, temp_bytes=1 << 30, inodes=1024):
    if min(output_bytes, temp_bytes, inodes) < 0:
        raise ValueError("空间与inode阈值必须非负；阈值只是最低余量，不是文件大小预测")
    destination = inspect_directory(output, output_bytes, inodes)
    temporary = inspect_directory(temp_dir, temp_bytes, inodes)
    if destination["filesystem_device"] == temporary["filesystem_device"]:
        combined = output_bytes + temp_bytes
        if min(destination["free_bytes"], temporary["free_bytes"]) < combined:
            raise RuntimeError(
                f"输出与临时目录共用文件系统，free_bytes={temporary['free_bytes']}，"
                f"合计最低余量={combined}；output={destination['path']}，temp={temporary['path']}。"
                "请释放空间或显式选择其他 --output/--temp-dir。阈值不预测profiling大小。")
    # 在import torch_npu/创建parser子进程之前设置；同时覆盖tempfile已缓存的旧路径。
    os.environ["TMPDIR"] = temporary["path"]
    tempfile.tempdir = temporary["path"]
    sys.dont_write_bytecode = True
    result = {"output": destination, "temporary": temporary, "size_prediction": False}
    print("profiling文件系统门禁:", json.dumps(result, ensure_ascii=False), flush=True)
    for entry in (destination, temporary):
        if Path(entry["path"]).is_relative_to(Path("/dev/shm").resolve()):
            print(f"提示：{entry['path']} 位于/dev/shm，属于临时存储，请在会话/机器结束前转存结果。", flush=True)
    return result


def preflight_args(args, output):
    return preflight(output, args.temp_dir, args.min_output_free_bytes,
                     args.min_temp_free_bytes, args.min_free_inodes)


def write_json(path, data):
    """同目录原子替换；失败保留pending文件，绝不删除原始采集目录。"""
    path = Path(path)
    pending = path.with_name(path.name + ".pending_" + uuid.uuid4().hex)
    with pending.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def record_failure(path, report, exception):
    """任何二次写盘异常都只告警；调用方必须用bare raise保留原始异常。"""
    report.update(status="failed", failure=f"{type(exception).__name__}: {exception}")
    try:
        write_json(path, report)
    except BaseException as secondary:
        try:
            print(f"失败状态写入失败: {path}: {secondary}; 原始异常: {exception}", file=sys.stderr, flush=True)
        except BaseException:
            pass


@contextmanager
def diagnostic_log(path):
    """捕获Python、C/C++以及继承stdio的profiling解析子进程日志。"""
    path = Path(path)
    print("详细采集日志:", path, flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    with path.open("a", encoding="utf-8") as stream:
        saved_out, saved_err = os.dup(1), os.dup(2)
        try:
            os.dup2(stream.fileno(), 1)
            os.dup2(stream.fileno(), 2)
            try:
                yield
            except BaseException:
                # 日志再次满盘时不能替换原始异常。
                try:
                    traceback.print_exc()
                except BaseException:
                    pass
                raise
        finally:
            for target in (sys.stdout, sys.stderr):
                try:
                    target.flush()
                except BaseException:
                    pass
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
            os.close(saved_out)
            os.close(saved_err)


def diagnostic_details(directory, log_path):
    paths = [Path(log_path)]
    try:
        paths += sorted(Path(directory).rglob("*.log"))[:20]
        paths += sorted(Path(directory).rglob("*.err"))[:20]
    except OSError:
        pass
    first = None
    traceback_location = None
    for path in dict.fromkeys(paths):
        try:
            with path.open(encoding="utf-8", errors="replace") as stream:
                for line_no, line in enumerate(stream, 1):
                    if line_no > 50000:
                        break
                    if "Traceback (" in line and traceback_location is None:
                        traceback_location = f"{path}:{line_no}: {line.strip()}"
                    if any(mark in line for mark in ("Error:", "Exception:", "ERROR", "No space left",
                                                     "Disk quota exceeded", "Errno 28")):
                        first = f"{path}:{line_no}: {line.strip()[:1000]}"
                        break
        except OSError:
            continue
        if first:
            break
    return {"capture_directory": str(directory), "detailed_log": str(log_path),
            "first_locatable_exception": first or traceback_location or "日志中未定位到更早异常；请查看详细日志与原始采集文件"}


def capture_error(directory, log_path, error):
    details = diagnostic_details(directory, log_path)
    return RuntimeError(f"profiling采集/解析失败: {type(error).__name__}: {error}; "
                        f"capture_directory={details['capture_directory']}; "
                        f"detailed_log={details['detailed_log']}; "
                        f"first_exception={details['first_locatable_exception']}")
