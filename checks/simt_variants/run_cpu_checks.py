#!/usr/bin/env python3
"""Execute the actual SIMT row/scan source through a CPU semantic shim, never NPU timing."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def prepare_build(variant, build, threads):
    build.mkdir(parents=True, exist_ok=True)
    config = (variant / "sparse_kv_plan_config.h").read_text()
    config = config.replace("PLAN_THREADS = 2048U", f"PLAN_THREADS = {threads}U")
    (build / "sparse_kv_plan_config.h").write_text(config)
    kernel = (variant / "sparse_kv_plan_compute.h").read_text()
    prefix = kernel
    (build / "kernel_under_test.h").write_text(prefix)
    for name in ["kernel_operator.h", "simt_api/asc_simt.h", "simt_api/device_atomic_functions.h",
                 "simt_api/device_sync_functions.h", "simt_api/device_warp_functions.h"]:
        stub = build / name
        stub.parent.mkdir(exist_ok=True)
        stub.write_text("// CPU stub: API implementations live in cpu_runtime.h.\n")


def compile_and_run(variant, build, driver, threads, arrays, extra=()):
    prepare_build(variant, build, threads)
    binary = build / "check"
    subprocess.run(["g++", "-std=c++20", "-O1", "-g", "-pthread", "-no-pie",
                    "-fsanitize=address,undefined", "-fno-omit-frame-pointer", f"-DCAPACITY_ARRAYS={arrays}",
                    "-I", str(build), "-I", str(HERE), str(HERE / driver), "-o", str(binary)], check=True)
    env = dict(os.environ, ASAN_OPTIONS="detect_leaks=0:halt_on_error=1", UBSAN_OPTIONS="halt_on_error=1")
    result = subprocess.run([str(binary), *extra], check=True, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="*")
    parser.add_argument("--threads", type=int, choices=[32, 64, 96, 128], default=64)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--large", action="store_true", help="also check 512/256, 4096/2048, 8192/2048")
    parser.add_argument("--scan-threads", nargs="*", type=int, default=[2048],
                        choices=[32, 64, 96, 1024, 1056, 2048, 3072])
    args = parser.parse_args()
    manifest = json.loads((ROOT / "variants_manifest.json").read_text())
    names = {v["name"] for v in manifest["variants"]}
    if args.variants and not set(args.variants) <= names:
        parser.error("unknown variant name")
    records = []
    with tempfile.TemporaryDirectory(prefix="simt_plan_cpu_checks_") as temporary:
        builds = Path(temporary)
        for variant in manifest["variants"]:
            name = variant["name"]
            if args.variants and name not in args.variants:
                continue
            path = ROOT / "csrc/variants" / name
            actual_hash = hashlib.sha256((path / "sparse_kv_plan_compute.h").read_bytes()).hexdigest()
            if actual_hash != variant["compute_sha256"]:
                raise RuntimeError(f"{name}: manifest source hash is stale")
            output = compile_and_run(path, builds / name / "row", "cpu_row_driver.cpp",
                                     args.threads, variant["capacity_arrays"], ["--large"] if args.large else [])
            print(name, output, flush=True)
            record = {"variant": name, "row_result": output}
            if "B" in variant["points"]:
                record["scan_results"] = []
                for threads in args.scan_threads:
                    scan = compile_and_run(path, builds / name / f"scan_{threads}", "scan_driver.cpp", threads,
                                           variant["capacity_arrays"])
                    print(name, scan, flush=True)
                    record["scan_results"].append(scan)
            records.append(record)
    report = {"kind": "CPU semantic checks with ASAN/UBSAN; not CANN/NPU validation",
              "baseline_commit": manifest["baseline_commit"], "row_test_threads": args.threads, "results": records}
    report["large_shapes"] = args.large
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
