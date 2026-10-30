"""联合Plan直调公共层：prepared上下文、状态恢复、保护区及语义门禁。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from harness_io import add_storage_arguments
from joint_plan_cases import INPUTS, PERSISTENT, OUTPUTS, POISON, layout, initial_arrays, validate

ROOT = Path(__file__).resolve().parent
IMPLEMENTATIONS = ("private", "joint")
KERNELS = {"private": "sparse_kv_plan_private_kernel", "joint": "sparse_kv_plan_joint_kernel"}
TENSOR_ORDER = (*INPUTS, "active_rows", "last_req_ids", "slot_to_token", "lru_slots",
                "current_slots", "miss_count", "miss_tokens", "miss_slots", "workspace",
                "row_modes", "resident_rows", "resident_block_table")
GUARD = 16
GUARD_VALUE = 0x5A5A5A5A


def add_runtime_arguments(parser):
    add_storage_arguments(parser)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--library", type=Path, default=ROOT / "build_joint_plan/libmtp_joint_plan_ops.so")


def load_runtime(args):
    if args.device < 0:
        raise ValueError("device必须非负")
    from checks.joint_build_info import identity
    expected = identity()  # 源码身份先检查，过期源码不能以旧库继续跑。
    import torch
    import torch_npu
    if not torch.npu.is_available() or not args.library.is_file():
        raise RuntimeError("需要可用NPU和本分支build_joint_plan产物，不允许跳过设备验证")
    torch.npu.set_device(args.device)
    torch.empty(1, dtype=torch.int32, device=f"npu:{args.device}")
    torch.npu.synchronize()
    torch.ops.load_library(str(args.library.resolve()))
    info = json.loads(torch.ops.mtp_joint_plan.source_info())
    if info["source"] != expected:
        raise RuntimeError("联合Plan共享库身份过期，请用build_joint_plan.sh重建")
    return torch, torch_npu, {"device": args.device, "device_name": torch.npu.get_device_name(args.device),
        "torch": torch.__version__, "torch_npu": torch_npu.__version__, "identity": info,
        "library": str(args.library.resolve()), "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest()}


def upload_inputs(torch, case):
    device = f"npu:{torch.npu.current_device()}"
    result = {key: torch.from_numpy(np.ascontiguousarray(case[key])).to(device) for key in INPUTS}
    result["active_rows"] = torch.tensor([case.get("active_rows", case["rows"])], dtype=torch.int32, device=device)
    return result


def update_inputs(torch, bundle, case):
    for key in INPUTS:
        bundle["inputs"][key].copy_(torch.from_numpy(np.ascontiguousarray(case[key])).to(bundle["inputs"][key].device))
    bundle["inputs"]["active_rows"].fill_(case.get("active_rows", case["rows"]))


def prepare(torch, case, name, inputs=None):
    enabled = name == "joint" and case["mtp"] > 0
    info = layout(case["topk"], case["capacity"], case["rows"], enabled)
    inputs = upload_inputs(torch, case) if inputs is None else inputs
    host = initial_arrays(case, enabled)
    host["workspace"] = np.full(info["workspace_elements"], np.iinfo(np.int32).max, np.int32)
    tensors, buffers, initial = {}, {}, {}
    device = inputs["req_ids"].device
    for key, array in host.items():
        dtype = torch.int64 if array.dtype == np.int64 else torch.int32
        storage = torch.full((array.size + 2 * GUARD,), GUARD_VALUE, dtype=dtype, device=device)
        view = storage[GUARD:GUARD + array.size].view(array.shape)
        view.copy_(torch.from_numpy(np.ascontiguousarray(array)).to(device))
        tensors[key], buffers[key], initial[key] = view, storage, view.clone()
    all_tensors = inputs | tensors
    ordered = [all_tensors[key] for key in TENSOR_ORDER]
    attrs = [case[key] for key in ("topk", "capacity", "max_token", "block_size", "host_num_blocks")]
    attrs.append(int(enabled))
    context = torch.classes.mtp_joint_plan.Prepared(ordered, attrs, name == "joint")
    torch.npu.synchronize()
    native = json.loads(context.info())
    if native["dynamic_ub_bytes"] != info["dynamic_ub_bytes"] or native["joint_enabled"] != int(enabled):
        raise AssertionError("Python/C++ tiling模型不一致")
    return {"name": name, "joint_enabled": enabled, "inputs": inputs, "state": tensors,
            "buffers": buffers, "initial": initial, "host_initial": host,
            "ordered": ordered, "attrs": attrs, "context": context, "layout": info | native}


def restore(torch, bundle):
    for key, source in bundle["initial"].items():
        bundle["state"][key].copy_(source)
    torch.npu.synchronize()


def invoke(bundle):
    bundle["context"].run(bundle["ordered"], bundle["attrs"])


def snapshot(bundle):
    return {key: bundle["state"][key].cpu().numpy().copy() for key in (*PERSISTENT, *OUTPUTS)}


def poison(bundle):
    for key in OUTPUTS:
        bundle["state"][key].fill_(POISON)
    bundle["state"]["workspace"].fill_(np.iinfo(np.int32).max)


def check(case, bundle, before=None):
    before = bundle["host_initial"] if before is None else before
    for key, storage in bundle["buffers"].items():
        for guard in (storage[:GUARD], storage[-GUARD:]):
            np.testing.assert_array_equal(guard.cpu().numpy(), np.full(GUARD, GUARD_VALUE), err_msg="guard/" + key)
    for key in INPUTS:
        np.testing.assert_array_equal(bundle["inputs"][key].cpu().numpy(), case[key], err_msg="input write/" + key)
    np.testing.assert_array_equal(bundle["inputs"]["active_rows"].cpu().numpy(),
                                 [case.get("active_rows", case["rows"])], err_msg="active_rows write")
    return validate(case, before, snapshot(bundle), bundle["joint_enabled"])


def audit(torch, case, bundle, repeats=3):
    result = None
    for _ in range(repeats):
        restore(torch, bundle)
        poison(bundle)
        torch.npu.synchronize()
        before = snapshot(bundle)
        invoke(bundle)
        torch.npu.synchronize()
        result = check(case, bundle, before)
    return result
