"""联合Plan直调公共层：prepared上下文、状态恢复、保护区及语义门禁。"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
import numpy as np
import peer_plan
from joint_variants import supports_eight_rows,supports_single_sorted,INTEGRATED_NAMES
from harness_io import add_storage_arguments
from joint_plan_cases import INPUTS, PERSISTENT, OUTPUTS, POISON, layout, initial_arrays, validate, validate_abcdf, require_sorted_topk

ROOT = Path(__file__).resolve().parent
from joint_variants import hot_config, VARIANTS, DEFAULT_IMPLEMENTATIONS, is_joint, description, TURBO_MASKS, MERGE_NAMES, SORTED_NAMES
PAGED_NAMES = ("abcdf_pages", "abcdf_pages_a", "abcdf_pages_b", "abcdf_pages_ab")
INDEPENDENT = ("abcdf", *PAGED_NAMES)
IMPLEMENTATIONS = (peer_plan.NAME, "abcdf", "abcdf_pages", "abcdf_pages_a", "abcdf_pages_b", "abcdf_pages_ab", "private", "joint", *VARIANTS)
KERNELS = {"abcdf_pages": "sparse_kv_plan_kernel_plan_abcdf_pages", "abcdf": "sparse_kv_plan_kernel_plan_abcdf",
           "private": "sparse_kv_plan_private_kernel", "joint": "sparse_kv_plan_joint_kernel"}
KERNELS.update({name: "sparse_kv_plan_" + name.split("_", 1)[0] + "_kernel_" + name.split("_", 1)[1]
                for name in VARIANTS})
KERNELS[peer_plan.NAME] = peer_plan.KERNEL
KERNELS.update({n: "sparse_kv_plan_kernel_plan_" + n for n in PAGED_NAMES})
ABCDF_STATE = ("last_req_ids", "slot_to_token", "lru_slots")
ABCDF_OUTPUT = ("current_slots", "miss_count", "miss_tokens", "miss_slots")
TENSOR_ORDER = (*INPUTS, "active_rows", "last_req_ids", "slot_to_token", "lru_slots",
                "current_slots", "miss_count", "miss_tokens", "miss_slots", "workspace",
                "row_modes", "resident_rows", "resident_block_table")
GUARD = 16
GUARD_VALUE = 0x5A5A5A5A


def add_runtime_arguments(parser):
    add_storage_arguments(parser)
    parser.set_defaults(temp_dir=Path(os.environ.get("TMPDIR", "/workspace/joint_plan_results/tmp")))
    parser.add_argument("--device", type=int, default=2)
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
        "hostname": platform.node(),
        "cann_package_home": os.environ.get("ASCEND_CANN_PACKAGE_HOME", os.environ.get("ASCEND_HOME_PATH")),
        "torch": torch.__version__, "torch_npu": torch_npu.__version__, "identity": info,
        "library": str(args.library.resolve()), "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest()}


def upload_inputs(torch, case):
    device = f"npu:{torch.npu.current_device()}"
    result = {key: torch.from_numpy(np.ascontiguousarray(case[key])).to(device) for key in INPUTS}
    result["active_rows"] = torch.tensor([case.get("active_rows", case["rows"])], dtype=torch.int32, device=device)
    return result


def update_inputs(torch, bundle, case):
    if bundle["name"] in MERGE_NAMES or bundle["name"] == peer_plan.NAME:
        require_sorted_topk(case)
    for key in INPUTS:
        bundle["inputs"][key].copy_(torch.from_numpy(np.ascontiguousarray(case[key])).to(bundle["inputs"][key].device))
    bundle["inputs"]["active_rows"].fill_(case.get("active_rows", case["rows"]))


def prepare(torch, case, name, inputs=None):
    if name not in IMPLEMENTATIONS:
        raise ValueError("Unknown implementation: " + name)
    if name in MERGE_NAMES or name == peer_plan.NAME:
        require_sorted_topk(case)
    # Integrated entry groups actual tokenToReq runs even if a fixture calls it MTP0.
    enabled = name == peer_plan.NAME or (is_joint(name) and (case["mtp"] > 0 or name in INTEGRATED_NAMES))
    union_next = name.startswith("joint_union_")
    controller_enabled = enabled or supports_single_sorted(name)
    is_turbo = VARIANTS.get(name) in TURBO_MASKS
    info = layout(case["topk"], case["capacity"], case["rows"], controller_enabled, case["max_token"],
                  name == "joint_bitmap", is_turbo, name in SORTED_NAMES, union_next=union_next, hot_packet=bool((hot_config(name) or {}).get("packet")),mtp8=supports_eight_rows(name))
    if name == peer_plan.NAME:
        info = peer_plan.layout(case["topk"], case["capacity"], case["rows"], case["q_len"])
    inputs = upload_inputs(torch, case) if inputs is None else inputs
    if name in INDEPENDENT:
        host = {key: case[key].copy() for key in ABCDF_STATE}
        host.update({key: np.full((case["rows"],) if key == "miss_count" else
                                  (case["rows"], case["topk"]), POISON, np.int32) for key in ABCDF_OUTPUT})
    else:
        max_queries = (info["scratch_elements"] - 2 * info["joint_hash_capacity"] - case["capacity"]
                       if is_turbo and enabled else None)
        if name in SORTED_NAMES: max_queries=(8 if supports_eight_rows(name) else 4)*case["topk"] if enabled else None
        if name == peer_plan.NAME: max_queries = case["q_len"] * case["topk"]
        host = initial_arrays(case, enabled, max_queries)
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
    ordered = [all_tensors[key] for key in (TENSOR_ORDER[:14] if name in INDEPENDENT else TENSOR_ORDER)]
    attrs = [case[key] for key in (("topk", "capacity", "max_token") if name in INDEPENDENT
                                   else ("topk", "capacity", "max_token", "block_size"))]
    context = None
    if name not in INDEPENDENT and name != peer_plan.NAME:
        attrs.append(int(controller_enabled))
        if name in VARIANTS:
            context = torch.classes.mtp_joint_plan.PreparedVariant(ordered, attrs, is_joint(name), VARIANTS[name])
        else:
            context = torch.classes.mtp_joint_plan.Prepared(ordered, attrs, is_joint(name))
    if name in PAGED_NAMES or name == peer_plan.NAME:
        blocks = (case["max_token"] + case["block_size"] - 1) // case["block_size"]
        host_blocks = case["requests"] * blocks
        if host_blocks > np.iinfo(np.int32).max:
            raise ValueError("source physical block id exceeds int32")
        table = np.random.default_rng(case["seed"] + 90001).permutation(host_blocks).astype(np.int32).reshape(case["requests"], blocks)
        device_table = torch.from_numpy(table).to(device)
        inputs = dict(inputs, block_table=device_table)
        if name == peer_plan.NAME:
            ordered.append(device_table)
            attrs += [1, case["q_len"]]
            context = torch.classes.mtp_joint_plan.PreparedPeerQli(ordered, attrs, True)
        else:
            ordered.insert(5, device_table)
            attrs += [case["block_size"], host_blocks]
    torch.npu.synchronize()
    native = json.loads(context.info()) if context is not None else {
        "joint_enabled": 0, "dynamic_ub_bytes": info["dynamic_ub_bytes"],
        "interface": "independent ABCDF without source-page validation, 14 tensors + 3 attributes (no block_table)",
        "tiling_upload": "per_call_original_interface; excluded by kernel-duration filter"}
    if native["dynamic_ub_bytes"] != info["dynamic_ub_bytes"] or native["joint_enabled"] != int(controller_enabled):
        raise AssertionError("Python/C++ tiling模型不一致")
    if name in VARIANTS and native.get("variant_mask") != VARIANTS[name]:
        raise AssertionError("Prepared variant mask mismatch")
    if name == peer_plan.NAME:
        native.update(info, interface="standalone peer Plan,18tensors/6attrs",
                      page_table_sha256=hashlib.sha256(table.tobytes()).hexdigest())
    native["logical_joint_enabled"] = bool(enabled)
    native["single_sorted_requested"] = (supports_single_sorted(name) or name == peer_plan.NAME) and case["mtp"] == 0
    if name in PAGED_NAMES:
        native["interface"] = "independent ABCDF with source-page validation, 15 tensors + 5 attributes"
        native["page_table_sha256"] = hashlib.sha256(table.tobytes()).hexdigest()
    native["experiment"] = ({"family": "independent paged ABCDF", "optimization_mask":
        {"abcdf_pages": 0, "abcdf_pages_a": 1, "abcdf_pages_b": 2, "abcdf_pages_ab": 3}[name],
        "A": "warm scratch token forwarding", "B": "ping-pong register-total Scan"}
        if name in PAGED_NAMES else peer_plan.description() if name == peer_plan.NAME else description(name))
    return {"name": name, "torch": torch, "invocation_count": 0,
            "workload": {key: case.get(key) for key in ("scenario", "overlap", "mtp", "rows", "requests", "topk", "capacity", "max_token", "seed")}, "joint_enabled": enabled, "inputs": inputs, "state": tensors,
            "buffers": buffers, "initial": initial, "host_initial": host,
            "page_table_host": table.copy() if name in PAGED_NAMES or name == peer_plan.NAME else None, "ordered": ordered, "attrs": attrs, "context": context, "layout": info | native}


def restore(torch, bundle):
    for key, source in bundle["initial"].items():
        bundle["state"][key].copy_(source)
    torch.npu.synchronize()


def invoke(bundle):
    bundle["invocation_count"] = bundle.get("invocation_count", 0) + 1
    if bundle["name"] in INDEPENDENT:
        getattr(bundle["torch"].ops.sparse_kv_plan_op, "plan_" + bundle["name"])(*bundle["ordered"], *bundle["attrs"])
    else:
        bundle["context"].run(bundle["ordered"], bundle["attrs"])


def close(bundle):
    if bundle["context"] is not None:
        bundle["context"].close()


def snapshot(bundle):
    keys = (*ABCDF_STATE, *ABCDF_OUTPUT) if bundle["name"] in INDEPENDENT else (*PERSISTENT, *OUTPUTS)
    cached=bundle.get("_readback")
    return {key: (cached["state/"+key].copy() if cached is not None else bundle["state"][key].cpu().numpy().copy()) for key in keys}


def poison(bundle):
    for key in (ABCDF_OUTPUT if bundle["name"] in INDEPENDENT else OUTPUTS):
        bundle["state"][key].fill_(POISON)
    bundle["state"]["workspace"].fill_(np.iinfo(np.int32).max)


def _check_uncached(case, bundle, before=None):
    before = bundle["host_initial"] if before is None else before
    cached=bundle.get("_readback")
    if cached is not None:
        _check_readback(case,bundle,cached)
    else:
        if bundle["name"] in PAGED_NAMES or bundle["name"] == peer_plan.NAME:
            np.testing.assert_array_equal(bundle["inputs"]["block_table"].cpu().numpy(),bundle["page_table_host"],err_msg="read-only/block_table")
        for key, storage in bundle["buffers"].items():
            for guard in (storage[:GUARD], storage[-GUARD:]):
                np.testing.assert_array_equal(guard.cpu().numpy(),np.full(GUARD,GUARD_VALUE),err_msg="guard/"+key)
        for key in INPUTS:
            np.testing.assert_array_equal(bundle["inputs"][key].cpu().numpy(),case[key],err_msg="input write/"+key)
        np.testing.assert_array_equal(bundle["inputs"]["active_rows"].cpu().numpy(),[case.get("active_rows",case["rows"])],err_msg="active_rows write")
    if bundle["name"] in INDEPENDENT:
        result = validate_abcdf(case, before, snapshot(bundle))
        if bundle["name"] in PAGED_NAMES:
            result["protocol"] = "independent paged ABCDF, 15 tensors; upstream guarantees valid source pages"
        return result
    if bundle["name"] == peer_plan.NAME:
        return validate(case, before, snapshot(bundle), True, case["q_len"] * case["topk"],
                        merge=True, float_keys=True, single_sorted=True,
                        max_group_rows=case["q_len"], require_power_two_groups=False)
    layout_info = bundle.get("layout", {})
    max_queries = (layout_info["scratch_elements"] - 2 * layout_info["joint_hash_capacity"] - case["capacity"]
                   if VARIANTS.get(bundle["name"]) in TURBO_MASKS and bundle["joint_enabled"] else None)
    if bundle["name"] in SORTED_NAMES: max_queries=(8 if supports_eight_rows(bundle["name"]) else 4)*case["topk"] if bundle["joint_enabled"] else None
    return validate(case, before, snapshot(bundle), bundle["joint_enabled"], max_queries, merge=bundle["name"] in MERGE_NAMES, float_keys=bundle["name"] in SORTED_NAMES, preserve_tail=(bundle["name"].startswith("joint_union_") and bool((VARIANTS[bundle["name"]]-74)&8)), single_sorted=supports_single_sorted(bundle["name"]),max_group_rows=8 if supports_eight_rows(bundle["name"]) else 4)


def _check_readback(case,bundle,values):
    for key in bundle['buffers']:
        for side in ('left','right'):
            np.testing.assert_array_equal(values[side+'/'+key],np.full(GUARD,GUARD_VALUE),err_msg='guard/'+key)
    for key in INPUTS:
        np.testing.assert_array_equal(values['input/'+key],case[key],err_msg='input write/'+key)
    np.testing.assert_array_equal(values['input/active_rows'],[case.get('active_rows',case['rows'])],err_msg='active_rows write')
    if bundle['name'] in PAGED_NAMES or bundle['name'] == peer_plan.NAME:
        np.testing.assert_array_equal(values['input/block_table'],bundle['page_table_host'],err_msg='read-only/block_table')


def check(case,bundle,before=None):
    start=time.monotonic()
    try:
        if not bundle.get('fast_fixed_check') or (before is not None and not bundle.get('_fixed_audit')):
            return _check_uncached(case,bundle,before)
        base=bundle['host_initial'] if before is None else before
        from fixed_plan_check import FixedExpectation,batched_readback
        t=time.monotonic();bundle['_readback']=batched_readback(bundle)
        timing=bundle.setdefault('check_timing',{'readback_seconds':0.,'reference_seconds':0.,'compare_seconds':0.,'reference_builds':0,'cache_hits':0})
        timing['readback_seconds']+=time.monotonic()-t
        _check_readback(case,bundle,bundle['_readback'])
        t=time.monotonic();cached=bundle.get('_fixed_expectation')
        if cached is None:
            result=_check_uncached(case,bundle,before)
            bundle['_fixed_expectation']=FixedExpectation(case,base,snapshot(bundle),result)
            timing['reference_builds']+=1;timing['reference_seconds']+=time.monotonic()-t
            return result
        result=cached.check(case,base,snapshot(bundle))
        timing['cache_hits']+=1;timing['compare_seconds']+=time.monotonic()-t
        return result
    finally:
        bundle.pop('_readback',None)
        bundle['check_seconds']=bundle.get('check_seconds',0.)+time.monotonic()-start


def dump_failure(case, bundle, destination):
    """只在运行机器保存数值现场；二次采集/磁盘错误不覆盖原异常，不恢复输入。"""
    try:
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / (bundle["name"] + "_failure_context.json")).write_text(json.dumps({
            "implementation": bundle["name"], "invocation_count": bundle.get("invocation_count", 0),
            "phase": bundle.get("phase"), "workload": bundle.get("workload", {})}, ensure_ascii=False, indent=2))
        values = {"input_actual_" + key: tensor.cpu().numpy().copy()
                  for key, tensor in bundle["inputs"].items()}
        values.update({"input_expected_" + key: case[key] for key in INPUTS})
        values.update({"state_" + key: value for key, value in snapshot(bundle).items()})
        values.update({"guarded_storage_" + key: value.cpu().numpy().copy()
                       for key, value in bundle["buffers"].items()})
        np.savez(destination / (bundle["name"] + "_failure_state.npz"), **values)
    except BaseException as secondary:
        import sys
        try:
            print(f"现场保存失败(保留原异常): {secondary}", file=sys.stderr, flush=True)
        except BaseException:
            pass


def audit(torch, case, bundle, repeats=5, failure_dir=None, progress_every=0):
    result = None
    for iteration in range(repeats):
        bundle["phase"] = f"audit-{iteration + 1}/{repeats}"
        try:
            restore(torch, bundle)
            poison(bundle)
            torch.npu.synchronize()
            before = snapshot(bundle)
            invoke(bundle)
            torch.npu.synchronize()
            bundle["_fixed_audit"]=True
            try:result = check(case,bundle,before)
            finally:bundle.pop("_fixed_audit",None)
            if progress_every and (iteration + 1) % progress_every == 0:
                print(f"输入错写回归: {iteration + 1}/{repeats} passed", flush=True)
        except BaseException as exc:
            if hasattr(exc, "add_note"):
                exc.add_note(f"{bundle['name']} audit iteration {iteration + 1}/{repeats}")
            if failure_dir is not None:
                dump_failure(case, bundle, failure_dir)
            raise
    return result
