"""Plan→Transfer实验公共逻辑。所有初始化、恢复、golden、D2H检查均在目标调用外。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

import compare as plan
import transfer_variants as transfer_catalog
from harness_io import add_storage_arguments
from full_cache_cases import (HIT_PERCENTAGES, make_full_cache_case, payload_pattern,
                              physical_tokens, reference_transfer_ids, assert_case, describe)

ROOT = Path(__file__).resolve().parent
PLAN_KERNEL = "sparse_kv_plan_kernel_plan_abcdf"
TRANSFER_KERNEL = "sparse_kv_transfer_kernel"
GUARD_BYTES = 64
GUARD_VALUE = 0xa7
CHUNK_BYTES = 8 * 1024 * 1024
GLM_PRESET = {
    "name": "glm_moe_dsa_bf16", "cache_dtype": "bfloat16", "cache_heads": 1,
    "kv_lora_rank": 512, "qk_rope_head_dim": 64,
    "token_bytes_k": 1024, "token_bytes_v": 128,
    "meaning_k": "compressed KV", "meaning_v": "RoPE K",
    "scope": "one layer, one device; indexer cache excluded",
}


def add_arguments(parser, transfer_selectors=True):
    add_storage_arguments(parser)
    parser.add_argument("--launch-mode", choices=("legacy", "prepared"), default="legacy")
    if transfer_selectors:
        parser.add_argument("--transfer-variant", choices=tuple(transfer_catalog.VARIANTS), default="baseline")
        parser.add_argument("--transfer-core-limit", type=int, default=64)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rows", type=int, default=32, help="本实验MTP关闭，一行一个请求")
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--capacity", type=int, default=4096)
    parser.add_argument("--max-token", type=int, default=262144)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--hit-percentages", nargs="+", type=int, choices=HIT_PERCENTAGES,
                        default=list(HIT_PERCENTAGES))
    parser.add_argument("--payload-preset", choices=(GLM_PRESET["name"], "explicit"),
                        default=GLM_PRESET["name"])
    parser.add_argument("--token-bytes-k", type=int)
    parser.add_argument("--token-bytes-v", type=int)
    parser.add_argument("--identity-pages", action="store_true", help="默认物理页随机排列且跨请求不共享")
    parser.add_argument("--ordered-lru", action="store_true")
    parser.add_argument("--library", type=Path, default=ROOT / "build_plan_transfer/libsimt_plan_variants_ops.so")


def resolve_payload(args, parser):
    if not 1 <= args.transfer_core_limit <= 64:
        parser.error("transfer-core-limit必须为1..64；实际核数还受平台限制")
    if args.device < 0 or len(args.hit_percentages) != len(set(args.hit_percentages)):
        parser.error("device必须非负，命中率不能重复")
    explicit = args.token_bytes_k is not None or args.token_bytes_v is not None
    if explicit and (args.token_bytes_k is None or args.token_bytes_v is None):
        parser.error("K/V字节参数必须同时给出")
    if args.payload_preset == "explicit" and not explicit:
        parser.error("--payload-preset explicit需要--token-bytes-k和--token-bytes-v")
    if not explicit:
        args.token_bytes_k, args.token_bytes_v = GLM_PRESET["token_bytes_k"], GLM_PRESET["token_bytes_v"]
    else:
        args.payload_preset = "explicit"
    if not (0 < args.token_bytes_k <= plan.I32_MAX and 0 < args.token_bytes_v <= plan.I32_MAX):
        parser.error("每token字节数必须为正且不超过int32上限")
    return dict(GLM_PRESET) if args.payload_preset != "explicit" else {
        "name": "explicit", "token_bytes_k": args.token_bytes_k, "token_bytes_v": args.token_bytes_v,
        "scope": "raw byte stride supplied by caller; no inferred dtype/layout",
    }


def load_runtime(args):
    import torch
    import torch_npu
    if not torch.npu.is_available() or not args.library.is_file():
        raise RuntimeError("需要可用NPU及本分支共享库，不能跳过设备验证")
    torch.npu.set_device(args.device)
    # 在申请Host映射前建立本卡ACL context，避免set_device的惰性初始化差异。
    torch.empty(1, dtype=torch.uint8, device=f"npu:{args.device}")
    torch.npu.synchronize()
    torch.ops.load_library(str(args.library.resolve()))
    from checks.plan_build_info import catalog_identity
    from checks.transfer_build_info import identity
    if json.loads(torch.ops.sparse_kv_plan_op.plan_variants_info()) != catalog_identity():
        raise RuntimeError("Plan库身份过期，请重建")
    actual = json.loads(torch.ops.sparse_kv_plan_op.transfer_source_info())
    if actual["source"] != identity():
        raise RuntimeError("Transfer/测试框架库身份过期，请重建")
    return torch, torch_npu, {
        "device": args.device, "device_name": torch.npu.get_device_name(args.device),
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "library": str(args.library.resolve()),
        "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
        "source": actual, "plan_compiler": json.loads(torch.ops.sparse_kv_plan_op.plan_compiler_info()),
    }


def fill_pattern_cpu(destination, width, kind):
    view = destination.numpy().reshape(-1, width)
    per_chunk = max(1, CHUNK_BYTES // width)
    next_progress = 1 << 30
    for start in range(0, len(view), per_chunk):
        end = min(len(view), start + per_chunk)
        view[start:end] = payload_pattern(np.arange(start, end, dtype=np.int64), width, kind)
        if end * width >= next_progress:
            print(f"  Host payload已初始化 {end * width / 2**30:.1f} GiB", flush=True)
            next_progress += 1 << 30


def prepare_sources(torch, case, bytes_k, bytes_v):
    """真实Host锁页并映射成DVA；不允许普通CPU地址或HBM偷偷替代Host源。"""
    tokens = case["host_num_blocks"] * case["block_size"]
    pools, addresses = {}, []
    for kind, key, width in ((1, "k", bytes_k), (2, "v", bytes_v)):
        print(f"初始化Host {key.upper()}池: {tokens * width / 2**30:.3f} GiB", flush=True)
        pool, address = torch.ops.sparse_kv_plan_op.mapped_host_empty(tokens * width)
        if pool.device.type != "cpu" or not address:
            raise RuntimeError("Host映射返回值无效")
        fill_pattern_cpu(pool, width, kind)
        pools[key] = pool
        addresses.append(address)
    pools["bases"] = torch.tensor(addresses, dtype=torch.int64, device=f"npu:{torch.npu.current_device()}")
    torch.npu.synchronize()
    return pools


def prepare_case(torch, case, sources, bytes_k, bytes_v, launch_mode="legacy",
                 transfer_variant="baseline", core_limit=64):
    golden = plan.reference_simt(case)
    assert_case(case, golden)
    expected_ids = reference_transfer_ids(case, golden["miss_count"], golden["miss_tokens"], golden["miss_slots"])
    np.testing.assert_array_equal(expected_ids, physical_tokens(case, golden["slot_to_token"]))
    device = torch.device(f"npu:{torch.npu.current_device()}")
    inputs, initial, states = plan.prepare_npu(torch, case, None, device, ("abcdf",))
    state = states["abcdf"]
    state["sources"] = sources  # 强引用，必须活到所有异步设备任务结束。
    state["payload"] = {}
    initial_ids = physical_tokens(case, case["slot_to_token"]).reshape(-1)
    for kind, key, width in ((1, "k", bytes_k), (2, "v", bytes_v)):
        size = case["rows"] * case["capacity"] * width
        storage = torch.full((size + 2 * GUARD_BYTES,), GUARD_VALUE, dtype=torch.uint8, device=device)
        tensor = storage[GUARD_BYTES:GUARD_BYTES + size].view(case["rows"], case["capacity"], width)
        per_chunk = max(1, CHUNK_BYTES // width)
        flat = tensor.view(-1, width)
        for start in range(0, len(initial_ids), per_chunk):
            end = min(len(initial_ids), start + per_chunk)
            part = payload_pattern(initial_ids[start:end], width, kind)
            flat[start:end].copy_(torch.from_numpy(part).to(device))
        state["payload"][key] = {"tensor": tensor, "initial": tensor.clone(),
                                  "storage": storage, "width": width, "kind": kind}
    torch.npu.synchronize()
    configure_transfer(torch, case, inputs, state, launch_mode, transfer_variant, core_limit)
    return inputs, initial, state, golden, expected_ids


def configure_transfer(torch, case, inputs, state, launch_mode, transfer_variant, core_limit):
    """计时外更换启动上下文；保留所有输入、输出、payload及恢复快照地址。"""
    previous = state.pop("prepared", None)
    if previous is not None:
        previous.close()
    state.pop("prepared_info", None)
    state["launch_mode"] = launch_mode
    bytes_k, bytes_v = (state["payload"][key]["width"] for key in ("k", "v"))
    state["transfer_info"] = transfer_catalog.layout(transfer_variant, bytes_k, bytes_v, case["block_size"])
    state["transfer_info"]["requested_core_limit"] = core_limit
    state["transfer_info"]["actual_blocks"] = int(torch.ops.sparse_kv_plan_op.transfer_block_dim_info(core_limit))
    if launch_mode == "prepared":
        state["prepared"] = torch.classes.sparse_kv_plan_op.PreparedPlanTransfer(
            *prepared_arguments(case, inputs, state))
        state["prepared_info"] = json.loads(state["prepared"].info())
        torch.npu.synchronize()
    elif launch_mode != "legacy":
        raise ValueError("未知launch_mode")


def binding_hash(inputs, state):
    tensors = dict(inputs)
    tensors.update({key: state[key] for key in (*plan.SIMT_STATE, *plan.SIMT_OUTPUT, "workspace")})
    tensors.update({"payload_" + key: value["tensor"] for key, value in state["payload"].items()})
    tensors.update({"host_" + key: value for key, value in state["sources"].items()})
    records = [(key, tensor.data_ptr(), str(tensor.device), tuple(tensor.shape))
               for key, tensor in sorted(tensors.items())]
    return hashlib.sha256(repr(records).encode()).hexdigest()


def prepared_arguments(case, inputs, state):
    """与C++上下文绑定的唯一顺序；每次run用同一组Tensor做元信息校验。"""
    p = [inputs[key] for key in ("req_ids", "topk_indices", "stable_prefix_lens",
        "visible_seq_lens", "token_to_req", "block_table", "active_rows")]
    p += [state[key] for key in (*plan.SIMT_STATE, *plan.SIMT_OUTPUT, "workspace")]
    sources = state["sources"]
    x = [sources["bases"], sources["k"], sources["v"],
         state["payload"]["k"]["tensor"], state["payload"]["v"]["tensor"]]
    attrs = [case[key] for key in ("topk", "capacity", "max_token", "block_size", "host_num_blocks")]
    attrs += [state["payload"][key]["width"] for key in ("k", "v")]
    attrs += [state["transfer_info"]["mask"], state["transfer_info"]["requested_core_limit"]]
    return p, x, attrs


def restore(torch, initial, state, mode, poison=False):
    if mode != "transfer":
        for key, value in initial["abcdf"].items():
            state[key].copy_(value)
        if poison:
            for key in plan.SIMT_OUTPUT:
                state[key].fill_(-777)
            state["workspace"].fill_(plan.I32_MAX)
    # 包括miss目标在内的全部payload复原，确保Transfer未写时不能读到上一轮答案。
    for item in state["payload"].values():
        item["tensor"].copy_(item["initial"])
    torch.npu.synchronize()


def invoke(torch, mode, case, inputs, state):
    if state.get("launch_mode") == "prepared":
        state["prepared"].run(*prepared_arguments(case, inputs, state), mode)
        return
    if mode != "transfer":
        plan.invoke(torch, "abcdf", case, inputs, state)
    if mode != "plan":
        sources = state["sources"]
        torch.ops.sparse_kv_plan_op.sparse_kv_transfer(
            state["miss_count"], state["miss_tokens"], state["miss_slots"],
            inputs["token_to_req"], inputs["block_table"], sources["bases"], inputs["active_rows"],
            sources["k"], sources["v"], state["payload"]["k"]["tensor"], state["payload"]["v"]["tensor"],
            case["topk"], case["capacity"], case["block_size"], case["host_num_blocks"],
            state["payload"]["k"]["width"], state["payload"]["v"]["width"],
            state["transfer_info"]["mask"], state["transfer_info"]["requested_core_limit"])
    # chain中不手动插入Plan→Transfer同步；同一current stream依赖保证先后。


def check_payload(state, expected_ids, phase):
    flat_ids = np.asarray(expected_ids, dtype=np.int64).reshape(-1)
    for key, item in state["payload"].items():
        storage, width, kind = item["storage"], item["width"], item["kind"]
        np.testing.assert_array_equal(storage[:GUARD_BYTES].cpu().numpy(), GUARD_VALUE,
                                      err_msg=phase + "/" + key + "/prefix-guard")
        np.testing.assert_array_equal(storage[-GUARD_BYTES:].cpu().numpy(), GUARD_VALUE,
                                      err_msg=phase + "/" + key + "/suffix-guard")
        flat = item["tensor"].view(-1, width)
        per_chunk = max(1, CHUNK_BYTES // width)
        for start in range(0, len(flat_ids), per_chunk):
            end = min(len(flat_ids), start + per_chunk)
            np.testing.assert_array_equal(flat[start:end].cpu().numpy(),
                payload_pattern(flat_ids[start:end], width, kind),
                err_msg=f"{phase}/{key}/slots-{start}:{end}")


def check_result(case, state, golden, expected_ids, mode, phase):
    plan.check_npu_result("abcdf", state, {"abcdf": golden}, phase)
    check_payload(state, physical_tokens(case, case["slot_to_token"]) if mode == "plan" else expected_ids, phase)


def prime_transfer(torch, case, inputs, initial, state, golden):
    """Transfer-only也由真实ABCDF产生miss输入，禁止用golden重写设备miss输出。"""
    restore(torch, initial, state, "plan", poison=True)
    invoke(torch, "plan", case, inputs, state)
    torch.npu.synchronize()
    plan.check_npu_result("abcdf", state, {"abcdf": golden}, "transfer-only/real-plan")


def audit(torch, case, inputs, initial, state, golden, ids, mode, repeats=3):
    if mode == "transfer":
        prime_transfer(torch, case, inputs, initial, state, golden)
    for iteration in range(repeats):
        restore(torch, initial, state, mode, poison=True)
        invoke(torch, mode, case, inputs, state)
        torch.npu.synchronize()
        check_result(case, state, golden, ids, mode, f"audit/{mode}/{iteration}")


def case_from_args(args, percent):
    return make_full_cache_case(args.rows, args.topk, args.capacity, args.max_token, percent,
                                args.seed, args.block_size, not args.ordered_lru, not args.identity_pages)
