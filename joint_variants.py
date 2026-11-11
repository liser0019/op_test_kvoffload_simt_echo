"""Joint A/B/C/D 编译目录；与原 ABCDF 的字母命名完全独立。"""
OPTIONS = {
    "a": "non-joint VF compile-time isolation (protocol retained)",
    "b": "reuse per-query union hash bucket",
    "c": "write resident only on reset/invalidation",
    "d": "one union overflow reduction per row",
}
MASKS = {"".join(letter for bit, letter in enumerate(OPTIONS) if mask & (1 << bit)): mask
         for mask in range(1, 16)}
MASKS["bitmap"] = 16
MASKS["turbo"] = 17
MASKS.update(turbo_a=18, turbo_b=19, turbo_ab=20)
MASKS["merge"] = 21
MERGE_OPTIONS = {"a": "resident bucket forwarding in partition high bits",
                 "b": "tiled merge-path pairwise linear merge",
                 "c": "single currentSlots store per query",
                 "d": "stage-end tile barriers with ping-pong Scan"}
MERGE_VARIANTS = {"merge_" + "".join(c for i, c in enumerate(MERGE_OPTIONS) if bits & (1 << i)): 21 + bits
                  for bits in range(1, 16)}
MASKS.update(MERGE_VARIANTS)
MASKS.update(sorted_stream=37, sorted_mrgsort=38)
HW_OPTIONS={"a":"no empty refill / direct single queue", "b":"32-bin resident search index",
            "c":"group-sized map and adaptive/full packets", "d":"retain unconsumed UB heads",
            "e":"256-thread joint VFs; original private fallback"}
HW_SWEEP={"hw_"+"".join(c for i,c in enumerate(HW_OPTIONS) if bits&(1<<i)):38+bits for bits in range(1,32)}
HW_SWEEP.update(hw_abcd_512=70,hw_bcd_512=71,hw_ac_512=72,hw_c_512=73,hw_ac_256=122)
MASKS.update(HW_SWEEP)
UNION_OPTIONS = {"a": "SIMD adjacent flags + register inclusive prefix; SIMT scatter without Scan",
                 "b": "uint16 query-to-union map in UB, int32 GM fallback",
                 "c": "fuse consumed-packet processing with next refill preparation",
                 "d": "initialize invalid queries only; preserve unused miss descriptor tail"}
UNION_SWEEP = {"union_" + ("".join(c for i,c in enumerate(UNION_OPTIONS) if bits & (1<<i)) or "base") + "_" + str(threads):
               74 + group*16 + bits for group,threads in enumerate((1024,512,256)) for bits in range(16)}
MASKS.update(UNION_SWEEP)
COLD_SWEEP={"union_cold_512":123,"hw_ac_cold_512":124}
MASKS.update(COLD_SWEEP)
MASKS["integrated_512"]=125
MASKS["integrated_bounds_512"]=126
INTEGRATED_NAME="joint_integrated_512"
BOUNDS_NAME="joint_integrated_bounds_512"
REFINE_OPTIONS={"a":"strict singleton direct unique/UID in Phase0; skip packing and Phase3",
                "b":"ballot/popcount boolean Scan; packed hit/evict Scan unchanged",
                "c":"warm UB resident snapshot in partition's earlier lifetime; no extra UB",
                "d":"complete SIMD Phase3 using Unsqueeze + Scatter; UB only"}
REFINE_SWEEP={"refine_"+("".join(c for i,c in enumerate(REFINE_OPTIONS) if bits&(1<<i)) or "base")+"_512":127+bits for bits in range(16)}
MASKS.update(REFINE_SWEEP)
REFINE_NAMES=tuple("joint_"+n for n in REFINE_SWEEP)
REFINE_REFS=("abcdf_pages",BOUNDS_NAME,"joint_refine_base_512")
EXEC_OPTIONS={"a":"full-block deinterleaving LoadAlign; masked Gather tail",
              "b":"hit/evict packed ballot Scan; old cross-warp/barriers retained",
              "c":"single actual group Phase0+5 in one SIMT VF",
              "d":"packet index carry cursor when chunk>=threads; old small-chunk path"}
EXEC_SWEEP={"exec_"+("".join(c for i,c in enumerate(EXEC_OPTIONS) if bits&(1<<i)) or "base")+"_512":143+bits for bits in range(16)}
MASKS.update(EXEC_SWEEP)
EXEC_NAMES=tuple("joint_"+n for n in EXEC_SWEEP)
EXEC_REFS=("abcdf_pages",BOUNDS_NAME,"joint_refine_abd_512","joint_refine_bd_512","joint_exec_base_512")
HOT_OPTIONS={"a":"4KiB UB negative filter before exact resident search",
             "b":"SIMD stable partition and miss assignment, combined in one SIMD VF",
             "c":"production UB merge packet expansion within existing112KiB limit",
             "d":"reuse invalidation tags in partition; no second resident read",
             "e":"2048/4096 stage specialization inside existing VF; generic fallback"}
HOT_SWEEP={"hot_"+("".join(c for i,c in enumerate(HOT_OPTIONS) if bits&(1<<i)) or "base")+"_512":159+bits for bits in range(32)}
HOT_SWEEP.update(hot_part_512=191,hot_miss_512=192)
MASKS.update(HOT_SWEEP)
MASKS["hot_bd8_512"]=193
MTP8_NAME="joint_hot_bd8_512"
MASKS.update(mtp7_parallel_512=194,mtp7_rank_512=195)
MTP7_FIX_NAMES=("joint_mtp7_parallel_512","joint_mtp7_rank_512")
MTP7_PACK_MASKS={"mtp7_reuse_parallel_512":196,"mtp7_reuse_rank_512":197,"mtp7_tile_512":198,"mtp7_reuse_tile_512":199}
MASKS.update(MTP7_PACK_MASKS)
MTP7_PACK_NAMES=tuple('joint_'+n for n in MTP7_PACK_MASKS)
MTP7_PREFIX_MASKS = {"mtp7_prefix_a_512": 200, "mtp7_prefix_b_512": 201, "mtp7_prefix_ab_512": 202}
MASKS.update(MTP7_PREFIX_MASKS)
MTP7_PREFIX_NAMES = tuple("joint_" + name for name in MTP7_PREFIX_MASKS)
def mtp7_config(mask):
    if not 193<=mask<=202:return None
    return {'opt':0 if mask==193 else 1 if mask in (194,196,200,201,202) else 2,
            'reuse':mask in (196,197,199,200,201,202),'tile':mask in (198,199)}
def supports_eight_rows(name):return name==MTP8_NAME or name in (*MTP7_FIX_NAMES,*MTP7_PACK_NAMES,*MTP7_PREFIX_NAMES)
HOT_NAMES=tuple("joint_"+n for n in HOT_SWEEP)
HOT_REFS=("abcdf_pages",BOUNDS_NAME,"joint_refine_abd_512","joint_refine_bd_512","joint_hot_base_512")
def hot_config(name):
    value=VARIANTS.get(name,0)
    if not 159<=value<=192:return None
    bits=value-159 if value<=190 else 0
    return {"bits":bits,"filter":bool(bits&1),"partition":bool(bits&2) or value==191,
            "miss":bool(bits&2) or value==192,"packet":bool(bits&4),"reuse":bool(bits&8),"fixed":bool(bits&16)}
INTEGRATED_NAMES=(INTEGRATED_NAME,BOUNDS_NAME,*REFINE_NAMES,*EXEC_NAMES,*HOT_NAMES,MTP8_NAME,*MTP7_FIX_NAMES,*MTP7_PACK_NAMES,*MTP7_PREFIX_NAMES)
INTEGRATED_COMPARE_NAMES=("abcdf_pages","joint_union_cold_512","joint_hw_ac_cold_512",INTEGRATED_NAME,BOUNDS_NAME)
def supports_single_sorted(name):
    return name.startswith("joint_union_") or name in INTEGRATED_NAMES

COLD_NAMES=tuple("joint_"+n for n in COLD_SWEEP)
COLD_COMPARE_NAMES=("abcdf_pages","joint_hw_ac_512","joint_hw_ac_256","joint_union_base_512",*COLD_NAMES)
UNION_NAMES=tuple("joint_"+n for n in UNION_SWEEP)
def union_config(name):
    mask=VARIANTS.get(name,0)
    if not 74<=mask<=121: return None
    return {"bits":(mask-74)%16,"threads":(1024,512,256)[(mask-74)//16],"single_sorted":True,
            "descriptor_tail":"preserve" if (mask-74)&8 else "minus_one"}

HW_NAMES=tuple("joint_"+n for n in HW_SWEEP)
SORTED_NAMES=("joint_sorted_stream", "joint_sorted_mrgsort", *HW_NAMES, *UNION_NAMES, *COLD_NAMES, *INTEGRATED_NAMES)
MERGE_NAMES = ("joint_merge", *("joint_" + n for n in MERGE_VARIANTS), *SORTED_NAMES)
TURBO_MASKS = tuple(range(17, 203))
VARIANTS = {"private_a": 1, "private_turbo": 17} | {"joint_" + letters: mask for letters, mask in MASKS.items()}
DEFAULT_IMPLEMENTATIONS = ("abcdf", "joint_abc", "joint_bitmap", "joint_turbo")


def algorithm_mask(mask):
    """Public ID122 reuses exact old ACE (A+C+256) compute configuration59."""
    return 72 if 127<=mask<=202 else {122:59,123:90,124:72,125:72,126:72}.get(mask,mask)


def is_joint(name):
    return name == "joint" or name.startswith("joint_")


def description(name):
    mask = VARIANTS.get(name, 0)
    if 200 <= mask <= 202:
        return description("joint_mtp7_reuse_parallel_512") | {
            "mask": mask, "parent": "joint_mtp7_reuse_parallel_512",
            "safe_prefix": mask in (200, 202), "compact_map": mask in (201, 202),
            "max_packet_records_per_row": 816 if mask in (201, 202) else 612,
            "scope": "eight-row path only;512threads;181024byte production dynamic UB",
            "packet_contract": "safe inclusive source prefixes" if mask in (200,202) else "hardware exhausted prefixes",
            "map_dtype": "uint16 for eight-row supported groups" if mask in (201,202) else "int32",
            "map_contract": "uint16 UID,0xffff invalid;B16 even-lane Scatter" if mask in (201,202) else "int32 UID"}
    if 196<=mask<=199:
        config=mtp7_config(mask)
        parent='joint_mtp7_parallel_512' if mask==196 else 'joint_mtp7_rank_512'
        packet=612 if config['reuse'] else 512
        return description(parent) | {'mask':mask,'parent':parent,'optimization_config':config,
            'dynamic_ub_change':False,'max_packet_records_per_row':packet,
            'packet_contract':f"eight-row packet: up to{packet} records/source; {'40' if config['reuse'] else '48'}*chunk int32 workspace",
            'tile_outputs_per_thread':8 if config['tile'] else None}
    if mask in (194,195):
        return description(MTP8_NAME) | {"mask":mask,"parent":MTP8_NAME,"optimization":"parallel source histogram" if mask==194 else "parallel rank merge + fused source histogram","dynamic_ub_change":False}
    if mask==193:
        return description("joint_hot_bd_512") | {"mask":193,"max_group_rows":8,"control_words":128,"UB_limit_bytes":192*1024,"merge":"4+4 then2; only final prefix advances sources","parent":"joint_hot_bd_512","packet_contract":"up to8 sources, three UB packet regions; per-source512 records","GM_contract":"UB-only, reject shapes exceeding192KiB"}
    if 159<=mask<=192:
        config=hot_config(name)
        return description("joint_refine_abd_512") | {"mask":mask,"parent":"joint_refine_abd_512",
            "structure":"ABD512 hot pipeline factorial","hot_config":config,
            "new_optimization_factors":[v for i,v in enumerate(HOT_OPTIONS.values()) if config['bits']&(1<<i)],
            "partial_SIMD":"partition only" if mask==191 else "miss only" if mask==192 else None,
            "filter_contract":"negative-only, UB and capacity>=1024; exact search on positives",
            "packet_contract":"only production2048/4096 UB shape; original112KiB row limit",
            "GM_contract":"original SIMT compaction; negative filter bypass"}
    if 143<=mask<=158:
        bits=mask-143
        return description("joint_refine_abd_512") | {"mask":mask,"exec_bits":bits,"parent":"joint_refine_abd_512",
            "structure":"ABD512 execution-factor factorial",
            "new_optimization_factors":[v for i,v in enumerate(EXEC_OPTIONS.values()) if bits&(1<<i)],
            "load_tail":"old masked Gather for <64 records",
            "fused_single":"Phase0 and Phase5 share one VF; original block barriers retained"}
    if 127<=mask<=142:
        bits=mask-127
        return description(BOUNDS_NAME) | {"mask":mask,"refine_bits":bits,"parent":BOUNDS_NAME,
            "structure":"bounds512 + A/B/C/D refinement factorial",
            "new_optimization_factors":[v for i,v in enumerate(REFINE_OPTIONS.values()) if bits&(1<<i)],
            "D_GM_fallback":"original SIMT Phase3; B remains active if selected",
            "B_D_overlap":"D replaces Phase3 on UB; B still applies to Phase5 miss Scan",
            "A_D_overlap":"A bypasses Phase3 for eligible singleton; D handles remaining groups"}
    if mask==126:
        return description(INTEGRATED_NAME) | {"mask":126,"parent":INTEGRATED_NAME,
            "new_optimization_factors":["Phase0 sorted TopK endpoints bypass both lower_bounds when all valid"],
            "phase0_bounds":True,"bounds_fallback":"unchanged lower_bound(0) and lower_bound(clipped row upper)"}
    if mask==125:
        return {"mask":125,"algorithm_mask":72,"threads":512,"private_fallback_threads":1024,
                "structure":"one sorted controller; singleton private + multi-row joint",
                "single_parent":"joint_union_cold_512","multi_parent":"joint_hw_ac_cold_512",
                "single_sorted":True,"cold_finish":True,"descriptor_tail":"minus_one",
                "control_words":64,"map_dtype":"int32","hardware_bits":5,
                "new_optimization_factors":[],"baseline":"abcdf_pages"}
    if mask in (123,124):
        parent="joint_union_base_512" if mask==123 else "joint_hw_ac_512"
        return description(parent) | {"mask":mask,"algorithm_mask":algorithm_mask(mask),
            "cold_finish":True,"parent":parent,"threads":512,
            "reset_only":"request identity or mode mismatch; all-miss is NOT reset",
            "descriptor_tail":"minus_one","private_fallback_threads":1024,
            "single_sorted":mask==123,"mtp0_cold_finish":mask==123}
    if mask == 122:
        return {"mask":122,"algorithm_mask":59,"hardware_bits":21,"threads":256,
                "structure":"explicit AC256 launch; existing ACE compute configuration",
                "merge_engine":"hardware MrgSort","optimizations":[HW_OPTIONS[k] for k in ("a","c","e")],
                "equivalent_configuration_of":"joint_hw_ace","baseline":"abcdf_pages",
                "hardware_reference":"joint_hw_ac_512","private_fallback_threads":1024,
                "single_sorted":False,"descriptor_tail":"minus_one"}
    if mask >= 74:
        c=union_config(name)
        return {"mask":mask,**c,"structure":"AC hardware foundation + new ABCD factorial",
                "merge_engine":"hardware MrgSort", "optimizations":[v for i,v in enumerate(UNION_OPTIONS.values()) if c["bits"]&(1<<i)],
                "baseline":"abcdf_pages", "hardware_reference":"joint_hw_ac_512",
                "single_contract":"strictly increasing input uses sorted private protocol; duplicate rows fall back",
                "private_fallback_threads":1024}
    if mask >= 39:
        bits=mask-38 if mask<=69 else {70:15,71:14,72:5,73:4}[mask]
        return {"mask":mask,"hardware_bits":bits,"threads":512 if mask>=70 else 256 if bits&16 else 1024,
                "structure":"hardware five-factor experiment", "merge_engine":"hardware MrgSort",
                "optimizations":[v for i,v in enumerate(HW_OPTIONS.values()) if bits&(1<<i)],
                "baseline":"abcdf_pages","hardware_reference":"joint_sorted_mrgsort"}
    if mask >= 37:
        return {"mask": mask, "structure": "hash-free sorted query union + resident binary lookup",
                "merge_engine": "hardware MrgSort exhaustion" if mask==38 else "SIMT packet merge",
                "baseline": "abcdf_pages", "source": "sorted inputs; no TopK precomputed hit metadata"}
    if mask >= 22:
        bits = mask - 21
        return {"mask": mask, "merge_bits": bits, "structure": "Merge ABCD factorial",
                "optimizations": [v for i, v in enumerate(MERGE_OPTIONS.values()) if bits & (1 << i)],
                "baseline": "abcdf_pages", "merge_reference": "joint_merge"}
    if mask == 21:
        return {"mask": 21, "structure": "resident-first + sorted miss rank-scatter merge",
                "optimizations": ["sorted TopK contract", "resident LRU-last owner", "miss-only union",
                                  "dynamic 32-bit token/query packing", "capacity check before commit"],
                "baseline": "abcdf_pages"}
    structure = "bitmap+unique+direct_map" if mask == 16 else ("hash+compact_unique+fast_source+finish_tile" if mask in TURBO_MASKS else "hash")
    opts = ([
        "single-VF launch structure + compile-time isolated private VF",
        "fast group discovery with register-cached tokenToReq and early-break bound",
        "FastGroupSource single unsigned token bound + bit-shift resident page division",
        "dedicated UB queryBuckets array + active-bucket tracking in partition high-16-bits",
        "cold/mode-reset zero-scan Joint fast path",
        "single-barrier ping-pong TurboExclusiveScan + private UB token forwarding",
    ] if mask in TURBO_MASKS else [value for bit, value in enumerate(OPTIONS.values()) if mask & (1 << bit)])
    if mask in (18, 20):
        opts.append("A: frozen-table lookup phase, then CAS-only insertion of absent queries")
    if mask in (19, 20):
        opts.append("B: one union compaction Scan per query row, with scratch new-entry flags")
    return {"mask": mask, "structure": structure, "optimizations": opts,
            "baseline": "joint" if is_joint(name) else "private"}
