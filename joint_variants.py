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
MTP7_MERGE_MASKS={"mtp7_mid_unique_512":203,"mtp7_fused_count_512":204}
MASKS.update(MTP7_MERGE_MASKS)
MTP7_MERGE_NAMES=tuple("joint_"+n for n in MTP7_MERGE_MASKS)
MTP7_FUSED_TUNE_MASKS={"mtp7_fused_"+letters+"_512":204+bits for bits,letters in
    ((1,"a"),(2,"b"),(3,"ab"),(4,"c"),(5,"ac"),(6,"bc"),(7,"abc"))}
MASKS.update(MTP7_FUSED_TUNE_MASKS)
CAPACITY8K_MASKS={"mtp7_fused_8k_ref_512":212,"mtp7_fused_ab_8k_512":213}
MASKS.update(CAPACITY8K_MASKS)
CAPACITY8K_NAMES=tuple("joint_"+n for n in CAPACITY8K_MASKS)
CAPACITY16_NAMES=("joint_mtp7_stage16_512","joint_mtp7_ub16_512")
MASKS.update({"mtp7_stage16_512":214,"mtp7_ub16_512":215})
LAYOUT_HOT_COMBOS=("a","ab","ac","abc","acd","abcd")
LAYOUT_HOT_MASKS={f"layout_{family}_{combo}_512":216+f*6+i for f,family in enumerate(("stage","ub")) for i,combo in enumerate(LAYOUT_HOT_COMBOS)}
MASKS.update(LAYOUT_HOT_MASKS)
LAYOUT_HOT_NAMES=tuple("joint_"+n for n in LAYOUT_HOT_MASKS)
LAYOUT_HOT_REFS=("abcdf_pages",*CAPACITY16_NAMES)
# 本轮四因素以 stage ABCD 为父版；字母与上一轮独立。
STAGE_EXEC_OPTIONS={"a":"continuous unpack16 flags load and contiguous owner load/store",
                    "b":"one hit prefix; derive evict rank from global LRU position",
                    "c":"invalidate only unallocated slots while committing LRU",
                    "d":"512 directory bins at4K;1024 bins at8K/16K"}
STAGE_EXEC_MASKS={"stage_exec_"+("".join(c for i,c in enumerate(STAGE_EXEC_OPTIONS) if bits&(1<<i)) or "base")+"_512":228+bits for bits in range(16)}
MASKS.update(STAGE_EXEC_MASKS)
STAGE_EXEC_NAMES=tuple("joint_"+n for n in STAGE_EXEC_MASKS)
STAGE_EXEC_REFS=("abcdf_pages","joint_layout_stage_abcd_512",STAGE_EXEC_NAMES[0])
# 本轮三因素：在exec ABCD之上进行精确查询/块前缀/输出流水组合。
RANKPIPE_OPTIONS={"a":"exact UB word bitmap and first UID instead of binary search (max_token<=262144)",
                  "b":"32-position word counts, cross-warp prefix and local popcount ranks",
                  "c":"stage full LRU and descriptor tails then MTE3 copies; cold and hot"}
RANKPIPE_MASKS={"rankpipe_"+("".join(c for i,c in enumerate(RANKPIPE_OPTIONS) if bits&(1<<i)) or "base")+"_512":244+bits for bits in range(8)}
MASKS.update(RANKPIPE_MASKS)
RANKPIPE_NAMES=tuple("joint_"+n for n in RANKPIPE_MASKS)
RANKPIPE_REFS=("abcdf_pages","joint_stage_exec_abcd_512",RANKPIPE_NAMES[0])
# 固定 MTP3/4x2048/8192 的独立候选；252..255 不参与 MTP7 选择器。
MTP3_MASKS={"mtp3_full_512":252,"mtp3_map_512":253,"mtp3_load_512":254,"mtp3_full_load_512":255}
EXTERNAL_MTP3_MASKS={"mtp3_bt_direct_odd_512":264}
MASKS.update(MTP3_MASKS)
MTP3_NAMES=tuple("joint_"+n for n in (*MTP3_MASKS,*EXTERNAL_MTP3_MASKS))
def mtp3_config(mask):
    if 252<=mask<=255:
        return {"full":mask in (252,255),"map":mask==253,"load":mask in (254,255)}
    if mask==264:
        return {"full":True,"map":False,"load":False,"bitmap_team":"direct",
                "bitmap_odd_stride":True,"max_token":486400}
    return None
def rankpipe_bits(mask):return 5 if 252<=mask<=264 else mask-244 if 244<=mask<=251 else 0
def stage_exec_bits(mask): return 15 if 228<=mask<=264 else 0
def layout_hot_bits(mask):
    return 15 if 228<=mask<=264 else sum({"a":1,"b":2,"c":4,"d":8}[x] for x in LAYOUT_HOT_COMBOS[(mask-216)%6]) if 216<=mask<=227 else 0
def layout16_mode(name):
    if name in (*STAGE_EXEC_NAMES,*RANKPIPE_NAMES,*MTP3_NAMES):return 1
    if name in LAYOUT_HOT_NAMES:return 1+(LAYOUT_HOT_MASKS[name[6:]]-216)//6
    return {CAPACITY16_NAMES[0]:1,CAPACITY16_NAMES[1]:2}.get(name,0)
def is_capacity8k(name): return name in CAPACITY8K_NAMES
def fused_tune_bits(mask): return 3 if mask in (213,214,215) or 216<=mask<=251 else mask-204 if 205<=mask<=211 else 0
MTP7_FUSED_TUNE_NAMES=tuple("joint_"+n for n in MTP7_FUSED_TUNE_MASKS)
def mtp7_config(mask):
    if 252<=mask<=264:return {'opt':1,'reuse':False,'tile':False}
    if not 193<=mask<=251:return None
    return {'opt':0 if mask==193 else 1 if mask in (194,196,200,201,202,203,204) or 205<=mask<=251 else 2,
            'reuse':mask in (196,197,199,200,201,202,203,204) or 205<=mask<=251,'tile':mask in (198,199)}
def supports_eight_rows(name):return name==MTP8_NAME or name in (*MTP7_FIX_NAMES,*MTP7_PACK_NAMES,*MTP7_PREFIX_NAMES,*MTP7_MERGE_NAMES,*MTP7_FUSED_TUNE_NAMES,*CAPACITY8K_NAMES,*CAPACITY16_NAMES,*LAYOUT_HOT_NAMES,*STAGE_EXEC_NAMES,*RANKPIPE_NAMES,*MTP3_NAMES)
HOT_NAMES=tuple("joint_"+n for n in HOT_SWEEP)
HOT_REFS=("abcdf_pages",BOUNDS_NAME,"joint_refine_abd_512","joint_refine_bd_512","joint_hot_base_512")
def hot_config(name):
    value=VARIANTS.get(name,0)
    if not 159<=value<=192:return None
    bits=value-159 if value<=190 else 0
    return {"bits":bits,"filter":bool(bits&1),"partition":bool(bits&2) or value==191,
            "miss":bool(bits&2) or value==192,"packet":bool(bits&4),"reuse":bool(bits&8),"fixed":bool(bits&16)}
INTEGRATED_NAMES=(INTEGRATED_NAME,BOUNDS_NAME,*REFINE_NAMES,*EXEC_NAMES,*HOT_NAMES,MTP8_NAME,*MTP7_FIX_NAMES,*MTP7_PACK_NAMES,*MTP7_PREFIX_NAMES,*MTP7_MERGE_NAMES,*MTP7_FUSED_TUNE_NAMES,*CAPACITY8K_NAMES,*CAPACITY16_NAMES,*LAYOUT_HOT_NAMES,*STAGE_EXEC_NAMES,*RANKPIPE_NAMES,*MTP3_NAMES)
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
TURBO_MASKS = tuple(range(17, 256))
VARIANTS = ({"private_a": 1, "private_turbo": 17}
            | {"joint_" + letters: mask for letters, mask in MASKS.items()}
            | {"joint_" + name: mask for name, mask in EXTERNAL_MTP3_MASKS.items()})
DEFAULT_IMPLEMENTATIONS = ("abcdf", "joint_abc", "joint_bitmap", "joint_turbo")


def algorithm_mask(mask):
    """Public ID122 reuses exact old ACE (A+C+256) compute configuration59."""
    return 72 if 127<=mask<=264 else {122:59,123:90,124:72,125:72,126:72}.get(mask,mask)


def is_joint(name):
    return name == "joint" or name.startswith("joint_")


def description(name):
    mask = VARIANTS.get(name, 0)
    if mask==264:
        return {"mask":264,"algorithm_mask":72,"parent":"joint_mtp3_full_512",
                "threads":512,"private_fallback_threads":1024,"layout16":1,
                "structure":"dual-AIV direct bitmap lookup with odd helper stride",
                "max_group_rows":4,"capacity":8192,"topk":2048,
                "max_token":486400,"dynamic_ub_bytes":187936,
                "workspace_elements":"rows*topk/2 + requests*2048",
                "bitmap_team":"direct","bitmap_odd_stride":True,
                "source_commit":"88c06745b0260ef258ce3870945bdb7668d3d483",
                "status":"externally pinned Plan library"}
    if 252<=mask<=255:
        cfg=mtp3_config(mask)
        return {"mask":mask,"parent":"joint_rankpipe_ac_512","threads":512,"private_fallback_threads":1024,
          "structure":"fixed MTP3 4x2048 rankpipe candidate","rankpipe_bits":5,"stage_exec_bits":15,
          "scope":"exact topk2048/capacity8192/maxToken262144/rows4/512-thread gate",
          "mtp3_full":cfg["full"],"mtp3_hot_ub_map":cfg["map"],"mtp3_aligned_pair_load":cfg["load"],
          "merge_packet_bytes_per_side":65536 if cfg["full"] else 49152,
          "merge_workspace_delta_bytes":32768 if cfg["full"] else 0,
          "dynamic_ub_bytes":181024 if cfg["full"] else 148256,
          "map_staging_GM":not cfg["map"],"descriptor_tail":"minus_one",
          "status":"CPU candidate; CANN/NPU verification pending"}
    if 244<=mask<=251:
        bits=rankpipe_bits(mask)
        return description("joint_stage_exec_abcd_512") | {"mask":mask,"parent":"joint_stage_exec_abcd_512",
          "rankpipe_bits":bits,"factors":[v for i,v in enumerate(RANKPIPE_OPTIONS.values()) if bits&(1<<i)],
          "scope":"A/B hot path; C cold/hot full-output producer and kernel-side MTE3 consumer; sameUB budget",
          "bitmap_words":8192 if bits&1 else 0,"bitmap_base_uid_bits":32 if bits&1 else 0,
          "max_dynamic_ub_bytes":197408,"status":"CPU candidate; CANN/NPU pending"}
    if 228<=mask<=243:
        bits=stage_exec_bits(mask)
        return description("joint_layout_stage_abcd_512") | {"mask":mask,"parent":"joint_layout_stage_abcd_512",
            "stage_exec_bits":bits,"factors":[v for i,v in enumerate(STAGE_EXEC_OPTIONS.values()) if bits&(1<<i)],
            "directory_bins":"512 at4096;1024 otherwise" if bits&8 else 1024,
            "scope":"stage hot finish only; merge/cold path/protocol/UB allocation unchanged",
            "status":"CPU candidate; CANN/NPU validation required"}
    if 216<=mask<=227:
        mode=1+(mask-216)//6;bits=layout_hot_bits(mask)
        return description(CAPACITY16_NAMES[mode-1]) | {"mask":mask,"capacity":"4096/8192/16384","scope":"capacity layout and hot classification factors; merge algorithm retained","layout_hot_bits":bits,"factors":LAYOUT_HOT_COMBOS[(mask-216)%6],"right_sized_UB":True,"max_query_positions": "capacity", "hot_compaction":"SIMD packed32 load" if mode==1 and bits&2 else "bitmap word prefix" if bits&2 else "parent SIMT Scan","physical_slot_staging":bool(bits&4),"directory_bins":1024 if bits&8 else 0,"UB_limit_bytes":204*1024,"dynamic_ub_bytes":"runtime capacity-specific; see layout metadata"}
    if mask in (214,215):
        return description("joint_mtp7_fused_ab_512") | {"mask":mask,"layout16":mask-213,"capacity":16384,"map_bits":16,"hot_compaction":"SIMT Scan","dynamic_ub_bytes":197408 if mask==214 else 203552,"map_staging_GM":mask==214,"status":"CPU candidate; NPU pending"}
    if mask in (212,213):
        parent="joint_mtp7_fused_count_512" if mask==212 else "joint_mtp7_fused_ab_512"
        return description(parent) | {"mask":mask,"parent":parent,"capacity8k_adapter":True,
            "UB_limit_bytes":196*1024,"original_UB_limit_bytes":192*1024,
            "budget_exception_shape":{"topk":2048,"capacity":8192},
            "dynamic_ub_bytes_8k":197408,"status":"CPU candidate; CANN/NPU verification required",
            "scope":"parent compute unchanged; bounded UB adapter only, all unsupported shapes reject"}
    if 205<=mask<=211:
        bits=mask-204
        return description("joint_mtp7_fused_count_512") | {
            "mask":mask,"parent":"joint_mtp7_fused_count_512","tune_bits":bits,
            "hardware_counts":bool(bits&1),"aligned_pair_load":bool(bits&2),"lean_batched_cursor":bool(bits&4),
            "scope":"eight-row two-level merge only; A reuses full halves only with8 active rows; B complete64 records only; C seven counts and batched cursor"}
    if mask in (203,204):
        return description("joint_mtp7_reuse_parallel_512") | {
            "mask":mask,"parent":"joint_mtp7_reuse_parallel_512","threads":512,
            "mid_unique":mask==203,"fused_count":mask==204,
            "scope":"eight-row two-level merge only; int32 map; same181024B production UB",
            "mapping":"query->MID->union, speculative MID entries resolved/reset before commit" if mask==203 else "original query->union; SIMD cumulative row counters",
            "status":"CPU candidate; not locally CANN/NPU verified"}
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
