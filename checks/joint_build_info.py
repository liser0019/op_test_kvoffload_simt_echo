#!/usr/bin/env python3
"""构建/运行联合Plan身份校验；支持CPU源码检查；不代表NPU验证。"""
import argparse
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

def compute_region(text):
    start = text.index("namespace {")
    end = text.index('extern "C" __global__') if "// Direct launch adaptation only:" not in text else text.index("// Direct launch adaptation only:")
    return text[start:end].strip()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Verified workaround no longer matches preserved source")
    return text.replace(old, new, 1)


def expected_fixed_region(original):
    text = compute_region(original)
    text = replace_once(text,
        "  int64_t maxRequests;\n  int64_t maxNumBlocks;\n  int64_t hostNumBlocks;\n  int32_t blockSize;",
        "  int64_t maxRequests;\n  int32_t blockSize;")
    text = replace_once(text,
        "__simt_callee__ __aicore__ inline bool IsSourceTokenValid(int64_t row, int32_t token, __gm__ int32_t* visibleSeqLens,\n"
        "                                                          __gm__ int32_t* tokenToReq, __gm__ int32_t* blockTable,\n"
        "                                                          int64_t maxToken, int64_t maxRequests, int64_t maxNumBlocks,\n"
        "                                                          int64_t hostNumBlocks, int32_t blockSize) {\n"
        "  if (token < 0 || token >= maxToken || token >= visibleSeqLens[row]) {\n"
        "    return false;\n"
        "  }\n"
        "  const int32_t request = tokenToReq[row];\n"
        "  if (request < 0 || request >= maxRequests || blockSize <= 0) {\n"
        "    return false;\n"
        "  }\n"
        "  const int64_t blockId = token / blockSize;\n"
        "  if (blockId < 0 || blockId >= maxNumBlocks) {\n"
        "    return false;\n"
        "  }\n"
        "  const int64_t tableOffset = static_cast<int64_t>(request) * maxNumBlocks + blockId;\n"
        "  const int32_t physicalBlock = blockTable[tableOffset];\n"
        "  return physicalBlock >= 0 && physicalBlock < hostNumBlocks;\n"
        "}",
        "__simt_callee__ __aicore__ inline bool IsSourceTokenValid(int64_t row, int32_t token, __gm__ int32_t* visibleSeqLens,\n"
        "                                                          __gm__ int32_t* tokenToReq, int64_t maxToken,\n"
        "                                                          int64_t maxRequests) {\n"
        "  if (token < 0 || token >= maxToken || token >= visibleSeqLens[row]) {\n"
        "    return false;\n"
        "  }\n"
        "  const int32_t request = tokenToReq[row];\n"
        "  return request >= 0 && request < maxRequests;\n"
        "}")
    text = replace_once(text,
        "    __gm__ int32_t* visibleSeqLens, __gm__ int32_t* tokenToReq, __gm__ int32_t* blockTable,\n"
        "    __gm__ int32_t* slotToToken, __gm__ int32_t* lruSlots, __gm__ int32_t* currentSlots,\n"
        "    __gm__ int32_t* missCount, __gm__ int32_t* missTokens, __gm__ int32_t* missSlots,\n"
        "    uint32_t topk, uint32_t capacity, int32_t maxToken, int64_t maxRequests,\n"
        "    int64_t maxNumBlocks, int64_t hostNumBlocks, int32_t blockSize) {",
        "    __gm__ int32_t* visibleSeqLens, __gm__ int32_t* tokenToReq,\n"
        "    __gm__ int32_t* slotToToken, __gm__ int32_t* lruSlots, __gm__ int32_t* currentSlots,\n"
        "    __gm__ int32_t* missCount, __gm__ int32_t* missTokens, __gm__ int32_t* missSlots,\n"
        "    uint32_t topk, uint32_t capacity, int32_t maxToken, int64_t maxRequests) {")
    text = replace_once(text,
        "        IsSourceTokenValid(row, topkIndices[topkBase + position], visibleSeqLens, tokenToReq,\n"
        "                           blockTable, maxToken, maxRequests, maxNumBlocks, hostNumBlocks, blockSize);",
        "        IsSourceTokenValid(row, topkIndices[topkBase + position], visibleSeqLens, tokenToReq,\n"
        "                           maxToken, maxRequests);")
    text = replace_once(text,
        "    __gm__ int32_t* tokenToReq, __gm__ int32_t* blockTable, __gm__ int32_t* slotToToken, __gm__ int32_t* lruSlots,\n"
        "    __gm__ int32_t* currentSlots, __gm__ int32_t* missCount, __gm__ int32_t* missTokens, __gm__ int32_t* missSlots,\n"
        "    ScratchPtr rowWorkspace, uint32_t hashCapacity, uint32_t topk, uint32_t capacity, int32_t maxToken,\n"
        "    int64_t maxRequests, int64_t maxNumBlocks, int64_t hostNumBlocks, int32_t blockSize) {",
        "    __gm__ int32_t* tokenToReq, __gm__ int32_t* slotToToken, __gm__ int32_t* lruSlots,\n"
        "    __gm__ int32_t* currentSlots, __gm__ int32_t* missCount, __gm__ int32_t* missTokens, __gm__ int32_t* missSlots,\n"
        "    ScratchPtr rowWorkspace, uint32_t hashCapacity, uint32_t topk, uint32_t capacity, int32_t maxToken,\n"
        "    int64_t maxRequests) {")
    text = replace_once(text,
        "    ProcessColdRow(scanWorkspace, missPositions, row, thread, reqIds, lastReqIds, topkIndices,\n"
        "                   visibleSeqLens, tokenToReq, blockTable, slotToToken, lruSlots, currentSlots,\n"
        "                   missCount, missTokens, missSlots, topk, capacity, maxToken, maxRequests,\n"
        "                   maxNumBlocks, hostNumBlocks, blockSize);",
        "    ProcessColdRow(scanWorkspace, missPositions, row, thread, reqIds, lastReqIds, topkIndices,\n"
        "                   visibleSeqLens, tokenToReq, slotToToken, lruSlots, currentSlots,\n"
        "                   missCount, missTokens, missSlots, topk, capacity, maxToken, maxRequests);")
    text = replace_once(text,
        "    if (IsSourceTokenValid(row, token, visibleSeqLens, tokenToReq, blockTable, maxToken, maxRequests, maxNumBlocks,\n"
        "                           hostNumBlocks, blockSize)) {",
        "    if (IsSourceTokenValid(row, token, visibleSeqLens, tokenToReq, maxToken, maxRequests)) {")
    text = replace_once(text,
        "    __gm__ int32_t* stablePrefixLens, __gm__ int32_t* visibleSeqLens, __gm__ int32_t* tokenToReq,\n"
        "    __gm__ int32_t* blockTable, __gm__ int32_t* slotToToken, __gm__ int32_t* lruSlots, __gm__ int32_t* currentSlots,\n"
        "    __gm__ int32_t* missCount, __gm__ int32_t* missTokens, __gm__ int32_t* missSlots, __gm__ int32_t* compactWorkspace,\n"
        "    uint64_t hashCapacity, uint64_t workspaceRowElements, int64_t rowStart, int64_t rowStride, int64_t numReqs,\n"
        "    int64_t topk, int64_t capacity, int64_t maxToken, int64_t maxRequests, int64_t maxNumBlocks, int64_t hostNumBlocks,\n"
        "    int32_t blockSize, __gm__ int32_t* rowModes, __gm__ int32_t* residentRows, __gm__ int32_t* residentBlockTable,",
        "    __gm__ int32_t* stablePrefixLens, __gm__ int32_t* visibleSeqLens, __gm__ int32_t* tokenToReq,\n"
        "    __gm__ int32_t* slotToToken, __gm__ int32_t* lruSlots, __gm__ int32_t* currentSlots,\n"
        "    __gm__ int32_t* missCount, __gm__ int32_t* missTokens, __gm__ int32_t* missSlots, __gm__ int32_t* compactWorkspace,\n"
        "    uint64_t hashCapacity, uint64_t workspaceRowElements, int64_t rowStart, int64_t rowStride, int64_t numReqs,\n"
        "    int64_t topk, int64_t capacity, int64_t maxToken, int64_t maxRequests,\n"
        "    int32_t blockSize, __gm__ int32_t* rowModes, __gm__ int32_t* residentRows, __gm__ int32_t* residentBlockTable,")
    text = replace_once(text,
        "                           tokenToReq,\n                           blockTable,\n                           slotToToken,",
        "                           tokenToReq,\n                           slotToToken,")
    text = replace_once(text,
        "                           blockSize,\n                           maxRequests,\n                           maxNumBlocks,\n                           hostNumBlocks};",
        "                           blockSize,\n                           maxRequests};")
    text = replace_once(text,
        "ProcessRequestRun(args, row, end, scanWorkspace + PLAN_SCAN_STORAGE_ELEMENTS, scanWorkspace);",
        "ProcessRequestRun(args, row, end, scanWorkspace + PLAN_SCAN_STORAGE_ELEMENTS, scanWorkspace, rowModes);")
    return replace_once(text,
        "ProcessRequestRun(args, row, end, compactWorkspace + static_cast<uint64_t>(row) * scratchElements,\n                          scanWorkspace);",
        "ProcessRequestRun(args, row, end, compactWorkspace + static_cast<uint64_t>(row) * scratchElements,\n                          scanWorkspace, rowModes);")


def expected_fixed_helper(original):
    text = replace_once(original,
        "  __gm__ int32_t *topkIndices, *stablePrefixLens, *visibleSeqLens, *tokenToReq, *blockTable;\n"
        "  __gm__ int32_t *slotToToken, *lruSlots, *currentSlots, *missCount, *missTokens, *missSlots;\n"
        "  __gm__ int32_t *rowModes, *residentRows, *residentBlockTable;\n"
        "  uint32_t topk, capacity, hashCapacity, jointHashCapacity;\n"
        "  int32_t maxToken, blockSize;\n"
        "  int64_t maxRequests, maxNumBlocks, hostNumBlocks;",
        "  __gm__ int32_t *topkIndices, *stablePrefixLens, *visibleSeqLens, *tokenToReq;\n"
        "  __gm__ int32_t *slotToToken, *lruSlots, *currentSlots, *missCount, *missTokens, *missSlots;\n"
        "  __gm__ int32_t *rowModes, *residentRows, *residentBlockTable;\n"
        "  uint32_t topk, capacity, hashCapacity, jointHashCapacity;\n"
        "  int32_t maxToken, blockSize;\n"
        "  int64_t maxRequests;")
    text = replace_once(text,
        "  return IsSourceTokenValid(row, token, a.visibleSeqLens, a.tokenToReq, a.blockTable, a.maxToken, a.maxRequests,\n"
        "                            a.maxNumBlocks, a.hostNumBlocks, a.blockSize);",
        "  return IsSourceTokenValid(row, token, a.visibleSeqLens, a.tokenToReq, a.maxToken, a.maxRequests);")
    text = replace_once(text,
        "  ProcessRuntimeRow(scan, row, a.reqIds, a.lastReqIds, a.topkIndices, a.stablePrefixLens, a.visibleSeqLens,\n"
        "                    a.tokenToReq, a.blockTable, a.slotToToken, a.lruSlots, a.currentSlots, a.missCount, a.missTokens,\n"
        "                    a.missSlots, scratch, a.hashCapacity, a.topk, a.capacity, a.maxToken, a.maxRequests, a.maxNumBlocks,\n"
        "                    a.hostNumBlocks, a.blockSize);",
        "  ProcessRuntimeRow(scan, row, a.reqIds, a.lastReqIds, a.topkIndices, a.stablePrefixLens, a.visibleSeqLens,\n"
        "                    a.tokenToReq, a.slotToToken, a.lruSlots, a.currentSlots, a.missCount, a.missTokens,\n"
        "                    a.missSlots, scratch, a.hashCapacity, a.topk, a.capacity, a.maxToken, a.maxRequests);")
    text = replace_once(text,
        "                                                      __ubuf__ int32_t* scan, uint32_t thread) {",
        "                                                      __ubuf__ int32_t* scan, __gm__ int32_t* modeOutput,\n                                                      uint32_t thread) {")
    text = replace_once(text,
        "      a.rowModes[row] = row == first ? ROW_JOINT : ROW_FOLLOWER;",
        "    }\n    // Device experiments reproduced this mode value in a read-only input.\n"
        "    // Explicit output + thread 1 is an empirical workaround; root cause is\n"
        "    // unconfirmed. This is not a synchronization repair: no barrier is added.\n"
        "    if (thread == 1) {\n      modeOutput[row] = row == first ? ROW_JOINT : ROW_FOLLOWER;")
    text = replace_once(text,
        "                                                         ScratchPtr scratch, __ubuf__ int32_t* scan) {",
        "                                                         ScratchPtr scratch, __ubuf__ int32_t* scan,\n                                                         __gm__ int32_t* modeOutput) {")
    return replace_once(text,
        "PlanJointUnion(a, first, end, keys, slots, partition, scan, thread);",
        "PlanJointUnion(a, first, end, keys, slots, partition, scan, modeOutput, thread);")

def expected_cold_stage(text,single):
    lines=[line for line in text.splitlines(True) if 'const int64_t reqid=a.reqIds[req];const bool reset=' in line]
    if len(lines)!=1:raise ValueError('Cold finish reset insertion anchor changed')
    hook="""    if (reset) {
      // Uniform reset: no thread may publish mode/ID until all have captured it.
      asc_syncthreads();
      FinishColdSorted<SORTED_THREADS,SINGLE>(a,first,end,map,unique,ctx,t);
      return;
    }
""".replace('SINGLE','true' if single else 'false')
    return text.replace(lines[0],lines[0]+hook,1)


def expected_integrated_stage(text):
    text=replace_once(text,"rows<2U||rows>4U","rows<1U||rows>4U")
    text=replace_once(text,"if(ctx[0])for(uint32_t i=t;i<q;i+=SORTED_THREADS)map[i]=-1;","""if(ctx[0])for(uint32_t i=t;i<q;i+=SORTED_THREADS){
      // Independent semantics count duplicate query misses separately. Fall back;
      // never silently share their slot through the single-row unique list.
      if(rows==1U && i && a.topkIndices[qb+i]==a.topkIndices[qb+i-1U])asc_atomic_min(ctx,0);
      map[i]=-1;
    }""")
    text=replace_once(text,"a.rowModes[first]!=ROW_JOINT","a.rowModes[first]!=(rows==1U?ROW_PRIVATE:ROW_JOINT)")
    text=replace_once(text,"FinishColdSorted<SORTED_THREADS,false>","FinishColdSorted<SORTED_THREADS,true>")
    return replace_once(text,"a.rowModes[row]=r==0U?ROW_JOINT:ROW_FOLLOWER;", "a.rowModes[row]=rows==1U?ROW_PRIVATE:(r==0U?ROW_JOINT:ROW_FOLLOWER);")


def expected_phase0_bounds_stage(text):
    return replace_once(text,'        ctx[8+r]=static_cast<int32_t>(SortedLower(a.topkIndices+b,a.topk,0));\n        ctx[12+r]=static_cast<int32_t>(SortedLower(a.topkIndices+b,a.topk,upper));','        // Only Phase0 uses endpoint checks. SortedLower for resident lookup is unchanged.\n        // upper remains row-specific and clipped; equality at upper is NOT visible.\n        if(a.topk>0U && a.topkIndices[b]>=0 && a.topkIndices[b+a.topk-1U]<upper){\n          ctx[8+r]=0;ctx[12+r]=static_cast<int32_t>(a.topk);\n        }else{\n          ctx[8+r]=static_cast<int32_t>(SortedLower(a.topkIndices+b,a.topk,0));\n          ctx[12+r]=static_cast<int32_t>(SortedLower(a.topkIndices+b,a.topk,upper));\n        }')


def identity():
    manifest = json.loads((ROOT / "joint_plan_manifest.json").read_text())
    for path, expected in manifest["file_sha256"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
            raise ValueError("Joint Plan source fingerprint changed: " + path)
    for parent,cold,single in [('union_next_stage.h','cold_union_stage.h',True),('sorted_sweep_stage.h','cold_ac_stage.h',False)]:
        base=ROOT/'csrc/joint_variants'
        if expected_cold_stage((base/parent).read_text(),single)!=(base/cold).read_text():
            raise ValueError('Cold stage changed beyond reset-only hook: '+cold)
    base=ROOT/'csrc/joint_variants'
    if expected_integrated_stage((base/'cold_ac_stage.h').read_text())!=(base/'integrated_stage.h').read_text():
        raise ValueError('Integrated stage differs beyond single-row private semantics')
    if expected_phase0_bounds_stage((base/'integrated_stage.h').read_text())!=(base/'phase0_bounds_stage.h').read_text():
        raise ValueError('Phase0 bounds candidate differs beyond endpoint fast path')
    original = (ROOT / "csrc/joint_plan/upstream/sparse_kv_plan_apt.cpp").read_text()
    direct = (ROOT / "csrc/joint_plan/kernel.cpp").read_text()
    if expected_fixed_region(original) != compute_region(direct):
        raise ValueError("Joint VF differs beyond approved source-page removal and modeOutput forwarding")
    header = "csrc/joint_plan/sparse_kv_joint_plan.h"
    if expected_fixed_helper((ROOT / "csrc/joint_plan/upstream/sparse_kv_joint_plan.h").read_text()) != (ROOT / header).read_text():
        raise ValueError("Joint helper differs beyond source-page removal and thread1 store workaround")
    if (ROOT / "csrc/joint_plan/upstream/sparse_kv_plan_config.h").read_bytes() != (ROOT / "csrc/joint_plan/sparse_kv_plan_config.h").read_bytes():
        raise ValueError("Joint config changed")
    config = (ROOT / "csrc/plan_variants/13_abcdf/variant_config.h").read_text()
    common_config = (ROOT / "csrc/plan_variants/common/sparse_kv_plan_config.h").read_text()
    if "#define PLAN_OPT_MASK 47U" not in config or "PLAN_THREADS = 1024U" not in common_config:
        raise ValueError("Independent baseline must remain ABCDF(mask47), 1024 threads")
    variant = (ROOT / "csrc/joint_variants/kernel.cpp").read_text()
    prefix = variant[:variant.index('#include "options.h"')].replace('../joint_plan/', '')
    prefix=prefix.replace('#ifdef JOINT_REFINE\n#include "simt_api/device_functions.h"\n#endif\n','')
    original_prefix = direct[:direct.index('#include "sparse_kv_joint_plan.h"')]
    if prefix != original_prefix:
        raise ValueError("Experimental kernels changed original ABCDF helper computation")
    helper = (ROOT / "csrc/joint_variants/sparse_kv_joint_plan.h").read_text()
    if "if (thread == 1) {\n      modeOutput[row] = row == first ? ROW_JOINT : ROW_FOLLOWER;" not in helper:
        raise ValueError("Joint variants must preserve empirical thread1 workaround")
    from joint_variants import VARIANTS
    if manifest.get("joint_variant_masks") != VARIANTS:
        raise ValueError("Joint variant catalog mismatch")
    return manifest

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--flags", required=True)
    args = parser.parse_args()
    payload = json.dumps({"source": identity(), "compiler": args.compiler, "flags": args.flags}, sort_keys=True)
    source = '#include <torch/library.h>\n#include <string>\nnamespace { std::string Info() { return R"JOINT('
    source += payload + ')JOINT"; } }\n'
    source += 'TORCH_LIBRARY_FRAGMENT(mtp_joint_plan, m) { m.def("source_info() -> str", Info); }\n'
    if not args.output.exists() or args.output.read_text() != source:
        args.output.write_text(source)

if __name__ == "__main__":
    main()
