"""四因素全组合的固定候选选择、条件效应及数据索引。性能取自实际 CSV。"""
import math
from joint_variants import STAGE_EXEC_MASKS, STAGE_EXEC_NAMES, STAGE_EXEC_REFS

FROZEN = STAGE_EXEC_REFS[1]
CONTROL = STAGE_EXEC_REFS[2]
CAPACITIES = {0: 4096, 1: 4096, 3: 8192, 7: 16384}


def hot_runs(report):
    if report.get('status') != 'complete':
        raise ValueError('capture not complete')
    runs = [r for r in report['runs'] if r['workload']['scenario'] == 'full_cache_75pct']
    keys = [(r['workload']['mtp'], r['workload']['capacity']) for r in runs]
    if len(keys) != 4 or set(keys) != set(CAPACITIES.items()):
        raise ValueError('selection requires all four distinct hot workloads')
    return runs


def geomean(values):
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError('invalid duration ratio')
    return math.exp(sum(math.log(v) for v in values) / len(values))


def select(report):
    runs = hot_runs(report)
    scores = []
    for name in STAGE_EXEC_NAMES[1:]:
        ratios = [r['stats'][name]['p50_us'] / r['stats'][FROZEN]['p50_us'] for r in runs]
        scores.append((geomean(ratios), name))
    scores.sort()
    ranking = [{'name': name, 'hot_geomean_over_frozen': value} for value, name in scores]
    return [*STAGE_EXEC_REFS, *(name for _, name in scores[:2])], ranking


def factor_text(report):
    """完整初筛的条件对照；不把 kernel 净差值称为某阶段独立耗时。"""
    runs = hot_runs(report)
    names = {mask - 228: 'joint_' + name for name, mask in STAGE_EXEC_MASKS.items()}
    lines = ['# 四因素条件对照（仅初筛样本）', '',
             '比值为启用该因素后的P50/相同其余因素下的P50；四个热工况取几何平均。',
             '这是整个kernel的条件效应，包含交互与编译变化，不能相加或换算阶段耗时。', '',
             '| 因素 | 已启用的其他因素 | 条件耗时比 |', '|---|---|---:|']
    for bit, letter in enumerate('ABCD'):
        for before in range(16):
            if before & (1 << bit):
                continue
            after = before | (1 << bit)
            ratio = geomean([r['stats'][names[after]]['p50_us'] / r['stats'][names[before]]['p50_us'] for r in runs])
            other = ''.join(c for i, c in enumerate('ABCD') if before & (1 << i)) or 'control'
            lines.append(f'| {letter} | {other} | {ratio:.4f} |')
    return '\n'.join(lines) + '\n'


def report_text(report):
    lines = ['# Stage512 执行优化全组合对照', '',
             '本轮父版为冻结 joint_layout_stage_abcd_512。新A/B/C/D与上一轮字母含义不同。',
             'A=连续加载/写回；B=单hit前缀推导evict排名；C=失效清理合入LRU写回；D=4K用512目录、8K/16K仍1024。',
             '固定32请求（实际以workload记录为准）；MTP0/1/3/7的capacity为4K/4K/8K/16K。512线程；归并及UB预算保持父版。',
             '仅设备目标Plan kernel Duration，包含归并及GM映射暂存，不含恢复、校验、Transfer/SFA。', '',
             '| 工况/MTP/capacity | 实现 | P50/P95 μs | 样本数 | 实现/ABCDF | 实现/冻结stage | 实现/control | 联合/回退 | descriptor |',
             '|---|---|---|---:|---:|---:|---:|---|---:|']
    for run in report['runs']:
        w, stats = run['workload'], run['stats']
        for name, s in stats.items():
            c = s['correctness']
            ratios = [s['p50_us'] / stats[n]['p50_us'] for n in STAGE_EXEC_REFS]
            lines.append(f"| {w['scenario']}/{w['mtp']}/{w['capacity']} | {name} | {s['p50_us']:.3f}/{s['p95_us']:.3f} | {s['count']} | {ratios[0]:.3f} | {ratios[1]:.3f} | {ratios[2]:.3f} | {c.get('joint_groups', 0)}/{c.get('overflow_groups', 0)} | {c['total_transfer_descriptors']} |")
    lines += ['', '所有比值为左侧实现/参考，小于1更快。初筛与独立复测不混合；选择一个固定组合覆盖四个热工况。',
              'control复制父版收尾并关闭四因素；用来观察代码组织/编译变化，不能用较慢control替代冻结参考宣称收益。',
              'C依赖原有LRU为物理槽排列的契约，保留重复resident token的LRU-last owner和全部命中保护。',
              'D直接作用于4K；8K/16K的差异不能算作缩小目录的直接算法收益。四因素不改变冷分支，冷耗时变化需另看编译与采样。',
              '输入共同升序，保留重复token、因果可见性、stable prefix及完整miss尾部约定；逐样本检查状态、输入和保护区。',
              '固定地址预热；不同MTP同时改变容量和候选池。16K ABCDF仍保留原112KiB门槛、使用GM scratch。',
              '原始CSV、实际顺序/样本、源码/库/设备身份与墙钟统计见 capture_metadata.json/data_index.json。未测Transfer/SFA。']
    return '\n'.join(lines) + '\n'
