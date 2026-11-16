"""固定候选选择与报告；筛选以各自父版为准，不拼接逐工况赢家。"""
import math
from joint_variants import LAYOUT_HOT_NAMES,LAYOUT_HOT_REFS,layout16_mode,description

def select(report):
 if report.get('status')!='complete':raise ValueError('screen not complete')
 ranked={};selected=list(LAYOUT_HOT_REFS)
 for mode in (1,2):
  parent=LAYOUT_HOT_REFS[mode];scores=[]
  for name in LAYOUT_HOT_NAMES:
   if layout16_mode(name)!=mode:continue
   ratios=[]
   for run in report['runs']:
    if run['workload']['scenario']=='full_cache_75pct':
     ratios.append(run['stats'][name]['p50_us']/run['stats'][parent]['p50_us'])
   if not ratios:raise ValueError('need hot workloads for selection')
   score=math.exp(sum(math.log(v) for v in ratios)/len(ratios));scores.append((score,name))
  scores.sort();ranked[parent]=[{'name':n,'hot_geomean_over_parent':v} for v,n in scores]
  selected.append(scores[0][1])
 return selected,ranked

def report_text(report):
 lines=['# 动态容量热路径布局对照','',
 '固定请求数，MTP0/1=capacity4096，MTP3=8192，MTP7=16384。仅设备目标Plan kernel Duration；包括归并和GM映射暂存。',
 'A=容量布局；B=stage SIMD压缩/全UB位图字排名；C=物理槽顺序分类；D=1024区间目录（依赖C）。当前所有候选都包含A。',
 '同工况、同输入与初态，逐样本完整校验。比值为当前实现P50/参考P50，小于1更快。',
 '', '| 工况/MTP/capacity | 实现 | P50/P95 μs | 样本数 | 实现/ABCDF | 实现/冻结父版 | 共享/回退 | descriptor |',
 '|---|---|---|---:|---:|---:|---|---:|']
 for run in report['runs']:
  w=run['workload'];stats=run['stats']
  for name,s in stats.items():
   mode=layout16_mode(name);parent=LAYOUT_HOT_REFS[mode] if mode else 'abcdf_pages';c=s['correctness']
   lines.append(f"| {w['scenario']}/{w['mtp']}/{w['capacity']} | {name} | {s['p50_us']:.3f}/{s['p95_us']:.3f} | {s['count']} | {s['p50_us']/stats['abcdf_pages']['p50_us']:.3f} | {s['p50_us']/stats[parent]['p50_us']:.3f} | {c.get('joint_groups',0)}/{c.get('overflow_groups',0)} | {c['total_transfer_descriptors']} |")
 lines+=['','两种布局保留自己的冻结参考；初筛与复测数据分别报告，不混合样本。不同MTP同时改变容量和候选池，不能称纯MTP开关开销。',
 '每组独立判断P50/P95。16K ABCDF仍保留原112KiB工作区门槛；其GM路径应按原配置解读。',
 '候选支持query位置数<=capacity；超出先拒绝联合，再走既有private协议。冻结版本支持条件不变。',
 '源码/库/设备身份、动态UB、输入指纹、逐样本时长和原始CSV索引见capture_metadata.json、data_index.json及采集子目录。未测Transfer/SFA。']
 return '\n'.join(lines)+'\n'
