"""Byte intervals for every live UB phase; control+Scan are accounted separately."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from joint_plan_cases import layout
from joint_variants import LAYOUT_HOT_MASKS,layout16_mode,layout_hot_bits

def verify():
 records=[]
 for name,mask in LAYOUT_HOT_MASKS.items():
  mode=layout16_mode('joint_'+name);bits=layout_hot_bits(mask)
  for c in (4096,8192,16384):
   d=layout(2048,c,256,True,262144,sorted_union=True,mtp8=True,layout16=mode,layout_hot_bits=bits)
   data_bytes=(d['scratch_elements']-128)*4
   regions={'map':(0,2*c),'flags':(0,2*c),'unique':(2*c,6*c),'owner':(6*c,10*c),'directory':(6*c,6*c+1025*4),'slot_uid':(10*c,12*c),'partition':(10*c,12*c),'packet':(6*c,6*c+24576*4),'bitmaps':(12*c,12*c+3*c//8),'prefix':(12*c+3*c//8,12*c+c//2)}
   phases={'merge':['map','unique','packet'],'classify':['unique','owner']+(['flags'] if mode==1 else ['map','bitmaps'])+(['slot_uid'] if bits&4 else []),'partition':['unique','owner','partition']+(['flags'] if mode==1 else ['map','bitmaps']+(['prefix'] if bits&2 else [])), 'commit':['unique','owner','partition']+(['flags'] if mode==1 and not bits&2 else ['map'] if mode==2 else [])}
   if bits&4:phases['physical_lookup']=['unique','slot_uid']+(['map'] if mode==2 else [])+(['directory'] if bits&8 else [])
   peak=0
   for phase,names in phases.items():
    live=[(n,*regions[n]) for n in names]
    for n,lo,hi in live:assert 0<=lo<hi<=data_bytes,(name,c,phase,n,hi,data_bytes)
    for i,(n,lo,hi) in enumerate(live):
     for m,a,b in live[i+1:]:assert hi<=a or b<=lo,(name,c,phase,n,m)
    peak=max(peak,max(hi for _,_,hi in live))
   records.append({'name':name,'capacity':c,'dynamic_ub_bytes':d['dynamic_ub_bytes'],'maximum_live_end_in_data_bytes':peak,'regions':regions,'phases':phases})
 return records
if __name__=='__main__':
 records=verify();print(json.dumps({'status':'passed','count':len(records),'scope':'source byte intervals; excludes compiler spill and hardware cache effects','records':records},indent=2))
