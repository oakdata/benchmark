import os
import os.path as osp
import json
import pickle

inf_dir = ''


inf_files = sorted(os.listdir(inf_dir), key=lambda x: int(x.split('_')[0]) )
total_inf = {}
for i,inf_file in enumerate(inf_files):
    res = json.load(open(osp.join(inf_dir,inf_file),'r'))
    total_inf[inf_file] = res

f = open(osp.join(inf_dir,'total_inf.json'),'w')
res = json.dumps(res)
f.write(res)