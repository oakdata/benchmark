import os
import os.path as osp
import pickle

inf_dir = ''


inf_files = sorted(os.listdir(inf_dir), key=lambda x: int(x.split('_')[0]) )
total_inf = {}
for i,inf_file in enumerate(inf_files):
    res = pickle.load(open(osp.join(inf_dir,inf_file),'rb'))
    total_inf[inf_file] = res

f = open(osp.join(inf_dir,'total_inf.pkl'),'wb')
pickle.dump(total_inf,f)
f.close()