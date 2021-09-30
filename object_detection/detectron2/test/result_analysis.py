import os
import os.path as osp
import pickle
import json
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import copy
import numpy as np

from eval import voc_eval

sourcedir = '/grogu/user/jianrenw/data/OAK_TEST/Raw' 
annodir = '/grogu/user/jianrenw/data/OAK_TEST/Label'
config_fp = '/grogu/user/jianrenw/baseline/release/faster_rcnn_R_50_C4.yaml'
cat_fp = '/grogu/user/jianrenw/baseline/release/mapping.json'
saved_dir = '/grogu/user/jianrenw/baseline/release/baseline_res'
inter_p = ''


# data format
# {
#   i: (evaluation step)
#       {
#           clsname:
#               [{'image_id' : '1'
#                  'score'   : 0.5
#                  'bbox'    : {
#                            "x1": 41.0,
#                            "x2": 666.0,
#                            "y1": 374.5,
#                            "y2": 648.0
#                        } ]
#       }}

class wunderlust():
    # def __init__():
    # def process():

    # self.existence -> dictionary {category:[1/0 or number of instance](training set)} -> ground truth
    # self.map -> dictionary {category:[map](test set, evaluate times)} -> evaluation result
    # self.baseline -> dictionary {category:map (original model, prior knowledge)}
    # self.R -> please refer to "Gradient Episodic Memory for Continual Learning"
    # self.B -> please refer to "Gradient Episodic Memory for Continual Learning"
    def find_nearest_existence(self, category, current_frame):
        # the current frame represent the exact one frame after the training step
        i = 1
        find = False
        while not find:
            left = max([0, current_frame - i])
            if self.existence[category][left] > 0:
                find = True
            else:
                i += 1
            if left == 0 and not find:
                i = -1
                break
        return i

    def cap(self):
        self.cap = {}
        for category in self.categories:
            self.cap[category] = np.mean(self.map[category])
        return self.cap

    def forgetfulness(self):
        self.forgetfulness = {}
        for category in self.categories:
            ap_time_gap = {}
            all_dis = []
            all_ap = []
            for i, time_stamp in enumerate(self.image_id):
                dis = self.find_nearest_existence(category, time_stamp * 16)
                if dis == -1:
                    continue
                else:
                    if dis in ap_time_gap:
                        ap_time_gap[dis].append(self.map[category][i])
                    else:
                        ap_time_gap[dis] = [self.map[category][i]]
            for dis, ap in ap_time_gap.items():
                all_dis.append(dis)
                all_ap.append(np.mean(ap))
            sorted_dis, sorted_ap = zip(*sorted(zip(all_dis, all_ap)))
            sorted_dis = np.array(sorted_dis)
            sorted_forgetfulness = sorted_ap[0] - np.array(sorted_ap)
            self.forgetfulness[category] = 1 / np.sum(sorted_dis) * np.sum(
                sorted_dis * sorted_forgetfulness)


    def BWT(self):
        return np.mean(self.R[-1, :-1] - np.diag(self.R)[:-1])

    def FWT(self):
        return np.mean(np.diag(self.R, k=1) - self.B[1:])

def returnmap():
    f = open(cat_fp, 'r')
    content = f.read()
    categories = json.loads(content)
    return categories

def get_trainmap_fix():
    mapping = {i:i for i in range(0,len(limitset.keys()))}
    return mapping

steps_name = sorted(os.listdir(sourcedir))
all_testset = sorted(os.listdir(annodir))
limitset = returnmap()
model_path = osp.join(saved_dir,'models')
inf_path = osp.join(saved_dir,'inf/total_inf.pkl')
ini_inf_path = osp.join(saved_dir,'ini_inf.pkl')

inter = json.load(open('/grogu/user/jianrenw/baseline/code/inter.json','r'))
method = 'idk'
total_inf = pickle.load(open(inf_path,'rb'))
total_ini = pickle.load(open(ini_inf_path,'rb'))
total_ini = {'ini': total_ini}

def get_cap_fix(all_predictions):
    train_map = get_trainmap_fix()
    total = {}
    for name in inter:
        total[name] = []
    total['mAP'] = []

    time_steps = list(all_predictions.keys())
    time_steps = sorted(time_steps)

    for time_step in time_steps:
        predictions = all_predictions[time_step]

        totalap, num_ap = 0,0
        for name in inter:
            key,val = name,limitset[name]
            outputs = predictions.get(val, [""])
            if outputs != [""]:
                rec, prec,ap = voc_eval(key,val,all_testset,train_map,outputs,limitset)
            else:
                ap = 0
            if ap != -1:
                totalap += ap
                num_ap += 1
            total[key].append(ap)
        total['mAP'].append(totalap/num_ap)
            
    return total

def get_cap_idk(all_predictions):
    total = {}
    for name in inter:
        total[name] = []
    total['mAP'] = []
    total['idk'] = []    
    t_train_map = json.load(open(osp.join(model_path,time_step.split('_')[0] + '_trainmap.json'),'r'))
    time_steps = list(all_predictions.keys())
    time_steps = sorted(time_steps)

    for time_step in time_steps:
        predictions = all_predictions[time_step]
        train_map = {}
        for key in t_train_map:
            train_map[int(key)] = t_train_map[key]
        data_map = {value:key for key,value in train_map.items()}

        for name in inter: #common in train/test
            if limitset[name] not in train_map.values(): #has been seen during this frame
                total[name].append(-99)

        totalap, num_ap = 0,0
        for name in inter:
            if limitset[name] in train_map.values():
                key,val = name, data_map[limitset[name]]
                outputs = predictions.get(val, [""])
                if outputs != [""]:
                    rec, prec,ap = voc_eval(key,val,all_testset,train_map,outputs,limitset)
                else:
                    ap = 0
                if ap != -1:
                    totalap += ap
                    num_ap += 1
                total[key].append(ap)
        
        outputs = predictions.get(-1, [""])
        if outputs != [""]:
            rec, prec,ap = voc_eval('idk',-1,all_testset,train_map,outputs,limitset,idk=True)
        else:
            ap = 0
            
        if ap != -1:
            totalap += ap
            num_ap += 1
        total['idk'].append(ap)
        total['mAP'].append(totalap/num_ap)
                
    return total

# calculate cap
# if method == 'fix':
#     res = get_cap_fix(total_inf)
# else:
#     res = get_cap_idk(total_inf)
# f = open(osp.join(saved_dir,'res.json'),'w')
# f.write(json.dumps(res))
# f.close()

# calculate bwt and fwt
test = wunderlust()
from collections import defaultdict

def get_split_cap(catted_inf):
    train_map = get_trainmap_fix()
    split_res = []
    total_frames = 2000
    test_times = 50   # default value
    period   =   40  # default value

    total_cap = {}
    timesteps = sorted(list(catted_inf.keys()))

    for timestep in timesteps:
        cur_inf = catted_inf[timestep]
        total_cap[timestep] = {}
        for name in inter:
            if limitset[name] in train_map.keys():
                total_cap[timestep][name] = []
        total_cap[timestep]['mAP'] = []

        for test_time in range(test_times):
            if (test_time + 1) * period <= len(all_testset)-1:
                superframe = all_testset[test_time * period: (test_time+1) * period]
            else:
                superframe = all_testset[test_time * period:]
            
            predictions = defaultdict(list)
            for cls_name in cur_inf:
                tmplst = cur_inf[cls_name]
                for val in tmplst:
                    if val['image_id'] in superframe:
                        predictions[cls_name].append(val)

            totalap = 0
            num_ap = 0
            for name in inter:
                if limitset[name] in train_map.keys():
                    key,val = name,limitset[name]
                    outputs = predictions.get(val, [""])
                    if outputs != [""]:
                        rec, prec,ap = voc_eval(key,val,superframe,train_map,outputs,limitset)
                    else:
                        ap = 0
                if ap != -1:
                    totalap += ap
                    num_ap += 1
                total_cap[timestep][name].append(ap)

            mAP = -1 if num_ap == 0 else totalap / num_ap
            total_cap[timestep]['mAP'].append(mAP)

    return total_cap

def get_wt(): 
    catted_inf = {}
    for key in total_inf:
        if int(key.split('_')[0])%40 == 0 or int(key.split('_')[0]) == 1996:
            catted_inf[key] = total_inf[key]

    catted_cap = get_split_cap(catted_inf)
    ini_cap = get_split_cap(total_ini)
    
    f = open(osp.join(saved_dir,'catted_cap.json'),'w')
    f.write(json.dumps(catted_cap))
    f.close()
    f = open(osp.join(saved_dir, 'ini_cap.json'),'w')
    f.write(json.dumps(ini_cap))
    f.close()
    
    split_res_ini = ini_cap['ini']['mAP']
    split_res = [catted_cap[x]['mAP'] for x in catted_cap]
    test.B = np.array(split_res_ini)
    test.R = np.array(split_res)
    return test.FWT(),test.BWT()

fwt,bwt = get_wt()

# calculate forgetfulness
existence = {cat: [] for cat in inter}
for step_name in steps_name:
    jsons_name = sorted(os.listdir(osp.join(annodir, step_name)),key=lambda x: ('_'.join(x.split('_')[:3]), x.split('_')[3]))
    for json_name in jsons_name:
        f = osp.join(annodir,step_name,json_name)
        objs = json.load(open(f,'r'))
        for key, value in existence.items():
            existence[key].append(0)
        for obj in objs:
            label_cat = obj["category"]
            if label_cat in inter:
                existence[label_cat][-1] = 1

f = open(osp.join(saved_dir, 'existence.json'),'w')
f.write(json.dumps(existence))
f.close()

test.existence = existence
test.categories = inter
test.image_id = [40 * i for i in range(1,50)]
test.image_id.append(1996) # depend on how many times you test

res_dir = ''
test.map = json.load(open(res_dir,'r'))
def get_forgetfulness():
    test.forgetfulness()
    return test.forgetfulness
forget = get_forgetfulness()


def draw_curve(x,y,cat):
    fig = plt.figure(figsize=(10, 6))
    plt.plot((np.arange(len(x))+1) * 170 * 2 / 3600,y,linewidth=2.0)
    plt.xlabel('Time Span (hours)', fontsize=24)
    plt.ylabel('AP50', fontsize=24)
    plt.title(cat.capitalize(), fontsize=32)
    plt.grid(True)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.savefig(osp.join('your saved dir', '{}_ap_50_time.png'.format(cat)),
                bbox_inches='tight')
    plt.close(fig)