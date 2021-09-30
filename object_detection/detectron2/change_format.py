p = 'D:/file/file/paper/cmu/online_meta-learning/code/detectron2-0.2/release/otherfile/new_inf.pkl'
import json
import pickle
res = pickle.load(open(p,'rb'))

for time_step in res:
    for cls_id in res[time_step]:
        for i in range(len(res[time_step][cls_id])):
            before = res[time_step][cls_id][i]['image_id']
            # 20140913_172950_458.mp4_cropped_34_1020.json
            after = before.split('.mp4_cropped_')
            after[1] = after[1].split('_')
            after[1][0] = after[1][0].zfill(5)
            after[1] = '_'.join(after[1])
            after = '_'.join(after)
            res[time_step][cls_id][i]['image_id'] = after

np = 'D:/file/file/paper/cmu/online_meta-learning/code/detectron2-0.2/release/otherfile/new_inf.pkl'
f = open(np,'wb')
pickle.dump(res,f)