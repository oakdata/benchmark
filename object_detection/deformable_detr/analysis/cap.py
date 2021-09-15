import pickle
import os
import os.path as osp
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import copy
import json
from scipy.ndimage import gaussian_filter1d

categories = [
    'person', 'bag', 'fence', 'trashcan', 'awning', 'signboard', 'streetlight',
    'light', 'wheelchair', 'flag', 'bench', 'car', 'graffiti', 'dome', 'lamp',
    'booth', 'stroller', 'bulletin board', 'door', 'dining table', 'umbrella',
    'stairs', 'shelf', 'chair', 'transformer', 'bottle', 'bus', 'column',
    'sofa', 'computer', 'van', 'truck', 'trafficcone', 'construction vehicles',
    'traffic light', 'barrier', 'bicycle', 'fireplug', 'parking meter',
    'mailbox', 'tactile paving', 'coffee table', 'clock', 'stool',
    'potted plant', 'manhole', 'food', 'railing', 'book', 'phone', 'desk',
    'dog', 'counter', 'waiting shed', 'zebra crossing', 'sink', 'motorcycle',
    'case', 'spoon', 'fork', 'plate', 'vase', 'apparel', 'sculpture', 'cup',
    'scooter', 'slide', 'basket', 'painting', 'balloon', 'knife',
    'air conditioner', 'curtain', 'monitor', 'bridge', 'box', 'fountain',
    'palm', 'fruit', 'gas bottle', 'toilet',  'radiator'
]

# categories = [
#     'person', 'bag', 'fence', 'trashcan', 'awning', 'signboard', 'streetlight',
#     'light', 'wheelchair', 'flag', 'bench', 'car', 'graffiti', 'dome', 'lamp',
#     'booth', 'stroller', 'bulletin board', 'door', 'dining table', 'umbrella',
#     'stairs', 'shelf', 'chair', 'transformer', 'bottle', 'bus', 'column',
#     'sofa', 'computer', 'van', 'truck', 'trafficcone', 'construction vehicles',
#     'traffic light', 'barrier', 'bicycle', 'fireplug', 'parking meter',
#     'mailbox', 'tactile paving', 'coffee table', 'clock', 'stool',
#     'potted plant', 'manhole', 'food', 'railing', 'book', 'phone', 'desk',
#     'dog', 'counter', 'waiting shed', 'zebra crossing', 'sink', 'motorcycle',
#     'case', 'spoon', 'fork', 'plate', 'vase', 'apparel', 'sculpture', 'cup',
#     'scooter', 'slide', 'basket', 'painting', 'balloon', 'knife',
#     'air conditioner', 'curtain', 'monitor', 'bridge', 'box', 'fountain',
#     'palm', 'fruit', 'gas bottle', 'cabinet', 'shopping cart'
# ]

# categories = [
#     'person', 'bag', 'fence', 'trashcan', 'awning', 'signboard', 'streetlight',
#     'light', 'wheelchair', 'flag', 'bench', 'car', 'graffiti', 'dome', 'lamp',
#     'booth', 'stroller', 'bulletin board', 'door', 'dining table', 'umbrella',
#     'stairs', 'shelf', 'chair', 'transformer', 'bottle', 'bus', 'column',
#     'sofa', 'computer', 'van', 'truck', 'trafficcone', 'construction vehicles',
#     'traffic light', 'barrier', 'bicycle', 'fireplug', 'parking meter',
#     'mailbox', 'tactile paving', 'coffee table', 'clock', 'stool',
#     'potted plant', 'manhole', 'food', 'railing', 'book', 'phone', 'desk',
#     'dog', 'counter', 'waiting shed', 'zebra crossing', 'sink', 'motorcycle',
#     'case', 'spoon', 'fork', 'plate', 'vase', 'apparel', 'sculpture', 'cup',
#     'scooter', 'slide', 'basket', 'painting', 'balloon', 'knife',
#     'air conditioner', 'curtain', 'monitor', 'bridge', 'box', 'fountain',
#     'palm', 'fruit', 'gas bottle'
# ]

# Get the category to sorted index correspondance
sorted_categories = sorted(categories)
category_dict = {}
for cat_idx, cat in enumerate(sorted_categories):
    category_dict[cat] = cat_idx

existence = {cat: [] for cat in categories}


# Needed frames
# f = open('/grogu/user/jianrenw/data/relabel/train_frame.json',)
# data_train_frames = json.load(f)
# f.close()

# for data_train_frame in data_train_frames:
#     video_name = data_train_frame.split('.')[0]
#     file_name = os.path.join('/grogu/user/jianrenw/data/oak', video_name,'标注结果', data_train_frame)
#     with open(file_name) as f:
#         data = json.load(f)
#     try:
#         data = data['DataList']
#     except TypeError:
#         pass
#     objs = data[0]["labels"]
#     for key, value in existence.items():
#         existence[key].append(0)
#     for obj in objs:
#         label_cat = obj["category"]
#         if label_cat in categories:
#             existence[label_cat][-1] = 1

video_dir = '/grogu/user/jianrenw/data/oak'
video_names = sorted(os.listdir(video_dir))

for video_name in video_names:
    label_names = sorted(os.listdir(
        osp.join('/grogu/user/jianrenw/data/oak', video_name, '标注结果')),
                         key=lambda x: int(x.split('_')[4]))
    for label_name in label_names:
        file_name = osp.join('/grogu/user/jianrenw/data/oak', video_name,
                             '标注结果', label_name)
        with open(file_name) as f:
            data = json.load(f)
        try:
            data = data['DataList']
        except TypeError:
            pass
        objs = data[0]["labels"]
        for key, value in existence.items():
            existence[key].append(0)
        for obj in objs:
            label_cat = obj["category"]
            if label_cat in categories:
                existence[label_cat][-1] = 1

incremental_stats = {cat: [] for cat in categories}
iCaRL_stats = {cat: [] for cat in categories}
EWC_stats = {cat: [] for cat in categories}
nonadaptation_stats = {cat: [] for cat in categories}
offline_stats = {cat: [] for cat in categories}

incremental_dir = '/grogu/user/jianrenw/helen/Detr-Continual-Learning/output/incremental_oak_finetune_1/eval'
iCaRL_dir = '/grogu/user/jianrenw/helen/Detr-Continual-Learning/output/ic2_oak_ft_10iter_bs8_1/eval'
EWC_dir = '/grogu/user/jianrenw/helen/Detr-Continual-Learning/output/ewc_oak_ft_10iter_bs8_1/eval'
# incremental_dir = '/grogu/user/jianrenw/baseline/code_ht/baseline_res/inc_idk_0.005_10_16_0.2_0.2_0.6/total_stats'
# iCaRL_dir = '/grogu/user/jianrenw/baseline/code_ht/baseline_res/ic2_idk_0.005_10_16_0.2_0.2_0.6/total_stats'
# EWC_dir =  '/grogu/user/jianrenw/baseline/code_ht/baseline_res/ewc_idk_0.005_10_16_0.2_0.2_0.6/total_stats'
# nonadaptation_file = '/grogu/user/jianrenw/baseline/code_ht/baseline_res/initial/[0.02858894481593434]_stat.json'
# offline_file = '/grogu/user/jianrenw/baseline/code_ht/baseline_res/new_oracle_fix_0.05_16/0.05_test/33830_[0.49477678892124366]_stat.json'

save_dir = '/grogu/user/jianrenw/helen/Detr-Continual-Learning/analysis/cap_analysis'

# incremental_files = sorted(os.listdir(incremental_dir))
# mAP = []
# for incremental_file in incremental_files:
#     result = incremental_file.split('_')[1][1:-1]
#     mAP.append(float(result))

# print(np.mean(mAP))

# icarl_files = sorted(os.listdir(iCaRL_dir))
# mAP = []
# for icarl_file in icarl_files:
#     result = icarl_file.split('_')[1][1:-1]
#     mAP.append(float(result))

# print(np.mean(mAP))

# ewc_files = sorted(os.listdir(EWC_dir))
# mAP = []
# for ewc_file in ewc_files:
#     result = ewc_file.split('_')[1][1:-1]
#     mAP.append(float(result))

# print(np.mean(mAP))

incremental_files = sorted([i for i in os.listdir(incremental_dir) if i.endswith('.json')], key=lambda x: int(x.split('_')[0]))
# print(incremental_files)
for incremental_file in incremental_files:
    if not incremental_file.endswith('.json'):
        continue
    number = int(incremental_file.split('_')[0])
    if number ==0 or number % 10 != 0:
        continue
    with open(osp.join(incremental_dir, incremental_file)) as f:
        results = json.load(f)
    for cat in categories:
        incremental_stats[cat].append(results[f"{category_dict[cat]}_{cat}"][1])

# iCaRL_files = sorted(os.listdir(iCaRL_dir))
iCaRL_files = sorted([i for i in os.listdir(iCaRL_dir) if i.endswith('.json')], key=lambda x: int(x.split('_')[0]))
for iCaRL_file in iCaRL_files:
    if not iCaRL_file.endswith('.json'):
        continue

    number = int(iCaRL_file.split('_')[0])
    if number ==0 or number % 10 != 0:
        continue

    with open(osp.join(iCaRL_dir, iCaRL_file)) as f:
        results = json.load(f)
    for cat in categories:
        iCaRL_stats[cat].append(results[f"{category_dict[cat]}_{cat}"][1])

EWC_files = sorted([i for i in os.listdir(EWC_dir) if i.endswith('.json')], key=lambda x: int(x.split('_')[0]))
for EWC_file in EWC_files:
    if not EWC_file.endswith('.json'):
        continue

    number = int(EWC_file.split('_')[0])
    if number ==0 or number % 10 != 0:
        continue

    with open(osp.join(EWC_dir, EWC_file)) as f:
        results = json.load(f)
    for cat in categories:
        EWC_stats[cat].append(results[f"{category_dict[cat]}_{cat}"][1])

'''
incremental_files = sorted(os.listdir(incremental_dir))
for incremental_file in incremental_files:
    with open(osp.join(incremental_dir, incremental_file)) as f:
        results = json.load(f)
    for cat in categories:
        if results[cat][0] != -99:
            incremental_stats[cat].append(results[cat][0])

iCaRL_files = sorted(os.listdir(iCaRL_dir))
for iCaRL_file in iCaRL_files:
    with open(osp.join(iCaRL_dir, iCaRL_file)) as f:
        results = json.load(f)
    for cat in categories:
        if results[cat][0] != -99:
            iCaRL_stats[cat].append(results[cat][0])

EWC_files = sorted(os.listdir(EWC_dir))
for EWC_file in EWC_files:
    with open(osp.join(EWC_dir, EWC_file)) as f:
        results = json.load(f)
    for cat in categories:
        if results[cat][0] != -99:
            EWC_stats[cat].append(results[cat][0])
'''
# with open(nonadaptation_file) as f:
#     results = json.load(f)
# for cat in categories:
#     nonadaptation_stats[cat].append(results[cat][0])

# with open(offline_file) as f:
#     results = json.load(f)
# for cat in categories:
#     offline_stats[cat].append(results[cat][0])

incremental_CAP = []
EWC_CAP = []
iCaRL_CAP = []
nonadaptation_CAP = []
offline_CAP = []

for cat in categories:

    fig = plt.figure(figsize=(10, 6))
    incremental_value = incremental_stats[cat]
    iCaRL_value = iCaRL_stats[cat]
    EWC_value = EWC_stats[cat]
    # nonadaptation_value = nonadaptation_stats[cat]
    # offline_value = offline_stats[cat]
    incremental_CAP.append(np.mean(incremental_value))
    iCaRL_CAP.append(np.mean(iCaRL_value))
    EWC_CAP.append(np.mean(EWC_value))
    # nonadaptation_CAP.append(nonadaptation_value[0])
    # offline_CAP.append(offline_value[0])

    plt.plot((np.arange(len(incremental_value))+1) * 170 * 2 / 3600,
            incremental_value,
            linewidth=4.0)
    plt.plot((np.arange(len(iCaRL_value))+1) * 170 * 2 / 3600,
            iCaRL_value,
            linewidth=4.0)
    plt.plot((np.arange(len(EWC_value))+1) * 170 * 2 / 3600,
            EWC_value,
            linewidth=4.0)
    # plt.plot((np.arange(len(EWC_value))+1) * 170 * 2 / 3600,
    #         [nonadaptation_value] * len(EWC_value),
    #         linewidth=4.0)
    # plt.plot((np.arange(len(EWC_value))+1) * 170 * 2 / 3600,
    #         [offline_value] * len(EWC_value),
    #         linewidth=4.0)
    cat_stats = existence[cat]
    x = np.arange(len(cat_stats)) * 2 / 3600
    plt.plot(x, cat_stats, linewidth=5.0, alpha=0.4, color='tab:gray')

    plt.xlabel('Time Span (hours)', fontsize=24)
    plt.ylabel('AP50', fontsize=24)
    plt.title(cat.capitalize(), fontsize=32)
    plt.grid(True)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.savefig(osp.join(save_dir, '{}_ap_50_time.png'.format(cat)),
                bbox_inches='tight')
    plt.close(fig)

iCaRL_CAP = np.array(iCaRL_CAP)
EWC_CAP = np.array(EWC_CAP)
incremental_CAP = np.array(incremental_CAP)
# nonadaptation_CAP = np.array(nonadaptation_CAP)
# offline_CAP = np.array(offline_CAP)

iCaRL_index = np.argsort(-iCaRL_CAP)
iCaRL_CAP = iCaRL_CAP[iCaRL_index]
iCaRL_cat = np.array(categories)[iCaRL_index]
print('iCaRL-20', np.mean(iCaRL_CAP[:20]))
content = {'cap': iCaRL_CAP.tolist(), 'cat': iCaRL_cat.tolist()}
with open('iCaRL_idk_CAP.json','w') as f:
    json.dump(content, f)

EWC_index = np.argsort(-EWC_CAP)
EWC_CAP = EWC_CAP[EWC_index]
EWC_cat = np.array(categories)[EWC_index]
print('EWC-20: ', np.mean(EWC_CAP[:20]))
content = {'cap': EWC_CAP.tolist(), 'cat': EWC_cat.tolist()}
with open('EWC_idk_CAP.json','w') as f:
    json.dump(content, f)

incremental_index = np.argsort(-incremental_CAP)
incremental_CAP = incremental_CAP[incremental_index]
incremental_cat = np.array(categories)[incremental_index]
print('Incremental-20: ', np.mean(incremental_CAP[:20]))
content = {'cap': incremental_CAP.tolist(), 'cat': incremental_cat.tolist()}
with open('incremental_idk_CAP.json','w') as f:
    json.dump(content, f)

print('iCaRL: ', np.mean(iCaRL_CAP))
print('EWC: ', np.mean(EWC_CAP))
print('incremental: ', np.mean(incremental_CAP))
# nonadaptation_index = np.argsort(-nonadaptation_CAP)
# nonadaptation_CAP = nonadaptation_CAP[nonadaptation_index]
# nonadaptation_cat = np.array(categories)[nonadaptation_index]
# print(np.mean(nonadaptation_CAP[:20]))
# content = {'cap': nonadaptation_CAP.tolist(), 'cat': nonadaptation_cat.tolist()}
# with open('nonadaptation_idk_CAP.json','w') as f:
#     json.dump(content, f)

# offline_index = np.argsort(-offline_CAP)
# offline_CAP = offline_CAP[offline_index]
# offline_cat = np.array(categories)[offline_index]
# print(np.mean(offline_CAP[:20]))
# content = {'cap': offline_CAP.tolist(), 'cat': offline_cat.tolist()}
# with open('offline_idk_CAP.json','w') as f:
#     json.dump(content, f)