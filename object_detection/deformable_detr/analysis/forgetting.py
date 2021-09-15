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

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def find_nearest_existence(category, existence, current_frame):
    i = 1
    find = False
    while not find:
        left = max([0, current_frame - i])
        # print(existence[category])
        # print(current_frame - i)
        # print('--', len(existence[category]))
        if existence[category][left] > 0:
            find = True
        else:
            i += 1
        if left == 0 and not find:
            i = -1
            break
    return i

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
categories = [
    'person', 'bag', 'fence', 'trashcan', 'awning', 'signboard', 'streetlight',
    'light', 'wheelchair', 'flag', 'bench', 'car', 'graffiti', 'dome', 'lamp',
    'booth', 'stroller', 'bulletin board', 'door', 'dining table', 'umbrella',
    'stairs', 'shelf', 'chair', 'transformer', 'bottle', 'bus', 'column',
    'sofa', 'computer', 'van', 'truck', 'trafficcone', 'construction vehicles',
    'traffic light', 'barrier', 'bicycle', 'fireplug', 'parking meter',
    'mailbox', 'tactile paving', 'coffee table', 'clock', 'stool',
    'potted plant', 'manhole', 'food', 'railing', 'book', 'phone', 'desk',
    'dog', 'counter', 'waiting shed', 'zebra crossing', 'motorcycle',
    'case', 'spoon', 'fork', 'plate', 'vase', 'apparel', 'sculpture', 'cup',
    'scooter', 'slide', 'basket', 'painting', 'balloon', 'knife',
    'air conditioner', 'curtain', 'monitor', 'bridge', 'box', 'fountain',
    'palm', 'fruit', 'gas bottle', 'toilet', 'sink', 'radiator'
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
#     'palm', 'fruit', 'gas bottle'
# ]

existence = {cat: [] for cat in categories}


# # Needed frames
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


# Get the category to sorted index correspondance
sorted_categories = sorted(categories)
category_dict = {}
for cat_idx, cat in enumerate(sorted_categories):
    category_dict[cat] = cat_idx

print(category_dict)
stats = {cat: [] for cat in categories}
incremental_stats = {cat: [] for cat in categories}
iCaRL_stats = {cat: [] for cat in categories}
EWC_stats = {cat: [] for cat in categories}
nonadaptation_stats = {cat: [] for cat in categories}
offline_stats = {cat: [] for cat in categories}

incremental_dir = '/grogu/user/jianrenw/helen/Detr-Continual-Learning/output/incremental_oak_finetune_1/eval'
iCaRL_dir = '/grogu/user/jianrenw/helen/Detr-Continual-Learning/output/ic2_oak_ft_10iter_bs8_1/eval'
EWC_dir = '/grogu/user/jianrenw/helen/Detr-Continual-Learning/output/ewc_oak_ft_10iter_bs8_1/eval'
# nonadaptation_file = '/grogu/user/jianrenw/baseline/code_ht/baseline_res/initial/[0.02858894481593434]_stat.json'
# offline_file = '/grogu/user/jianrenw/baseline/code_ht/baseline_res/new_oracle_fix_0.05_16/0.05_test/33830_[0.49477678892124366]_stat.json'

save_dir = '/grogu/user/jianrenw/helen/Detr-Continual-Learning/analysis/forgetting_analysis'

#'/grogu/user/jianrenw/wanderlust/camera_ready_analysis/forgetfulness'

# folders = [incremental_dir, iCaRL_dir, EWC_dir]

# for folder in folders:
#     files = sorted(os.listdir(folder))

#     for file in files:
#         if not file.endswith('.json'):
#             continue
#         with open(osp.join(folder, file)) as f:
#             results = json.load(f)
#         for cat in categories:
#             stats[cat].append(results[f"{category_dict[cat]}_{cat}"][1])
#     for cat in categories:

#         fig = plt.figure(figsize=(10, 6))
#         value = stats[cat]
#         # iCaRL_value = iCaRL_stats[cat]
#         # EWC_value = EWC_stats[cat]

#         ap_time_gap = {}
#         all_dis = []
#         all_ap = []
#         for i in range(199):
#             dis = find_nearest_existence(cat, existence, (i + 1) * 170)
#             if dis == -1:
#                 continue
#             else:
#                 if dis in ap_time_gap:
#                     ap_time_gap[dis].append(value[i])
#                 else:
#                     ap_time_gap[dis] = [value[i]]

#         for dis, ap in ap_time_gap.items():
#             all_dis.append(dis)
#             all_ap.append(np.mean(ap))
#         sorted_dis, sorted_ap = zip(*sorted(zip(all_dis, all_ap)))
#         # sorted_dis = np.array(sorted_dis)
#         sorted_dis = np.array(sorted_dis) - sorted_dis[0] 
#         sorted_forgetfulness = sorted_ap[0] - np.array(sorted_ap)
#         forgetfulness = 1 / np.sum(sorted_dis) * np.sum(
#                 sorted_dis * sorted_forgetfulness)
#         forget.append(forgetfulness)
#         plt.plot(sorted_dis * 2,
#                 gaussian_filter1d(sorted_forgetfulness,1),
#                 linewidth=4.0)

#         ap_time_gap = {}
#         all_dis = []
#         all_ap = []
#         for i in range(199):
#             dis = find_nearest_existence(cat, existence, (i + 1) * 170)
#             if dis == -1:
#                 continue
#             else:
#                 if dis in ap_time_gap:
#                     ap_time_gap[dis].append(iCaRL_value[i])
#                 else:
#                     ap_time_gap[dis] = [iCaRL_value[i]]

#         for dis, ap in ap_time_gap.items():
#             all_dis.append(dis)
#             all_ap.append(np.mean(ap))
#         sorted_dis, sorted_ap = zip(*sorted(zip(all_dis, all_ap)))
#         # sorted_dis = np.array(sorted_dis)
#         sorted_dis = np.array(sorted_dis) - sorted_dis[0] 
#         sorted_forgetfulness = sorted_ap[0] - np.array(sorted_ap)
#         forgetfulness = 1 / np.sum(sorted_dis) * np.sum(
#                     sorted_dis * sorted_forgetfulness)
#         iCaRL_forget.append(forgetfulness)
#         plt.plot(sorted_dis * 2,
#                 gaussian_filter1d(sorted_forgetfulness,1),
#                 linewidth=4.0)

#         ap_time_gap = {}
#         all_dis = []
#         all_ap = []
#         for i in range(199):
#             dis = find_nearest_existence(cat, existence, (i + 1) * 170)
#             if dis == -1:
#                 continue
#             else:
#                 if dis in ap_time_gap:
#                     ap_time_gap[dis].append(EWC_value[i])
#                 else:
#                     ap_time_gap[dis] = [EWC_value[i]]

#         for dis, ap in ap_time_gap.items():
#             all_dis.append(dis)
#             all_ap.append(np.mean(ap))
#         sorted_dis, sorted_ap = zip(*sorted(zip(all_dis, all_ap)))
#         # sorted_dis = np.array(sorted_dis)
#         sorted_dis = np.array(sorted_dis) - sorted_dis[0] 
#         sorted_forgetfulness = sorted_ap[0] - np.array(sorted_ap)
#         forgetfulness = 1 / np.sum(sorted_dis) * np.sum(
#                     sorted_dis * sorted_forgetfulness)
#         EWC_forget.append(forgetfulness)
#         plt.plot(sorted_dis * 2,
#                 gaussian_filter1d(sorted_forgetfulness,1),
#                 linewidth=4.0)

#         plt.xlabel('Time from Last Label (seconds)', fontsize=24)
#         plt.ylabel('Forgetting', fontsize=24)
#         plt.title(cat.capitalize(), fontsize=32)
#         # plt.legend(['Incremental','iCaRL','EWC'], loc='upper left', fontsize=18)
#         plt.grid(True)
#         plt.xticks(fontsize=24)
#         plt.yticks(fontsize=24)
#         plt.savefig(osp.join(save_dir,'{}_forgetting.png'.format(cat)), bbox_inches='tight')
#         plt.close(fig)

        
incremental_files = sorted([i for i in os.listdir(incremental_dir) if i.endswith('.json')], key=lambda x: int(x.split('_')[0]))
for incremental_file in incremental_files:
    if not incremental_file.endswith('.json'):
        continue
    number = int(incremental_file.split('_')[0])
    if number ==0 or number % 10 != 0:
        continue
    # print(number)
    with open(osp.join(incremental_dir, incremental_file)) as f:
        results = json.load(f)
    for cat in categories:
        incremental_stats[cat].append(results[f"{category_dict[cat]}_{cat}"][1])

iCaRL_files = sorted([i for i in os.listdir(iCaRL_dir) if i.endswith('.json')], key=lambda x: int(x.split('_')[0]))
for iCaRL_file in iCaRL_files:
    if not iCaRL_file.endswith('.json'):
        continue

    number = int(iCaRL_file.split('_')[0])
    if number == 0 or number % 10 != 0:
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




incremental_forget = []
EWC_forget = []
iCaRL_forget = []


for cat in categories:
    fig = plt.figure(figsize=(10, 6))
    incremental_value = incremental_stats[cat]
    iCaRL_value = iCaRL_stats[cat]
    EWC_value = EWC_stats[cat]

    ap_time_gap = {}
    all_dis = []
    all_ap = []
    for i in range(199):
        dis = find_nearest_existence(cat, existence, (i + 1) * 170)
        if dis == -1:
            continue
        else:
            if dis in ap_time_gap:
                ap_time_gap[dis].append(incremental_value[i])
            else:
                ap_time_gap[dis] = [incremental_value[i]]

    for dis, ap in ap_time_gap.items():
        all_dis.append(dis)
        all_ap.append(np.mean(ap))
    sorted_dis, sorted_ap = zip(*sorted(zip(all_dis, all_ap)))
    # sorted_dis = np.array(sorted_dis)
    sorted_dis = np.array(sorted_dis) - sorted_dis[0] 
    sorted_forgetfulness = sorted_ap[0] - np.array(sorted_ap)
    forgetfulness = 1 / np.sum(sorted_dis) * np.sum(
            sorted_dis * sorted_forgetfulness)
    incremental_forget.append(forgetfulness)
    plt.plot(sorted_dis * 2,
             gaussian_filter1d(sorted_forgetfulness,1),
             linewidth=4.0)

    ap_time_gap = {}
    all_dis = []
    all_ap = []
    for i in range(199):
        dis = find_nearest_existence(cat, existence, (i + 1) * 170)
        if dis == -1:
            continue
        else:
            if dis in ap_time_gap:
                ap_time_gap[dis].append(iCaRL_value[i])
            else:
                ap_time_gap[dis] = [iCaRL_value[i]]

    for dis, ap in ap_time_gap.items():
        all_dis.append(dis)
        all_ap.append(np.mean(ap))
    sorted_dis, sorted_ap = zip(*sorted(zip(all_dis, all_ap)))
    # sorted_dis = np.array(sorted_dis)
    sorted_dis = np.array(sorted_dis) - sorted_dis[0] 
    sorted_forgetfulness = sorted_ap[0] - np.array(sorted_ap)
    forgetfulness = 1 / np.sum(sorted_dis) * np.sum(
                sorted_dis * sorted_forgetfulness)
    iCaRL_forget.append(forgetfulness)
    plt.plot(sorted_dis * 2,
             gaussian_filter1d(sorted_forgetfulness,1),
             linewidth=4.0)

    ap_time_gap = {}
    all_dis = []
    all_ap = []
    for i in range(199):
        dis = find_nearest_existence(cat, existence, (i + 1) * 170)
        if dis == -1:
            continue
        else:
            if dis in ap_time_gap:
                ap_time_gap[dis].append(EWC_value[i])
            else:
                ap_time_gap[dis] = [EWC_value[i]]

    for dis, ap in ap_time_gap.items():
        all_dis.append(dis)
        all_ap.append(np.mean(ap))
    sorted_dis, sorted_ap = zip(*sorted(zip(all_dis, all_ap)))
    # sorted_dis = np.array(sorted_dis)
    sorted_dis = np.array(sorted_dis) - sorted_dis[0] 
    sorted_forgetfulness = sorted_ap[0] - np.array(sorted_ap)
    forgetfulness = 1 / np.sum(sorted_dis) * np.sum(
                sorted_dis * sorted_forgetfulness)
    EWC_forget.append(forgetfulness)
    plt.plot(sorted_dis * 2,
             gaussian_filter1d(sorted_forgetfulness,1),
             linewidth=4.0)

    plt.xlabel('Time from Last Label (seconds)', fontsize=24)
    plt.ylabel('Forgetting', fontsize=24)
    plt.title(cat.capitalize(), fontsize=32)
    # plt.legend(['Incremental','iCaRL','EWC'], loc='upper left', fontsize=18)
    plt.grid(True)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.savefig(osp.join(save_dir,'{}_forgetting.png'.format(cat)), bbox_inches='tight')
    plt.close(fig)

iCaRL_forget = np.array(iCaRL_forget)
EWC_forget = np.array(EWC_forget)
incremental_forget = np.array(incremental_forget)

print('iCarl: ',np.mean(iCaRL_forget))
print('EWC: ',np.mean(EWC_forget))
print('Incremental: ', np.mean(incremental_forget))


iCaRL_index = np.argsort(-iCaRL_forget)
iCaRL_forget = iCaRL_forget[iCaRL_index]
iCaRL_cat = np.array(categories)[iCaRL_index]
print('iCarl-20: ', np.mean(iCaRL_forget[:20]))
content = {'cap': iCaRL_forget.tolist(), 'cat': iCaRL_cat.tolist()}
with open(os.path.join(save_dir,'iCaRL_forget.json'),'w') as f:
    json.dump(content, f)

EWC_index = np.argsort(-EWC_forget)
EWC_forget = EWC_forget[EWC_index]
EWC_cat = np.array(categories)[EWC_index]
print('EWC-20: ', np.mean(EWC_forget[:20]))
content = {'cap': EWC_forget.tolist(), 'cat': EWC_cat.tolist()}
with open(os.path.join(save_dir,'EWC_forget.json'),'w') as f:
    json.dump(content, f)

incremental_index = np.argsort(-incremental_forget)
incremental_forget = incremental_forget[incremental_index]
incremental_cat = np.array(categories)[incremental_index]
print('Incremental-20: ',np.mean(incremental_forget[:20]))
content = {'cap': incremental_forget.tolist(), 'cat': incremental_cat.tolist()}
with open(os.path.join(save_dir,'incremental_forget.json'),'w') as f:
    json.dump(content, f)