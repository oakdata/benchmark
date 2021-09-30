# trainmap (112 pairs)
# 1996 * 1(length)
import os
import os.path as osp
import json

# annodir = '/grogu/user/jianrenw/data/OAK_LABEL_N'
# steps_name = sorted(os.listdir(annodir))
# train_map = {i:i for i in range(0,20)} # net : annotation


# cat_fp = '/grogu/user/jianrenw/baseline/release/mapping.json'

# def returnmap():
#     f = open(cat_fp, 'r')
#     content = f.read()
#     categories = json.loads(content)
#     return categories

# def calclass(files_name, train_map, limitset):
#     for file_name in files_name:
#         step_name, json_name = file_name.split('#')[0], file_name.split('#')[1]
#         f = osp.join(annodir,step_name,json_name)
#         labels = json.load(open(f,'r'))
        
#         for label in labels:
#             category = label['category']
#             if category in limitset.keys():
#                 if limitset[category] not in train_map.values():
#                     train_map [ len(train_map) ] = limitset[category]
#     return train_map
  

# limitset = returnmap()
# final_length = []
# for step_name in steps_name:
#     jsons_name = sorted(os.listdir(osp.join(annodir, step_name)),key=lambda x: ('_'.join(x.split('_')[:3]), x.split('_')[3]))
#     trainset = [step_name + '#' +  json_name for json_name in jsons_name]
#     trainmap = calclass(trainset, train_map, limitset)
#     final_length.append(len(train_map))

# p = '/grogu/user/jianrenw/baseline/release/baseline_res'
# f = open(osp.join(p,'trainmap.json'),'w')
# f.write(json.dumps(trainmap))

# f = open(osp.join(p,'trainmap_length.json'),'w')
# f.write(json.dumps(final_length))

# pth = '/grogu/user/jianrenw/data/OAK_LABEL_N'
# steps_name = osp.join(sorted(os.listdir(pth)))

# for step_name in steps_name:
#     json_names = sorted(os.listdir(osp.join(pth,step_name)),key=lambda x: ('_'.join(x.split('_')[:3]), x.split('_')[3]))
    
    
# mapping
# annodir = '/grogu/user/jianrenw/data/OAK_LABEL_N'
# steps_name = sorted(os.listdir(annodir))
# for step_name in steps_name:
#     pass

# inter
# train_cat = set()
# test_cat = set()
# train_pth = '/grogu/user/jianrenw/data/OAK_LABEL_N'
# test_pth = '/grogu/user/jianrenw/data/OAK_TEST/Label'

# steps_name = sorted(os.listdir(train_pth))
# for step_name in steps_name:
#     jsons_name = sorted(os.listdir(osp.join(train_pth,step_name)))
#     for json_name in jsons_name:
#         labels = json.load(open(osp.join(train_pth,step_name,json_name),'r'))
#         for label in labels:
#             category = label['category']
#             if category not in train_cat:
#                 train_cat.add(category)

# jsons_name = sorted(os.listdir(test_pth))
# for json_name in jsons_name:
#     labels = json.load(open(osp.join(test_pth,json_name),'r'))
#     for label in labels:
#         category = label['category']
#         if category not in test_cat:
#             test_cat.add(category)

