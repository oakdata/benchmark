import os

import matplotlib.pyplot as plt
import json

def plot_helper(category_dictionary, epochs_list,save_path, metric_type):
    # if not os.path.exists(os.path.join(res_dir, trainer_name, 'plots')):
    #     os.mkdir(os.path.join(res_dir, trainer_name, 'plots'))
    for category, list_items in category_dictionary.items():
        plt.figure()
        plt.plot(epochs_list,list_items)
        plt.title(category)
        plt.xlabel('Epoch')
        plt.ylabel(f'{metric_type}')
        plt.savefig(os.path.join(save_path, f'{category}_{metric_type}.jpg'))
        plt.close()

if __name__ == '__main__':
    plot_ap_overall = True
    plot_ap_class = True
    ap_selection = 1

    path_to_jsons = '/project_data/held/jianrenw/helen/Deformable-DETR/output/test_4/eval'
    #'/project_data/held/jianrenw/helen/Deformable-DETR/output/incremental_4/eval'
    #'/project_data/held/jianrenw/helen/Deformable-DETR/output/test_4/eval'

    metric_dictionary = {}
    epoch_numbers = []
    metric_names = ['mAP', 'AP50', 'AP75', 'mAP_small', 'mAP_med', 'mAP_large', 'recall_1', 'recall_10', 'recall_100', 'recall_small', 'recall_med', 'recall_large']

    # Iterate through json files and retrieve relevant information
    for json_file in sorted([file for file in os.listdir(path_to_jsons) if file.endswith('json')],key=lambda x: int(x.split('_')[0])):
        
        # if not json_file.endswith('.json'):
        #     continue
        
        # Get the epoch number
        epoch_numbers.append(int(json_file.split('_')[0]))

        # Read the json file
        with open(os.path.join(path_to_jsons, json_file)) as f:
            data = json.load(f)
        
        # If we want to plot graphs for all individual classes
        for cat in data.keys():
            if cat not in metric_dictionary.keys():
                metric_dictionary[cat] = [data[cat][ap_selection]]
            else:
                metric_dictionary[cat].append(data[cat][ap_selection])
    
    # Plot the plots

    # Make the output plot directory if it doesn't exist yet
    if not os.path.exists(os.path.join(path_to_jsons, 'plots')):
        os.mkdir(os.path.join(path_to_jsons, 'plots'))
    if plot_ap_class:
        plot_helper(metric_dictionary, epoch_numbers, save_path = os.path.join(path_to_jsons, 'plots'), metric_type=metric_names[ap_selection])
        
    # elif plot_ap_overall:







