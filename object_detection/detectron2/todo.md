1. Train
    Run the corresponding main function in the specific method file folder (main_known.py in fix case and main_idk.py in idk case) to get the training result. To successfully run the code, you need to provide the following information.

    - In main_known.py / main_idk.py:
    curdir      # the dir you put the code at
    sourcedir   # source directory of the training frame
    annodir     # annotation directory of the training frame
    res_dir     # where you would like to save the file
    config_fp   # the dir of detectron config setting. 'detectron2/configs/PascalVOC-Detection/faster_rcnn_R_50_C4.yaml'

    - In util.py:
    cat_fp      # the category name and their corresponding id. 'mapping.json' is provided.
    pretrained_fp # the initial weight of the network. 'model_final_b1acc2.pkl' can be downloaded at https://dl.fbaipublicfiles.com/detectron2/PascalVOC-Detection/faster_rcnn_R_50_C4/142202221/model_final_b1acc2.pkl.

2. Test
    2.1 Run the test/inf_fix.py or inf_idk.py to get the inference result
        - You need to specify the sourcedir,annodir,config_fp,cat_fp directory just according to the information listed above.
    2.2 Run the test/cat_inf.py to get a concatenated inference file
        - It will take in all inference file from a method to generate a concatenated inference file, which will be used in further analysis.
    2.3 Run the function in test/result_analysis.py to get the metric, like cap, bwt/fwt, forgetfulness.


