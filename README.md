# CTGNN
Multi-dimentional Information Integrated Graph Neural Network for Sequential Recommendation

We setup our experiment on a Titan V and 256G memory on CentOs.

Experiment Environment
-------
* python 3.6.5
* tensorflow-gpu 1.12
* numpy

Project Struct
------
### datasets file
* data\ctgnn\adj_matrix\ctgnn_clothes\s_norm_adj_mat_time.npz        ------ the adjacent matrix of Amazon clothes dataset 
* data\ctgnn\ctgnn_clothes_category_idx.pk                           ------ item categories of Amazon clothes
* data\ctgnn\ctgnn_clothes_time_4.txt.pk                             ------ train and test dataset

### program file
* main_ctgnn.py                   ------ this is the program entry
* model_ctgnn.py                  ------ the CTGNN model
* utils.py                        ------ some helper functions we used
* modules_time.py                 ------ the CTGNN model used function
* sampler_time_gcn.py             ------ the data processing function

Recommended Setup
------
You can run the main_ctgnn.py directly for easily running the program. 
If you run the code on linux, just running the following command:<br>
<br>
      `python main_ctgnn.py`
