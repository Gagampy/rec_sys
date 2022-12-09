# rec_sys
Recommender system advanced course

## 0. Environment setup:
`conda env create -f environment.yml & conda activate vb_recsys_course`  

 Environment update:  
`conda env update -n vb_recsys_course --file environment.yml`

## 1. Download Kaggle competition files (e.g. for Task 1):  
- Generate Kaggle API key-pair and save it as `./aux_files/kaggle.json`
- Run `download_competition_files.py --name=<your_competition_name>`

## 2. Run task 1, simple recommender system:
- Run `tasks_solutions/task_1_simple_recsys/run_pipeline.py --model_name=<model_name> --pipeline_type=<pipeline_type>`
- Parameters are:  
`model_name`: one of `most_frequent`, `svd_product_sim` and `svd_user_sim`  
`pipeline_type`: one of `train` and `predict`  
