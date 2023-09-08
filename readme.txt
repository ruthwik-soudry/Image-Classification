Description of all the files and dataset


1. Link of the dataset - https://www.kaggle.com/datasets/grassknoted/asl-alphabet
2. asl_classification_ddp.py - Python file to train the models using ddp module for different GPUs
To run the file  -  ‘python asl_classification_ddp.py alexnet 4’

3. asl_DP_resnet.py - Python file to train the resnet models on the dataset for different GPUs using DP module
4. asl_DP_inception.py - Python file to train the inception models on the dataset for different GPUs using DP module
5. ASL_TransferLearning_final.ipynb - Python notebook to show performance of different models using DP modules on sample of 10k datapoints.
6. asl_streamlit.py - Python file to load the model and host it on the server so that user can upload a image on we browser and gets the predicted class.
7. DP_10k_images_multiGPUs - This module consists different ipynb notebooks for DP module for different number of GPUs.

