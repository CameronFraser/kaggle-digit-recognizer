# Kaggle - Digit Recognizer

To create gpu environment (recommended):
`conda env create -f environment-gpu.yml`

For GPU make sure you have CUDA toolkit 10.0 installed and not 10.1 as well as cuDNN 7.4.1.5. CUDA 10.0 doesn't work with tensorflow 2.0 for some reason.

To create cpu environment:
`conda env create -f environment.yml`

In order to download dataset you need to follow the directions on setting up an API token with Kaggle - [API credentials](https://github.com/Kaggle/kaggle-api#api-credentials)

To download dataset:
`python get_data.py`

Run jupyter notebook with `jupyter notebook` and open the Digit Recognizer notebook.

Submit results with `python submit.py`



This project uses [Black](https://github.com/python/black).