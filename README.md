# 4th Place Solution - NFL Player Contact Detection

## overview
This repository covers the training code for 2nd stage GBDT models in our pipeline.  
The rest of the code can be found in the following repositories.

K_mat's NN training:  
https://www.kaggle.com/code/kmat2019/nfl-training-sample-4thplace-kmatpart

Camaro's NN training:  
https://github.com/bamps53/kaggle-nfl2022  

As for details of our entire solution, Please check out this discussion.  
https://www.kaggle.com/competitions/nfl-player-contact-detection/discussion/391761

---

## 2nd Stage GBDT models training

### 1. Preparation
```
# oof of camaro NN
kaggle datasets download -d bamps53/camaro-exp117
unzip camaro-exp117.zip -d ../input/camaro-exp117

kaggle datasets download -d bamps53/nfl-exp048
unzip nfl-exp048.zip -d ../input/nfl-exp048

# oof of kmat NN
kaggle datasets download -d kmat2019/mfl2cnnkmat0221
unzip mfl2cnnkmat0221.zip -d ../input/mfl2cnnkmat0221

# install
pip install -r requirements.txt
```

### 2. Training
Please execute notebook/train-gbdt.ipynb  
When the execution is complete, the following 4 GBDT models are saved for the inference notebook.

| Directory                   | Feature Set     | Model    | Corresponding Datasets                                                                  |
|-----------------------------|-----------------|----------|-----------------------------------------------------------------------------------------|
| ../input/nyanp-model-a-0227 | K_mat(A)+Camaro | LightGBM | [nyanpn/nyanp-model-a-0227](https://www.kaggle.com/datasets/nyanpn/nyanp-model-a-0227/) |
| ../input/nyanp-model-b-0227 | K_mat(B)+Camaro | LightGBM | [nyanpn/nyanp-model-b-0227](https://www.kaggle.com/datasets/nyanpn/nyanp-model-b-0227/) |
| ../input/nfl-kmat-only-2    | K_mat(B)        | LightGBM | [nyanpn/nfl-kmat-only-2](https://www.kaggle.com/datasets/nyanpn/nfl-kmat-only-2)        |
| ../input/nfl-xgb-8030       | K_mat(B)+Camaro | XGBoost  | [nyanpn/nfl-xgb-8030](https://www.kaggle.com/datasets/nyanpn/nfl-xgb-8030)              |


### 3. Inference
Plase refer to this notebook.  
Our best submission is made of above 4 GBDT models and 1 camaro 2nd stage model.  
https://www.kaggle.com/code/bamps53/lb0796-exp184-185-lgb095?scriptVersionId=120623474
