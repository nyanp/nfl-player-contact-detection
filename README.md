# 4th Place Solution - NFL Player Contact Detection

This repository covers the training code of Camaro's NN and 2nd stage GBDT models in our pipeline.
The rest of the code can be found in the following repositories.

K_mat's NN training:
https://www.kaggle.com/code/kmat2019/nfl-training-sample-4thplace-kmatpart

Camaro's NN training (camaro-exp117/nfl-exp048):
**TBD**

Inference Part:
https://www.kaggle.com/code/bamps53/lb0796-exp184-185-lgb095?scriptVersionId=120623474

## How to Run Code

### Camaro NN
```
# competition dataset
kaggle competitions download -c nfl-player-contact-detection
unzip nfl-player-contact-detection.zip -d ../input/nfl-player-contact-detection

# game fold
kaggle datasets download -d nyanpn/nfl-game-fold
unzip nfl-game-fold.zip -d ../input/nfl-game-fold

# pretrained model
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth
python camaro/scripts/save_cut_yolox_m_weight.py

# preprocess
python camaro/scripts/preprocess_data_frame.py
python camaro/scripts/preprocess_data_dict.py
python camaro/scripts/save_jpeg_images.py

# install
pip install -r requirements.txt

# run
python camaro_main.py --config_path camaro.configs.exp048
```

### 2nd-stage GBDT

1. run following script
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

2. execute notebook/train-gbdt.ipynb
When the execution is complete, the following four GBDT models are saved for the inference notebook.

| Directory                   | Feature Set     | Model    | Corresponding Datasets                                                                  |
|-----------------------------|-----------------|----------|-----------------------------------------------------------------------------------------|
| ../input/nyanp-model-a-0227 | K_mat(A)+Camaro | LightGBM | [nyanpn/nyanp-model-a-0227](https://www.kaggle.com/datasets/nyanpn/nyanp-model-a-0227/) |
| ../input/nyanp-model-b-0227 | K_mat(B)+Camaro | LightGBM | [nyanpn/nyanp-model-b-0227](https://www.kaggle.com/datasets/nyanpn/nyanp-model-b-0227/) |
| ../input/nfl-kmat-only-2    | K_mat(B)        | LightGBM | [nyanpn/nfl-kmat-only-2](https://www.kaggle.com/datasets/nyanpn/nfl-kmat-only-2)        |
| ../input/nfl-xgb-8030       | K_mat(B)+Camaro | XGBoost  | [nyanpn/nfl-xgb-8030](https://www.kaggle.com/datasets/nyanpn/nfl-xgb-8030)              |
