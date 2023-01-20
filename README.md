## Prepare for camaro_main.py
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
pip install -r camaro_requirements.txt

# run
python camaro_main.py --config_path camaro.configs.exp048
```