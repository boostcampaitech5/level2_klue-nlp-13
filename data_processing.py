import yaml
from utils.DataAugmentation import *
from utils.DataPreprocessing import *

if __name__ == '__main__':
    with open('config.yaml') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    data_augmentation(cfg)