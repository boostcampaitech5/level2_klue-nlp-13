import pickle as pickle
import os
import pandas as pd
import numpy as np
import torch.nn.functional as F
from datetime import datetime
from datetime import timezone
from datetime import timedelta
from glob import glob


def label_to_num(label):
  num_label = []
  with open('./utils/dict_label_to_num.pkl', 'rb') as f:
    dict_label_to_num = pickle.load(f)
  for v in label:
    num_label.append(dict_label_to_num[v])
  
  return num_label


def num_to_label(label):
  """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
  """
  origin_label = []
  with open('./utils/dict_num_to_label.pkl', 'rb') as f:
    dict_num_to_label = pickle.load(f)
  for v in label:
    origin_label.append(dict_num_to_label[v])
  
  return origin_label

def get_folder_name(CFG):
    now = datetime.now(tz=timezone(timedelta(hours=9)))
    folder_name = now.strftime('%Y-%m-%d-%H:%M:%S') + f"_{CFG['admin']}"
    save_path = f"./results/{folder_name}"
    CFG['save_path'] = save_path
    CFG['folder_name'] = folder_name
    os.makedirs(save_path)


def get_result(predicts):
    output_pred = []
    output_prob = []

    for predict in predicts:
        prob = F.softmax(predict, dim=-1).detach().cpu().numpy()
        predicts = predict.detach().cpu().numpy()
        pred = np.argmax(predict, axis=-1)
    
        output_pred.append(pred)
        output_prob.append(prob)
    return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()