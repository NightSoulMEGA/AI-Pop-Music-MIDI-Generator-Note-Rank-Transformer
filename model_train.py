
import os
import json
import yaml
import datetime
import numpy as np

import torch
from model_model import TransformerXL


def train():
    # config
    modelConfig, trainConfig = get_configs()

    # load train data
    training_data = np.load(os.path.join(trainConfig['ROOT'],'train_data.npz'))

    os.environ['CUDA_VISIBLE_DEVICES'] = trainConfig['devices']
    device = torch.device("cuda:{}".format(trainConfig['gpuID']) if not trainConfig["no_cuda"] and torch.cuda.is_available() else "cpu")
    print('Device to train:', device)
    
    resume = None

    # declare model
    model = TransformerXL(
            modelConfig,
            device,
            is_training=True)

    # train
    model.train(training_data,
                trainConfig,
                device,
                resume)
            

def get_configs():
    cfg = yaml.full_load(open("model_config.yml", 'r'))

    modelConfig = cfg['MODEL']
    trainConfig = cfg['TRAIN']

    cur_date = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    experiment_Dir = os.path.join(trainConfig['output_dir'],cur_date)
    if not os.path.exists(experiment_Dir):
        os.makedirs(experiment_Dir) 
    print('Experiment: ', experiment_Dir)
    trainConfig.update({'experiment_Dir': experiment_Dir})


    with open(os.path.join(experiment_Dir, 'config.yml'), 'w') as f:
        yaml.dump(cfg, f)

    print('----Model configs----')
    print(json.dumps(modelConfig, indent=1, sort_keys=True))
    print('----Training configs----')
    print(json.dumps(trainConfig, indent=1, sort_keys=True))
    return modelConfig, trainConfig


if __name__ == '__main__':
    train()


