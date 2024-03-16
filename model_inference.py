from model_model import TransformerXL
import os
import datetime
import torch
import yaml
import json
import numpy as np
from model_input import GenerateInput,EvalInput
from model_output import gen_midi


def inference(quavers,free_gen,inputs,bpm=120):
    cfg = yaml.full_load(open("model_config.yml", 'r'))
    inferenceConfig = cfg['INFERENCE']

    os.environ['CUDA_VISIBLE_DEVICES'] = inferenceConfig['devices']

    print('----Inference configs----')
    print(json.dumps(inferenceConfig, indent=1, sort_keys=True))

    # checkpoint information
    CHECKPOINT_FOLDER = inferenceConfig['checkpoint_dir']
    midi_folder = inferenceConfig["generated_dir"]

    checkpoint_type = inferenceConfig['checkpoint_type']
    if checkpoint_type == 'best_train':
        model_path = os.path.join(CHECKPOINT_FOLDER, 'model_best.pth.tar')
    elif checkpoint_type == 'best_val':
        model_path = os.path.join(CHECKPOINT_FOLDER, 'model_best_val.pth.tar')
    elif checkpoint_type == 'epoch_idx':
        model_path = os.path.join(CHECKPOINT_FOLDER, 'ep_{}.pth.tar'.format(str(inferenceConfig['model_epoch'])))

    pretrainCfg = yaml.full_load(open(os.path.join(CHECKPOINT_FOLDER, "config.yml"), 'r'))
    modelConfig = pretrainCfg['MODEL']

    # create result folder
    if not os.path.exists(midi_folder):
        os.mkdir(midi_folder)

    # declare model
    device = torch.device("cuda" if not inferenceConfig["no_cuda"] and torch.cuda.is_available() else "cpu")
    print('Device to generate:', device)

    model =  TransformerXL(
            modelConfig,
            device,
            is_training=False,
    )

    song_time_list = []
    words_len_list = []
    _, model_pretrained = model.get_model(model_path)

    #如果没有输入，将进入测试模式，调用测试集数据
    if inputs==None:
        input_conditions=[]
        eval_data = np.load('./data/test_data.npz')
        for e in range(18):
            input_conditions.append(EvalInput(quavers=eval_data['x'][e],free_gen=free_gen,word_embeddings=eval_data['condition'][e]))
    else:
        input_conditions = [GenerateInput(quavers=quavers, free_gen=free_gen, inputs=x) for x in inputs]

    num_samples = len(input_conditions)
    words_list = []
    path_list = []

    #逐个开始生成
    for idx in range(num_samples):
        print(f'---{idx}/{num_samples}---')
        input_condition=input_conditions[idx]
        print(f'quavers:',input_condition.quavers,'free_gen:',input_condition.free_gen)

        song_time, word_len, words= model.inference(
            model = model_pretrained,
            condition=input_condition,
            token_lim=2000, #8704
            strategies=['temperature', 'nucleus'],
            params={'t': 1.2, 'p': 0.9}
            )

        print('generation time:', song_time)
        print('number of tokens:', word_len)

        words_len_list.append(word_len)
        song_time_list.append(song_time)
        cur_date = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        output_path = '{}/{}.mid'.format(midi_folder, cur_date + '_' + str(idx))
        words_list.append(words)
        path_list.append(output_path)

        #生成midi
        gen_midi(words, output_path, bpm)


    print('avr token number:', np.mean(words_len_list))
    print('avr generation time:', np.mean(song_time_list))

    runtime_result = {
        'generation_time_list':song_time_list,
        'token_len_list': words_len_list,
        'avr token number': float(np.mean(words_len_list)),
        'avr generation time': float(np.mean(song_time_list)),
    }
    with open('runtime_stats.json', 'w') as f:
        json.dump(runtime_result, f)

    return path_list, words_list

if __name__ == '__main__':
    #Evaluation
    midi_folder, words_list=inference(quavers=0, free_gen=True, inputs=None, bpm=None)
