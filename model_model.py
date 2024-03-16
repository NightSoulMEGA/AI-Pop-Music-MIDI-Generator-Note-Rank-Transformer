import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import copy
import os
import time
import json

from model_modules import MemTransformerLM
import model_saver as saver



def network_paras(model):
    # compute only trainable params
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


class TransformerXL(object):
    def __init__(self, modelConfig, device, is_training=True):


        self.modelConfig = modelConfig

        # model settings    
        self.n_layer= modelConfig['n_layer']
        self.d_model = modelConfig['d_model']
        self.seq_len= modelConfig['seq_len']
        self.mem_len =  modelConfig['mem_len']

        self.tgt_len = modelConfig['tgt_len']
        self.ext_len = modelConfig['ext_len']
        self.eval_tgt_len = modelConfig['eval_tgt_len']

        self.init = modelConfig['init']
        self.init_range = modelConfig['init_range']
        self.init_std = modelConfig['init_std']
        self.proj_init_std = modelConfig['proj_init_std']

        #mode
        self.is_training = is_training
        self.device = device  
        

    def init_weight(self, weight):
        if self.init == 'uniform':
            nn.init.uniform_(weight, -self.init_range, self.init_range)
        elif self.init == 'normal':
            nn.init.normal_(weight, 0.0, self.init_std)

    def init_bias(self, bias):
        nn.init.constant_(bias, 0.0)
            
    def weights_init(self,m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                self.init_weight(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                self.init_bias(m.bias)
        elif classname.find('Embedding') != -1:
            if hasattr(m, 'weight'):
                self.init_weight(m.weight)
        elif classname.find('LayerNorm') != -1:
            if hasattr(m, 'weight'):
                nn.init.normal_(m.weight, 1.0, self.init_std)
            if hasattr(m, 'bias') and m.bias is not None:
                self.init_bias(m.bias)
        elif classname.find('TransformerLM') != -1:
            if hasattr(m, 'r_emb'):
                self.init_weight(m.r_emb)
            if hasattr(m, 'r_w_bias'):
                self.init_weight(m.r_w_bias)
            if hasattr(m, 'r_r_bias'):
                self.init_weight(m.r_r_bias)
            if hasattr(m, 'r_bias'):
                self.init_bias(m.r_bias)


    def get_model(self, pretrain_model=None):
        model = MemTransformerLM(self.modelConfig, is_training=self.is_training)

        st_eopch = 0
        if pretrain_model:
            if self.is_training:
                checkpoint = torch.load(pretrain_model, map_location='cuda:0')
            else:
                #由于设备限制，这里添加了使用cpu运行的逻辑
                if torch.cuda.is_available():
                    checkpoint = torch.load(pretrain_model, map_location='cuda:0')
                else:
                    checkpoint = torch.load(pretrain_model, map_location='cpu')
            print('Pretrained model config:')
            print('epoch: ', checkpoint['epoch'])
            print('best_loss: ', checkpoint['best_loss'])
            print(json.dumps(checkpoint['model_setting'], indent=1, sort_keys=True))
            print(json.dumps(checkpoint['train_setting'], indent=1, sort_keys=True))

            try:
                model.load_state_dict(checkpoint['state_dict'])
                print('{} loaded.'.format(pretrain_model))  
            except:
                print('Loaded weights have different shapes with the model. Please check your model setting.')
                exit()
            st_eopch = checkpoint['epoch'] + 1

        else:
            model.apply(self.weights_init)
            model.word_emb.apply(self.weights_init) 
        return st_eopch ,model.to(self.device)


    def save_checkpoint(self, state, root, save_freq=10):
        if state['epoch'] % save_freq == 0:
            torch.save(state, os.path.join(root,'ep_{}.pth.tar'.format(state['epoch'])))

    def train_loss_record(self, epoch, train_loss,checkpoint_dir, val_loss=None):

        if val_loss:
            df = pd.DataFrame({'epoch': [epoch+1],
                    'train_loss': ['%.3f'%train_loss],
                    'val_loss': ['%.3f'%val_loss]})
            
        else:
            df = pd.DataFrame({'epoch': [epoch+1],
                    'train_loss': ['%.3f'%train_loss]})

        csv_file = os.path.join(checkpoint_dir, 'loss.csv')

        if not os.path.exists(csv_file):
            df.to_csv(csv_file, index=False)
        else:
            df.to_csv(os.path.join(checkpoint_dir, 'loss.csv'), mode='a', header=False,  index=False)

    def train(self, train_data, trainConfig, device, resume):
        checkpoint_dir = trainConfig['experiment_Dir']  #save location
        batch_size = trainConfig['batch_size']
        torch.manual_seed(trainConfig["seed"])

        # create saver
        saver_agent = saver.Saver(checkpoint_dir)

        #Prepare model
        if resume != 'None':
            st_epoch, model = self.get_model(resume)
            print('Continue to train from {} epoch'.format(st_epoch))
        else:
            st_epoch, model = self.get_model()

        optimizer = optim.Adam(model.parameters(), lr=trainConfig['lr'])    #learning rate
        epoch_train_loss = []
        save_freq = trainConfig['save_freq']
        
        n_parameters = network_paras(model)
        print('n_parameters: {:,}'.format(n_parameters))
        saver_agent.add_summary_msg(
            ' > params amount: {:,d}'.format(n_parameters))

        # unpack
        train_x = train_data['x']  #(880, 17, 512)
        train_y = train_data['y'] 
        mask = train_data['mask'] 
        num_groups = train_data['num_groups']
        condition = train_data['condition'] #(880, 128, 512)

        num_batches = len(train_x) // batch_size
        
        print('>>> Start training')
        for epoch in range(st_epoch, trainConfig['num_epochs']):
            saver_agent.global_step_increment()

            train_loss = []
            st_time = time.time()
            model.train()

            for bidx in range(num_batches):

                model.zero_grad()

                # start and end location of this batch
                bidx_st = batch_size * bidx
                bidx_ed = batch_size * (bidx + 1)

                # get the data of this batch
                batch_x = train_x[bidx_st:bidx_ed]  # (8,17,512)
                batch_y = train_y[bidx_st:bidx_ed]
                batch_mask = mask[bidx_st:bidx_ed]
                batch_condition = condition[bidx_st:bidx_ed]    #(8, 128, 512)
                n_group  = np.max(num_groups[bidx_st:bidx_ed])

                # proc groups
                mems = tuple()
                for gidx in range(n_group):
                    #get the data of the No.gidx group in this batch
                    group_x = batch_x[:, gidx, :]   # (8,512)
                    group_y = batch_y[:, gidx, :]
                    group_mask = batch_mask[:, gidx, :]
                    group_condition = batch_condition    #(8, 128, 512)

                    group_x = torch.from_numpy(group_x).permute(1, 0).contiguous().to(self.device).long()  # [512,8]
                    group_y = torch.from_numpy(group_y).permute(1, 0).contiguous().to(self.device).long()
                    group_mask = torch.from_numpy(group_mask).to(self.device).float()   #[8,512]
                    group_condition = torch.from_numpy(group_condition).to(self.device).float() #[8,128,512]

                    ret = model(group_x, group_y, group_mask, group_condition, *mems)    #return
                    loss, mems = ret[0], ret[1:]
                    train_loss.append(loss.item())
                    loss.backward()

                    # output log
                    sys.stdout.write('epoch:{:3d}/{:3d}, batch: {:4d}/{:4d}, group: {:2d}/{:2d} | Loss: {:6f}\r'.format(
                        epoch,
                        trainConfig['num_epochs'],
                        bidx,
                        num_batches,
                        gidx,
                        n_group, 
                        loss.item()
                    ))
                    sys.stdout.flush()

                optimizer.step()

            curr_train_loss = sum(train_loss) / len(train_loss)
            saver_agent.add_summary('epoch loss', curr_train_loss)

            epoch_train_loss.append(curr_train_loss)
            epoch_info = 'Epoch: {}, Train Loss: {:.5f} ,  T: {:.3f}'.format(epoch+1, curr_train_loss, time.time()-st_time)
            print(epoch_info)

            self.train_loss_record(epoch, curr_train_loss, checkpoint_dir)
            self.save_checkpoint({
                    'epoch': epoch + 1,
                    'model_setting': self.modelConfig,
                    'train_setting': trainConfig,
                    'state_dict': model.state_dict(),
                    'best_loss': curr_train_loss,
                    'optimizer' : optimizer.state_dict(),
                                }, 
                    checkpoint_dir, 
                    save_freq)

            if curr_train_loss < 0.01:
                print('Experiment [{}] finished at loss < 0.01.'.format(checkpoint_dir))
                break


    def inference(self, model, condition, token_lim, strategies, params):
        model.eval()

        quavers=condition.quavers   #0,4,3
        free_gen=condition.free_gen
        word_nums=condition.word_nums   #(bar_num)
        word_embeddings=condition.word_embeddings   #(128,512)

        # initialize mem
        mems = tuple()
        song_init_time = time.time()

        #初始化
        words = [[187]]     #以一个起始标记作为开始

        if not quavers==0:
            ttime=100
        if not free_gen:
            bar_idx=0

        end_token=False
        song_end=False
        if free_gen:
            max_num = word_nums[0]
            temp_num=0

        # generate
        while len(words[0]) < token_lim and song_end==False:
            # prepare input
            temp_x = np.zeros((1, 1))
            temp_x[0][0] = words[0][-1]

            temp_x = torch.from_numpy(temp_x).long().to(self.device)
            condition = torch.from_numpy(word_embeddings).to(self.device).float()
            
            _logits, mems = model.generate(temp_x, condition, *mems)
            logits = _logits.cpu().squeeze().detach().numpy()
            if words[0][-1]==188:
                end_token=True
            if not quavers==0:  #小节长度受控情况下
                if not free_gen:    #歌词字数受控情况下
                    if ttime>=4*quavers:    #生成小节标记
                        word=0
                        ttime=0
                    elif words[0][-1]==0 or words[0][-1]==188:  #生成字数标记和结束标记
                        if bar_idx>=len(word_nums)-1 and end_token==False:
                            word=188
                        else:
                            word=word_nums[bar_idx]+170
                            bar_idx+=1
                    else:
                        if 'temperature' in strategies:
                            probs = self.temperature(logits=logits, temperature=params['t'])
                        else:
                            probs = self.temperature(logits=logits, temperature=1.)
                        # 使生成小节、字数、开始、结束的概率为0
                        for prob in range(170,189):
                            probs[prob] = 0
                        probs[0]=0
                        word = self.nucleus(probs=probs, p=params['p'])

                else:
                    if ttime >= 4 * quavers:  #生成小节标记
                        word = 0
                        ttime = 0
                    else:
                        if 'temperature' in strategies:
                            probs = self.temperature(logits=logits, temperature=params['t'])
                        else:
                            probs = self.temperature(logits=logits, temperature=1.)

                        probs[0] = 0
                        probs[187] = 0
                        word = self.nucleus(probs=probs, p=params['p'])

                        # 计算总字数，如果超过则不再生成新的旋律
                        if word>=170 and word<=186:
                            temp_num+=(word-170)
                            if temp_num>max_num:
                                word=max(word-(temp_num-max_num),170)

                # 记录步长
                if word >= 154 and word <= 169:
                    ttime += (word - 153)
                if ttime > 4 * quavers:
                    word -= (ttime - 4 * quavers)

            else:
                if not free_gen:    #歌词字数受控情况下
                    if words[0][-1]==0 or words[0][-1]==188:  #生成字数标记和结束标记
                        if bar_idx>=len(word_nums)-1 and end_token==False:
                            word=188
                        else:
                            word=word_nums[bar_idx]+170
                            bar_idx+=1
                    else:
                        if 'temperature' in strategies:
                            probs = self.temperature(logits=logits, temperature=params['t'])
                        else:
                            probs = self.temperature(logits=logits, temperature=1.)

                        for prob in range(170,189):
                            probs[prob] = 0
                        word = self.nucleus(probs=probs, p=params['p'])

                else:
                    if 'temperature' in strategies:
                        probs = self.temperature(logits=logits, temperature=params['t'])
                    else:
                        probs = self.temperature(logits=logits, temperature=1.)

                    probs[187] = 0
                    word = self.nucleus(probs=probs, p=params['p'])

                    if word >= 170 and word <= 186:
                        temp_num += (word - 170)
                        if temp_num > max_num:
                            word = max(word - (temp_num - max_num), 170)


            if end_token==True and word==0:
                song_end=True
            words[0].append(word)
            print(len(words[0]), [word])


        song_total_time = time.time() - song_init_time
        print('Total words generated: ', len(words[0]))
        return song_total_time, len(words[0]), words[0]

    ########################################
    # search strategy: temperature
    ########################################
    def temperature(self, logits, temperature): # t=1.2
        probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
        return probs

    ########################################
    # search strategy: topk
    ########################################
    def topk(self, probs, k):
        sorted_index = np.argsort(probs)[::-1]
        candi_index = sorted_index[:k]
        candi_probs = [probs[i] for i in candi_index]
        candi_probs /= sum(candi_probs)
        word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
        return word

    ########################################
    # search strategy: nucleus
    ########################################
    def nucleus(self, probs, p):    # p=0.9
        probs /= sum(probs)
        sorted_probs = np.sort(probs)[::-1]
        sorted_index = np.argsort(probs)[::-1]
        cusum_sorted_probs = np.cumsum(sorted_probs)
        after_threshold = cusum_sorted_probs > p
        if sum(after_threshold) > 0:
            last_index = np.where(after_threshold)[0][0] + 1
            candi_index = sorted_index[:last_index]
        else:
            candi_index = sorted_index[:3] # just assign a value
        candi_probs = [probs[i] for i in candi_index]
        candi_probs /= sum(candi_probs)
        word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
        return word