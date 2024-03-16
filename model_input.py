import time
from translator import Translate
import numpy as np
import pronouncing

MAX_CONDITION_LEN = 128

class GenerateInput():
    def __init__(self,quavers=0,free_gen=True,inputs=[]):

        self.quavers = quavers  # 每小节拍数，0或4或3，0表示无限制
        self.free_gen = free_gen   #是否参考每小节字数，False将严格按照word_num记录的每小节字数，True将按照字数总和设置
        self.word_nums=[]   #(小节数)
        self.word_embeddings=[] #(128, 512)

        if self.free_gen:
            self.word_nums=[sum(self.nums(inputs))]
        else:
            self.word_nums=[min(item, 16) for item in self.nums(inputs)]

        self.word_embeddings=self.tokenizer(self.translate(self.split(inputs)))


    def nums(self, lyriclist):
        #计算每小节字数（英文单词计算音节数）
        nums=[]
        for lyric in lyriclist[1:]:
            num=0
            en=''
            for k in lyric:
                if k == ',' or k == '.' or k == '，' or k == '。':
                    if not en=='':
                        try:
                            pronunciation_list = pronouncing.phones_for_word(en)
                            num += pronouncing.syllable_count(pronunciation_list[0])
                        except:
                            num+=1
                        en=''
                    continue
                elif (k >= 'a' and k <= 'z') or (k >= 'A' and k <= 'Z'):
                    en+=k
                elif k==' ':
                    if not en == '':
                        try:
                            pronunciation_list = pronouncing.phones_for_word(en)
                            num += pronouncing.syllable_count(pronunciation_list[0])
                        except:
                            num += 1
                        en = ''
                else:
                    if not en == '':
                        try:
                            pronunciation_list = pronouncing.phones_for_word(en)
                            num += pronouncing.syllable_count(pronunciation_list[0])
                        except:
                            num += 1
                        en = ''
                    num+=1
            if not en == '':
                try:
                    pronunciation_list = pronouncing.phones_for_word(en)
                    num += pronouncing.syllable_count(pronunciation_list[0])
                except:
                    num += 1
                en = ''
            nums.append(num)
        return nums

    def split(self, lyriclist):
        #按照标点符号（全角、半角的逗号、句号）划分用于语义计算的句子，最大长度128
        lyrics=''
        for lyric in lyriclist:
            if not lyric=='':
                lyrics+=lyric
        lyrics2=lyrics.replace('，', '.')
        lyrics3=lyrics2.replace('。', '.')
        lyrics4=lyrics3.replace(',', '.')
        new_lyrics=lyrics4.split('.')
        if len(new_lyrics)>MAX_CONDITION_LEN:
            new_lyrics=new_lyrics[:MAX_CONDITION_LEN]
        return new_lyrics

    def translate(self, lyriclist):
        #trans=[]
        for l in range(len(lyriclist)):
            if l >= len(lyriclist):
                break
            while lyriclist[l] == ' ' or lyriclist[l] == '':
                lyriclist.pop(l)
                if l >= len(lyriclist):
                    break
            # if l >= len(lyriclist):
            #     break
            #trans.append(Translate(lyriclist[l]))
            #time.sleep(0.1)
        trans=Translate(lyriclist)

        return trans

    def tokenizer(self, lyriclist):
        import laion_clap as clap

        model = clap.CLAP_Module(enable_fusion=False)
        model.load_ckpt(ckpt='630k-audioset-fusion-best.pt')
        embeds = model.get_text_embedding(lyriclist)
        # print(embeds.shape)  # (length,512)
        while len(embeds)<MAX_CONDITION_LEN:
            embeds = np.concatenate([embeds, embeds])
        embeds = embeds[:MAX_CONDITION_LEN]
        # print(embeds.shape)   #(128,512)
        return embeds


class EvalInput():
    def __init__(self, quavers, free_gen=True, word_embeddings=None):

        self.quavers = 0
        self.free_gen = free_gen
        self.word_nums = []
        self.word_embeddings = word_embeddings

        q=0
        qlist=[]
        for x in quavers[0]:
            if x==0:
                qlist.append(q//4)
                q=0
            if x>=154 and x<=169:
                q+=x-153
        if len(qlist)==0:
            self.quavers=0
        else:
            self.quavers=max(qlist)





