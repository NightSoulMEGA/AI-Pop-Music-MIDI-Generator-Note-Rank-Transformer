from model_inference import inference

if __name__ == '__main__':
    #每小节节拍数，0表示不限制，建议为3或4
    quavers = 0
    #是否自由生成旋律，True则歌词中的分隔符将失效
    free_gen = False
    #每分钟节拍数，默认120
    bpm = 120
    #歌曲标题，仅包含中文与英文文本
    topics = []
    topics.append('元夕')
    #歌词，仅包含中文与英文文本，使用全角或半角逗号或句号划分句子，使用分隔符“|”划分小节
    #例如，你可以用'||||'生成四个小节的前奏
    lyrics = []
    lyrics.append('||||东风夜放花千树。|更吹落，星如雨。|宝马雕车香满|路。|'
                  '凤箫声动，|玉壶光转，|一夜鱼龙|舞。|'
                  '蛾儿雪柳黄金缕。|笑语盈盈暗香去。|众里寻他千百|度。|'
                  '蓦然回首，|那人却在，|灯火阑珊|处。|')

    _input = []
    for i in range(len(lyrics)):
        _input.append((topics[i]+','+'|'+lyrics[i]+'|').split('|'))

    path_list, words_list = inference(quavers=quavers, free_gen=free_gen, inputs=_input, bpm=bpm)
    print(path_list[0])


















