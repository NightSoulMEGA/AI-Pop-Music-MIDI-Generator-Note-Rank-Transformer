自制的某种基于Transformer-XL的AI流行音乐midi生成器。
  
可以接受英文或中文歌词作为输入，使用Helsinki-NLP进行翻译，使用BART进行嵌入，最后生成具有特定结构的midi音乐。

生成的midi拥有两个轨道，分别是歌唱主旋律和钢琴伴奏。

使用一种经过重新标记的POP909数据集进行的训练。相关数据集和论文或许会在以后发布。

——————————————————————————————————————————————————

Some sort of self-made AI pop music midi generator based on Transformer-XL.

It is able to accept English or Chinese lyric as an input, using Helsinki-NLP to translate and BART to tokenize, and then generate midi pop music with the specific structrue.

The generated midi has two tracks, the main singing melody and the piano accompaniment.

Trained on a relabled version of POP909. The relevant datasets and papers may be published soon.
