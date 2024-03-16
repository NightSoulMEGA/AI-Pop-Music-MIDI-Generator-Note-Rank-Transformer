import mido
from chordDic import Dic1
import numpy as np
import random


def gen_midi(words, path_midi, bpm):
    #生成midi文件
    melody_notes = []   #[[音高（小节为-1），时间步]]
    chords = []         #[[[音符音高（小节为-1），时间步，音量]]]
    chord_notes = []    #[[音符音高（小节为-1），时间步，音量]]

    step=0
    root=0
    name=Dic1[0]['name']
    keyset=Dic1[0]['keyset']

    #解码输出序列（一个16分音符120步）
    for word in words:
        if word==0:
            melody_notes.append([-1, step*120])
            chords.append([[-1,step*120,0]])

        if word>=1 and word<=12:
            root=word-1

        if word >= 13 and word <= 41:
            name = Dic1[word-12]['name']
            keyset = Dic1[word-12]['keyset']
            chords.append([[-1,step*120,0]])

        if word>=42 and word<=80:
            series=word-42
            k=0
            while series>0:
                k+=1
                if keyset[k%12]==1:
                    series-=1
            try:
                chords[-1].append([root+k,step*120,64])
            except:
                pass

        if word>=81 and word<=96:
            velocity=random.randint(0,7)
            velocity+=(word-81)*8
            try:
                chords[-1][-1][-1]=velocity
            except:
                pass

        if word>=97 and word<=153:
            melody_notes.append([word-55, step*120])

        if word>=154 and word<=169:
            step+=word-153


    #分析和弦生成位置
    root_core = 45
    last_root = root_core
    for chord in chords:
        chord_notes.append(chord[0])
        if len(chord)==1:
            continue
        pitchlist=[p[0] for p in chord[1:]]
        p_min=min(pitchlist)
        p_max=max(pitchlist)
        add=0
        while True:
            if p_max+add>=128:
                break
            else:
                current_p = abs(p_min+add - root_core) + abs(p_min+add - last_root)
                next_p=abs(p_min+add+12 - root_core) + abs(p_min+add+12 - last_root)
                if next_p<current_p:
                    add+=12
                else:
                    break
        last_root = p_min+add
        for note in chord:
            if note[0]+add>=128:
                continue
            chord_notes.append([note[0]+add,note[1],note[2]])

    #生成MIDI
    mid = mido.MidiFile()
    track0 = mido.MidiTrack()
    track1 = mido.MidiTrack()
    mid.tracks.append(track0)
    mid.tracks.append(track1)

    track0.append(mido.MetaMessage('set_tempo', tempo = mido.bpm2tempo(bpm=bpm), time=0))

    playing_note=[0]*128
    last=0
    for msg in melody_notes:
        if msg[0]==-1:
            for n in range(128):
                if playing_note[n]==1:
                    track0.append(mido.Message('note_off', note=int(n), velocity=64, time=int(msg[1]-last)))
                    last=msg[1]
                    playing_note[n]=0
        else:
            for n in range(128):
                if playing_note[n] == 1:
                    track0.append(mido.Message('note_off', note=int(n), velocity=64, time=int(msg[1] - last)))
                    last = msg[1]
                    playing_note[n] = 0
            track0.append(mido.Message('note_on', note=int(msg[0]), velocity=100, time=int(msg[1]-last)))
            last = msg[1]
            playing_note[msg[0]]=1

    playing_note = [0] * 128
    last = 0
    for msg in chord_notes:
        #print(msg)
        if msg[0] == -1:
            for n in range(128):
                if playing_note[n] == 1:
                    track1.append(mido.Message('note_off', note=int(n), velocity=64, time=int(msg[1]-last)))
                    last = msg[1]
                    playing_note[n] = 0
        else:
            if playing_note[msg[0]] == 1:
                track1.append(mido.Message('note_off', note=int(msg[0]), velocity=64, time=int(msg[1]-last)))
                last = msg[1]
                playing_note[msg[0]] = 0
            track1.append(mido.Message('note_on', note=int(msg[0]), velocity=msg[2], time=int(msg[1]-last)))
            last = msg[1]
            playing_note[msg[0]] = 1

    mid.save(path_midi)

if __name__ == '__main__':
    pass