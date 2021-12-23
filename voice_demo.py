import os
import librosa
import numpy as np
from resemblyzer import preprocess_wav, VoiceEncoder
import utils

def read_voices(data_path, sr=44100):
    """读入data_path中的视频，记录其中每个标签的人声信息用于后续判断"""
    known_voices = []
    embed_average = [] # 每个人的平均embed
    encoder = VoiceEncoder()
    for i in range(0,20,1):
        voices = []
        path = data_path + "ID" + str(i+1) + "/"
        files = os.listdir(path)
        flag = True
        for file in files:
            if not file[-4:] == '.mp4':
                continue
            # wav = preprocess_wav(path + file) # dtype = float32
            wav,_ = librosa.load(path + file,sr=sr)
            embed = encoder.embed_utterance(np.array(wav))
            if flag:
                average = embed
                flag = False
            else:
                average += embed
            voices.append(embed)
        known_voices.append(voices)
        average = average/len(voices)
        # average = average/np.linalg.norm(average, ord=2, axis=None, keepdims=False)
        embed_average.append(average)
        print("\nReading Voice done: ID " + str(i+1))
    return known_voices, embed_average


if __name__ == "__main__":
    # DEMO 02: we'll show how this similarity measure can be used to perform speaker diarization
    # (telling who is speaking when in a recording).

    ## Get reference audios
    wav_1 = preprocess_wav("./train/ID1/014.mp4")
    wav_2 = preprocess_wav("./train/ID2/015.mp4")
    wav_3 = preprocess_wav("./train/ID3/015.mp4")
    wav_test = preprocess_wav("./train/ID3/015.mp4")
    speaker_wavs = [wav_1,wav_2,wav_3]
    speaker_names = ["1","2","3"]

    ## Compare speaker embeds to the continuous embedding of the interview
    # Derive a continuous embedding of the interview. We put a rate of 16, meaning that an 
    # embedding is generated every 0.0625 seconds. It is good to have a higher rate for speaker 
    # diarization, but it is not so useful for when you only need a summary embedding of the 
    # entire utterance. A rate of 2 would have been enough, but 16 is nice for the sake of the 
    # demonstration. 
    # We'll exceptionally force to run this on CPU, because it uses a lot of RAM and most GPUs 
    # won't have enough. There's a speed drawback, but it remains reasonable.
    encoder = VoiceEncoder("cpu")
    print("Running the continuous embedding on cpu, this might take a while...")
    _, cont_embeds, wav_splits = encoder.embed_utterance(wav_test, return_partials=True, rate=4)

    # Get the continuous similarity for every speaker. It amounts to a dot product between the 
    # embedding of the speaker and the continuous embedding of the interview
    speaker_embeds = [encoder.embed_utterance(speaker_wav) for speaker_wav in speaker_wavs]
    similarity_dict = {name: cont_embeds @ speaker_embed for name, speaker_embed in 
                    zip(speaker_names, speaker_embeds)}
    print(cont_embeds.shape)
    print(encoder.embed_utterance(speaker_wavs[0]).shape)
    for key in similarity_dict:
        similarity = 0
        temp = 0
        for i in similarity_dict[key]:
            if i > 0:
                similarity += i
                temp += 1
        if temp > 0:
            similarity /= temp
        print(key +  " " + str(similarity))
        # print(key +  " " + str(similarity_dict[key].sum()))

    ## Run the interactive demo
    # interactive_diarization(similarity_dict, wav, wav_splits)

    fpath = "./train/ID1/014.mp4"
    wav = preprocess_wav(fpath)
    encoder = VoiceEncoder()
    _,embed,_ = encoder.embed_utterance(wav, return_partials=True, rate=4)
    np.set_printoptions(precision=3, suppress=True)
    value = embed @ embed
