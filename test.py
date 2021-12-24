import os
import numpy as np
import soundfile as sf
import json
import utils
import face_demo,voice_demo
import face_recognition,torch,torchaudio,librosa
from resemblyzer import preprocess_wav, VoiceEncoder
from speechbrain.pretrained import SepformerSeparation as separator

def test_task1(video_path):
    """测试1,使用face_recognition"""
    result_dict = {}
    known_faces = face_demo.read_faces('./train/')

    for file_name in os.listdir(video_path):
        ## 读取MP4文件中的视频,可以用任意其他的读写库
        video_frames,video_fps = utils.read_video(os.path.join(video_path,file_name))
        print('video_frames have shape of:',video_frames.shape, 'and fps of:',video_fps)
        l,h,w,_ = video_frames.shape
        ## 做一些处理
        prediction = np.zeros(20)
        prediction_single = np.zeros(20)
        step = int(l/30) # 采样步长
        for i in range(0,l,step):
            face = video_frames[i]
            result_dict[file_name] = '0'
            try:
                face_encoding = face_recognition.face_encodings(face)[0]
            except IndexError:
                continue
            for k in range(0,20,1):
                results = face_recognition.compare_faces(known_faces[k], face_encoding)
                prediction_single[k] = sum(results)/len(results)
        prediction += prediction_single
        ## 返回一个ID
        answer = np.argmax(prediction) + 1
        result_dict[file_name] = utils.ID_dict[answer]
        print(file_name + " " + str(answer))
        
    return result_dict

def test_task2(wav_path):
    """测试2,使用resemblyzer"""
    result_dict = {}
    known_voices, embed_average = voice_demo.read_voices("./train/")
    encoder = VoiceEncoder()
    
    for file_name in os.listdir(wav_path):
        ## 读取WAV文件中的音频,可以用任意其他的读写库
        # audio_trace = utils.read_audio(os.path.join(wav_path,file_name),sr=44100)
        wav_test = preprocess_wav(os.path.join(wav_path,file_name))
        embed_test = encoder.embed_utterance(wav_test)
        ## 做一些处理
        # print('audio_trace have shape of:',audio_trace.shape,'and sampling rate of: 44100')
        prediction = np.zeros(20)
        for i in range(0,20,1):
            prediction[i] = np.sqrt(np.sum((embed_test - embed_average[i])**2))
        ## 返回一个ID
        answer = np.argmin(prediction) + 1
        result_dict[file_name] = utils.ID_dict[answer]
        print(file_name + " " + str(answer));print(prediction)

    return result_dict

def test_task3(video_path,result_path):
    """测试3,使用speechbrain（https://huggingface.co/speechbrain/sepformer-wsj03mix）"""
    sep_model = separator.from_hparams(source="speechbrain/sepformer-wsj03mix", savedir='pretrained_models/sepformer-wsj03mix')
    known_voices, embed_average = voice_demo.read_voices("./train/",sr=44100)
    known_faces = face_demo.read_faces('./train/')
    encoder = VoiceEncoder()

    if os.path.isdir(result_path):
        print('warning: using existed path as result_path')
    else:
        os.mkdir(result_path)
    video_path = "./test_offline/task3/"
    for file_name in os.listdir(video_path):
        if not file_name[-4:] == '.mp4':
            continue
        ## 读MP4中的图像和音频数据，例如：
        idx = file_name[-7:-4]  # 提取出序号：001, 002, 003.....
        video_frames,video_fps= utils.read_video(os.path.join(video_path, file_name))
        # mixed_wav = utils.read_audio(os.path.join(video_path,file_name),sr=8000)
        mixed_wav,_ = librosa.load(os.path.join(video_path,file_name),sr=8000)
        sf.write(os.path.join(video_path,file_name[0:-4]+'.wav'),mixed_wav,8000)
        ## 做一些处理
        # print('video_frames have shape of:',video_frames.shape, 'and fps of:',video_fps)
        # print('audio_trace have shape of:',audio_trace.shape,'and sampling rate of: 44100')
        l,h,w,_ = video_frames.shape
        videos = [video_frames[:,:,0:224,], video_frames[:,:,224:448,],video_frames[:,:,448:672,]]
        ID = []
        for n in range(0,3,1):
            prediction = np.zeros(20)
            prediction_single = np.zeros(20)
            step = int(l/30) # 采样步长
            for i in range(0,l,step):
                face = videos[n][i]
                try:
                    face_encoding = face_recognition.face_encodings(face)[0]
                except IndexError:
                    continue
                for k in range(0,20,1):
                    results = face_recognition.compare_faces(known_faces[k], face_encoding)
                    prediction_single[k] = sum(results)/len(results)
            prediction += prediction_single
            ## 返回一个ID
            ID.append(np.argmax(prediction) + 1)
        print(file_name + " " + str(ID))

        est_sources = sep_model.separate_file(path=os.path.join(video_path,file_name[0:-4]+'.wav'),savedir='./temp/') 
        audio_out = [est_sources[:,:,0],est_sources[:,:,1],est_sources[:,:,2]]
        torchaudio.save(os.path.join(result_path,idx+'_left.wav'), est_sources[:,:,0].detach().cpu(), 8000)
        torchaudio.save(os.path.join(result_path,idx+'_middle.wav'), est_sources[:,:,1].detach().cpu(), 8000)
        torchaudio.save(os.path.join(result_path,idx+'_right.wav'), est_sources[:,:,2].detach().cpu(), 8000)
        wav0 = preprocess_wav(os.path.join(result_path,idx+'_left.wav'))
        wav1 = preprocess_wav(os.path.join(result_path,idx+'_middle.wav'))
        wav2 = preprocess_wav(os.path.join(result_path,idx+'_right.wav'))
        audio_temp = [wav0,wav1,wav2]
        
        prediction = np.zeros([3,3]) # 音频d-vector距离矩阵
        ans = [-1,-1,-1]
        for j in [0,1,2]:
            for i in [0,1,2]:
                embed_test = encoder.embed_utterance(np.array(audio_temp[j]))
                prediction[j,i] = np.sqrt(np.sum((embed_test - embed_average[ID[i]-1])**2))
        # print(prediction)
        for i in [0,1,2]: # 利用距离矩阵预测音视频的对应关系
            min = np.argmin(prediction)
            audio_min = int(min/3)
            id_min = min - 3 * audio_min
            ans[id_min] = audio_min
            prediction[:,id_min] = 10
            prediction[audio_min,:] = 10
        
        # print(ans)
        torchaudio.save(os.path.join(result_path,idx+'_left.wav'), audio_out[ans[0]].detach().cpu(), 8000)
        torchaudio.save(os.path.join(result_path,idx+'_middle.wav'), audio_out[ans[1]].detach().cpu(), 8000)
        torchaudio.save(os.path.join(result_path,idx+'_right.wav'), audio_out[ans[2]].detach().cpu(), 8000)


if __name__=='__main__':
    # testing task1
    # with open('./test_offline/task1_gt.json','r') as f:
    #     task1_gt = json.load(f)
    # task1_pred = test_task1('./test_offline/task1')
    # task1_acc = utils.calc_accuracy(task1_gt,task1_pred)
    # print('accuracy for task1 is:',task1_acc)   

    # ## testing task2
    # with open('./test_offline/task2_gt.json','r') as f:
    #     task2_gt = json.load(f)
    # task2_pred = test_task2('./test_offline/task2')
    # task2_acc = utils.calc_accuracy(task2_gt,task2_pred)
    # print('accuracy for task2 is:',task2_acc)   

    # # ## testing task3
    test_task3('./test_offline/task3','./test_offline/task3_estimate')
    task3_SISDR_blind = utils.calc_SISDR('./test_offline/task3_gt','./test_offline/task3_estimate',permutaion=True)  # 盲分离
    print('strength-averaged SISDR_blind for task3 is:',task3_SISDR_blind)
    task3_SISDR_match = utils.calc_SISDR('./test_offline/task3_gt','./test_offline/task3_estimate',permutaion=False) # 定位分离
    print('strength-averaged SISDR_match for task3 is: ',task3_SISDR_match)

