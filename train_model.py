from resemblyzer import preprocess_wav, VoiceEncoder
import os,librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Multiply, ZeroPadding2D, concatenate, Conv2D, Input, Dense, Reshape, BatchNormalization, Activation, LSTM, Lambda, Bidirectional
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint
from Audio import Audio,HyperParams
import soundfile as sf
import voice_demo

def get_model():
    hyper_params = HyperParams() # model parameters
    T_dim = int(hyper_params.length/hyper_params.hop_length)
    emb_dim = hyper_params.embedder_emb_dim
    num_freq =  hyper_params.num_freq
    lstm_dim =  hyper_params.model_lstm_dim
    fc1_dim = hyper_params.model_fc1_dim
    fc2_dim = hyper_params.model_fc2_dim # num_freq

    dvec_inp = Input(shape=(emb_dim),name="dvec")
    input_spec = Input(shape=(T_dim,num_freq),name="input_spec")
    x = Reshape((T_dim,num_freq,1))(input_spec)
    
    #cnn1
    x = ZeroPadding2D(((0,0), (3,3)))(x)
    x = Conv2D(filters=64, kernel_size=[1,7], dilation_rate=[1, 1])(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #cnn2
    x = ZeroPadding2D(((3,3), (0,0)))(x)
    x = Conv2D(filters=64, kernel_size=[7,1], dilation_rate=[1, 1])(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #cnn3
    x = ZeroPadding2D(((2,2), (2,2)))(x)
    x = Conv2D(filters=64, kernel_size=[5,5], dilation_rate=[1, 1])(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #cnn4
    x = ZeroPadding2D(((4,4), (2,2)))(x)
    x = Conv2D(filters=64, kernel_size=[5,5], dilation_rate=[2, 1])(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #cnn5
    x = ZeroPadding2D(((8,8), (2,2)))(x)
    x = Conv2D(filters=64, kernel_size=[5,5], dilation_rate=[4, 1])(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #cnn6
    x = ZeroPadding2D(((16,16), (2,2)))(x)
    x = Conv2D(filters=64, kernel_size=[5,5], dilation_rate=[8, 1])(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #cnn7
    x = ZeroPadding2D(((32,32), (2,2)))(x)
    x = Conv2D(filters=64, kernel_size=[5,5], dilation_rate=[16, 1])(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #cnn8
    x = Conv2D(filters=8, kernel_size=[1,1], dilation_rate=[1, 1])(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Reshape((x.shape[1],x.shape[2]*x.shape[3]))(x) #else use -1 as last arg
    #x = tf.reshape(x, [x.shape[0],x.shape[1],-1])
    
    dvec = Lambda(lambda a : tf.expand_dims(a,1))(dvec_inp)
    dvec = Lambda(lambda a : tf.repeat(a,repeats =x.shape[1],axis =1))(dvec)
    #dvec= tf.expand_dims(dvec_inp,1)
    #dvec= tf.repeat(dvec,repeats =x.shape[1],axis =1)
    
    x = concatenate([x,dvec],-1)
    #x= tf.concat([x,dvec],-1)
    
    #lstm
    x = Bidirectional(LSTM(lstm_dim,return_sequences=True))(x)
    
    #fc1
    x = Dense(fc1_dim,activation ="relu")(x)
    #fc2
    mask = Dense(fc2_dim,activation ="sigmoid",name="mask")(x) #soft mask
    
    #element-wise
    output = Multiply()([input_spec,mask])

    model = Model(inputs=[input_spec,dvec_inp], outputs=output)
    model.summary()
    return model

def generate_data(data_path, audio, embed_average):
    ids=np.random.randint(1,21,size=3)
    embeddings = []
    wavs = []
    spec = []
    # encoder = VoiceEncoder()
    length = 80000
    for i in [0,1,2]:
        path = data_path + "ID" + str(ids[i]) + "/"
        files = os.listdir(path)
        np.random.shuffle(files)
        for file in files:
            wav,_ = librosa.load(path+file, sr=16000)
            wav = (0.8 * np.random.rand() + 0.2) * wav
            if len(wav) >= length:
                break
        wavs.append(wav[0:length - 1])
        # embeddings.append(encoder.embed_utterance(wavs[i]))
        embeddings.append(embed_average[ids[i]-1])
        wav_spec,_ = audio.wave2spec(wavs[i])
        spec.append(wav_spec)
    mixed_wav = wavs[0]
    mixed_wav += wavs[1]
    mixed_wav += wavs[2]
    mixed_spec,mixed_phase = audio.wave2spec(mixed_wav)
    return embeddings, spec, mixed_spec, mixed_phase


def create_dataset(data_path, audio, train_num=2):
    """生成数据集"""
    print("\nStart Creating Dataset\n")
    _,embed_average = voice_demo.read_voices("./train/")
    input_spec = []
    input_embeddings = []
    output_spec = []
    for n in range(0,train_num,1):
        embeddings, spec, mixed_spec, mixed_phase = generate_data(data_path, audio, embed_average)
        # for i in range(0,3,1):
        input_embeddings.append(embeddings[0])
        input_spec.append(mixed_spec)
        output_spec.append(spec[0])
    input_spec = tf.convert_to_tensor(input_spec)
    input_embeddings = tf.convert_to_tensor(input_embeddings)
    output_spec = tf.convert_to_tensor(output_spec)
    print("\nCreate Dataset Done\n")
    return input_spec,input_embeddings,output_spec

# class Dataset(Sequence):
#     def __init__(self, train_num, data_path):
#         self.train_num = train_num
#         self.datapath = data_path
#         self.input_spec,self.input_embedding,self.output_spec = create_dataset(data_path)
#     def __len__(self):
#         return self.datapath
#     def __getitem__(self, idx):
#         return ({'input_spec':self.input_spec, 'dvec': self.input_embedding}, self.output_spec)

def train_model(epochs, size, weights_path):
    hyper_params = HyperParams()
    audio = Audio(hyper_params)
    encoder = VoiceEncoder()

    # 保存训练好的weight的路径
    if not os.path.isdir(weights_path):
        os.mkdir(weights_path)
    model_checkpoint_callback = ModelCheckpoint(os.path.join(weights_path,'weights_epoch{epoch:04d}.h5'),save_weights_only=True,save_freq='epoch')

    input_spec, input_embedding, output_spec = create_dataset('./train/', audio, size)
    model = get_model()
    model.compile(optimizer='adam', loss='mse')

    model.load_weights(os.path.join(weights_path,'weights_epoch0001.h5'))

    history = model.fit({'input_spec':input_spec, 'dvec':input_embedding}, output_spec, 
                        epochs=epochs, shuffle=True,
                        callbacks=[model_checkpoint_callback])
    
    return model

if __name__ == "__main__":
    with tf.device('/gpu:0'):
        # 训练参数
        epochs = 2 # 估计loss在较小的epoch时就已经稳定了，为了节省时间就少训练几个epoch
        size = 400 # 在这里改训练数据量（先从500开始，之后1000 2000 5000 10000 20000）
        weights_path = './model/'

        train_model(epochs, size, weights_path)

        # model = get_model()
        # model.compile(optimizer='adam', loss='mse')
        # model.load_weights(os.path.join(weights_path,'weights_epoch%04d.h5'%epochs))

        # hyper_params = HyperParams()
        # audio = Audio(hyper_params)
        # encoder = VoiceEncoder()

        # wav,_ = librosa.load('./test_offline/task3/combine001.mp4', sr=16000)
        # input_spec,mixed_phase = audio.wave2spec(wav) 
        # input_spec = tf.convert_to_tensor([input_spec])
        # wav_7,_ = librosa.load('./test_offline/task3_gt/001_left.wav', sr=16000)
        # # wav_7 = preprocess_wav('./test_offline/task3_gt/001_left.wav', 16000)
        # embed = tf.convert_to_tensor([encoder.embed_utterance(wav_7)])

        # print(input_spec.shape)
        # print(embed.shape)
        # out_spec = model.predict(x={'input_spec':input_spec, 'dvec':embed},verbose=1)
        # out_wav = audio.spec2wave(out_spec[0],mixed_phase)

        # sf.write('./predict7.wav', out_wav, 16000)