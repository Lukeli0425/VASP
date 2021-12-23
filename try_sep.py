import os,sys
sys.path.insert(0, "./VoiceSplit/")
sys.path.insert(0, "./GE2E_Speaker_Encoder/")
from pathlib import Path
from encoder import inference as encoder
import numpy as np
import torch,librosa,resemblyzer
from VoiceSplit.utils.generic_utils import load_config_from_str
from VoiceSplit.utils.audio_processor import WrapperAudioProcessor as AudioProcessor 
import soundfile as sf
from VoiceSplit.models.voicefilter.model import VoiceFilter
from VoiceSplit.models.voicesplit.model import VoiceSplit
import voice_demo

def predict(model, ap, mixed_wav,  embedding, outpath='predict.wav'):
    # embedding_0, mixed_spec, mixed_phase, mixed_wav = normalise_and_extract_features(ap, mixed_path)
    # use the model
    norm_factor = np.max(np.abs(mixed_wav)) * 1.1
    mixed_wav = mixed_wav/norm_factor
    mixed_wav, _ = librosa.effects.trim(mixed_wav, top_db=20)
    mixed_spec, mixed_phase = ap.get_spec_from_audio(mixed_wav, return_phase=True)
    mixed_spec = torch.from_numpy(mixed_spec).float()

    # append 1 dimension on mixed, its need because the model spected batch
    mixed_spec = mixed_spec.unsqueeze(0)
    embedding = embedding.unsqueeze(0)

    mask = model(mixed_spec, embedding) # 滤波系数
    output = mixed_spec * mask
    output = output * mask
    print(mixed_spec.shape)
    # inverse spectogram to wav
    est_mag = output[0].cpu().detach().numpy()
    mixed_spec = mixed_spec[0].cpu().detach().numpy()
    # use phase from mixed wav for reconstruct the wave
    est_wav = ap.inv_spectrogram(est_mag, phase=mixed_phase)

    sf.write(outpath, est_wav, 44100)

    return est_wav

if __name__ == "__main__":
    print("Preparing the encoder, the synthesizer and the vocoder...")
    encoder.load_model(Path('./pretrained/encoder/saved_models/pretrained.pt'))
    print("Testing your configuration with small inputs.")
    wav = np.zeros(encoder.sampling_rate)    
    embed = encoder.embed_utterance(wav)

    # Paths
    checkpoint_path = 'best_checkpoint.pt' # load checkpoint 
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    c = load_config_from_str(checkpoint['config_str'])
    ap = AudioProcessor(c.audio) # create AudioProcessor for model
    model = VoiceFilter(c)
    model.load_state_dict(checkpoint['model'])

    known_voices, embed_average = voice_demo.read_voices("./train/")
    
    # mixed_wav = resemblyzer.preprocess_wav(mixed_path)
    # target_path = './test_offline/task3_estimate/001_middle.wav'
    # outpath='./predict6.wav'
    mixed_path = './test_offline/task3/combine001.mp4'
    mixed_wav = ap.load_wav(mixed_path)
    for i in [6,7,8]:
        outpath = './predict' + str(i) + '.wav'
        est_wav = predict(model, ap, mixed_wav, embed_average[i-1], outpath)