import sys
sys.path.insert(0, "./VoiceSplit/")
sys.path.insert(0, "./GE2E-Speaker-Encoder/")

# Imports from GE2E
from encoder import inference as encoder
from encoder.params_model import model_embedding_size as speaker_embedding_size
from pathlib import Path


# Imports from VoiceSplit model
from utils.audio_processor import WrapperAudioProcessor as AudioProcessor 
from utils.generic_utils import load_config
import librosa
import os
import numpy as np
import torch
from glob import glob
from tqdm import tqdm
import torch
import soundfile as sf
from VoiceSplit.models.voicefilter.model import VoiceFilter
from VoiceSplit.models.voicesplit.model import VoiceSplit

from utils.generic_utils import load_config, load_config_from_str
from resemblyzer import preprocess_wav, VoiceEncoder

#Load and test GE2E model
print("Preparing the encoder, the synthesizer and the vocoder...")
encoder.load_model(Path('./pretrained/encoder/saved_models/pretrained.pt'))
print("Testing your configuration with small inputs.")
# Forward an audio waveform of zeroes that lasts 1 second. Notice how we can get the encoder's
# sampling rate, which may differ.
# If you're unfamiliar with digital audio, know that it is encoded as an array of floats 
# (or sometimes integers, but mostly floats in this projects) ranging from -1 to 1.
# The sampling rate is the number of values (samples) recorded per second, it is set to
# 44100 for the encoder. Creating an array of length <sampling_rate> will always correspond 
# to an audio of 1 second.
print("\tTesting the encoder...")

def get_embedding(encoder, ap, wave_file_path):
    preprocessed_wav = encoder.preprocess_wav(wave_file_path)
    file_embedding = encoder.embed_utterance(preprocessed_wav)
    return torch.from_numpy(file_embedding.reshape(-1))

wav = np.zeros(encoder.sampling_rate)    
embed = encoder.embed_utterance(wav)
print(embed.shape)


# Paths
checkpoint_path = 'best_checkpoint.pt'
# load checkpoint 
checkpoint = torch.load(checkpoint_path, map_location='cpu')
#load config from checkpoint
c = load_config_from_str(checkpoint['config_str'])

ap = AudioProcessor(c.audio) # create AudioProcessor for model
model_name = c.model_name
cuda = False

# load model
if(model_name == 'voicefilter'):
    print('inicializado com voicefilter')
    model = VoiceFilter(c)
elif(model_name == 'voicesplit'):
    model = VoiceSplit(c)
else:
    raise Exception(" The model '"+model_name+"' is not suported")

if c.train_config['optimizer'] == 'adam':
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=c.train_config['learning_rate'])
else:
    raise Exception("The %s  not is a optimizer supported" % c.train['optimizer'])


model.load_state_dict(checkpoint['model'])


optimizer.load_state_dict(checkpoint['optimizer'])
step = checkpoint['step']

print("load model form Step:", step)
# convert model from cuda
if cuda:
    model = model.cuda()

# utils for plot spectrogram
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np
import imageio
def fig2np(fig):
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

def plot_spectrogram_to_numpy(spectrogram):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect='auto', origin='lower',
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel('Frames')
    plt.ylabel('Channels')
    plt.tight_layout()

    fig.canvas.draw()
    data = fig2np(fig)
    plt.close()
    return data

def save_spec(path, spec):
    data = plot_spectrogram_to_numpy(spec)
    imageio.imwrite(path, data)

# utils for calculate SNR and SDR
# this code is adpated from https://github.com/JusperLee/Calculate-SNR-SDR/
import torch
from mir_eval.separation import bss_eval_sources
from itertools import permutations

def SI_SNR(_s, s, mix, zero_mean=True):
    '''
         Calculate the SNR indicator between the two audios. 
         The larger the value, the better the separation.
         input:
               _s: Generated audio
               s:  Ground Truth audio
         output:
               SNR value 
    '''
    length = _s.shape[0]
    _s = _s[:length]
    s =s[:length]
    mix = mix[:length]
    if zero_mean:
        _s = _s - torch.mean(_s)
        s = s - torch.mean(s)
        mix = mix - torch.mean(mix)
    s_target = sum(torch.mul(_s, s))*s/(torch.pow(torch.norm(s, p=2), 2)+1e-8)
    e_noise = _s - s_target
    # mix ---------------------------
    mix_target = sum(torch.mul(mix, s))*s/(torch.pow(torch.norm(s, p=2), 2)+1e-8)
    mix_noise = mix - mix_target 
    return 20*torch.log10(torch.norm(s_target, p=2)/(torch.norm(e_noise, p=2)+1e-8)) - 20*torch.log10(torch.norm(mix_target, p=2)/(torch.norm(mix_noise, p=2)+1e-8))


def permute_SI_SNR(_s_lists, s_lists, mix):
    '''
        Calculate all possible SNRs according to 
        the permutation combination and 
        then find the maximum value.
        input:
               _s_lists: Generated audio list
               s_lists: Ground truth audio list
        output:
               max of SI-SNR
    '''
    length = len(_s_lists)
    results = []
    per = []
    for p in permutations(range(length)):
        s_list = [s_lists[n] for n in p]
        result = sum([SI_SNR(_s, s, mix, zero_mean=True) for _s, s in zip(_s_lists, s_list)])/length
        results.append(result)
        per.append(p)
    return max(results), per[results.index(max(results))]


def SDR(est, egs, mix):
    '''
        calculate SDR
        est: Network generated audio
        egs: Ground Truth
    '''
    length = est.numpy().shape[0]
    sdr, _, _, _ = bss_eval_sources(egs.numpy()[:length], est.numpy()[:length])
    mix_sdr, _, _, _ = bss_eval_sources(egs.numpy()[:length], mix.numpy()[:length])
    return float(sdr-mix_sdr)


def permutation_sdr(est_list, egs_list, mix, per):
    n = len(est_list)
    result = sum([SDR(est_list[a], egs_list[b], mix)
                      for a, b in enumerate(per)])/n
    return result

# extract caracteristics
def normalise_and_extract_features(encoder, ap, mixed_path, target_path, target_path2, emb_ref_path):
    mixed_path_norm = mixed_path.replace('.wav','-norm.wav') 
    target_path_norm = target_path.replace('.wav','-norm.wav')
    target_path_norm2 = target_path2.replace('.wav','-norm.wav')
    emb_ref_path_norm = emb_ref_path.replace('.wav','-norm.wav')
    
    # normalise wavs
    # ! ffmpeg-normalize $mixed_path -ar 44100 -o $mixed_path_norm -f
    # ! ffmpeg-normalize  $target_path -ar 44100 -o $target_path_norm -f 
    # ! ffmpeg-normalize  $target_path2 -ar 44100 -o $target_path_norm2 -f 
    # ! ffmpeg-normalize  $emb_ref_path -ar 44100 -o $emb_ref_path_norm -f

    # load wavs
    target_wav = ap.load_wav(target_path_norm)
    target_wav2 = ap.load_wav(target_path_norm2)
    mixed_wav = ap.load_wav(mixed_path_norm)
    emb_wav = ap.load_wav(emb_ref_path_norm)
    
    # trim initial and end  wave file silence using librosa
    # target_wav, _ = librosa.effects.trim(target_wav, top_db=20)
    # mixed_wav, _ = librosa.effects.trim(mixed_wav, top_db=20)
    # emb_wav, _ = librosa.effects.trim(emb_wav, top_db=20)

    # normalise wavs
    norm_factor = np.max(np.abs(mixed_wav)) * 1.1
    mixed_wav = mixed_wav/norm_factor
    emb_wav = emb_wav/norm_factor
    target_wav = target_wav/norm_factor
    target_wav2 = target_wav2/norm_factor

    # # save embedding ref 
    # librosa.output.write_wav(emb_ref_path_norm, emb_wav, 44100)
    # # save this is necessary for demo
    # librosa.output.write_wav(mixed_path_norm, mixed_wav, 44100)
    # librosa.output.write_wav(target_path_norm, target_wav, 44100)
    # librosa.output.write_wav(target_path_norm2, target_wav2, 44100)
    # save embedding ref 
    sf.write(emb_ref_path_norm, emb_wav, 44100)
    # save this is necessary for demo
    sf.write(mixed_path_norm, mixed_wav, 44100)
    sf.write(target_path_norm, target_wav, 44100)
    sf.write(target_path_norm2, target_wav2, 44100)

    embedding = get_embedding(encoder, ap, emb_ref_path_norm)
    mixed_spec, mixed_phase = ap.get_spec_from_audio(mixed_wav, return_phase=True)
    return embedding, mixed_spec, mixed_phase, target_wav, target_wav2, mixed_wav, emb_wav

def predict(encoder, ap, mixed_path, target_path, target_path2, emb_ref_path, outpath='predict.wav', save_img=False):
    embedding, mixed_spec, mixed_phase, target_wav, target_wav2, mixed_wav, emb_wav = normalise_and_extract_features(encoder, ap, mixed_path, target_path, target_path2,  emb_ref_path)
    # use the model
    mixed_spec = torch.from_numpy(mixed_spec).float()

    # append 1 dimension on mixed, its need because the model spected batch
    mixed_spec = mixed_spec.unsqueeze(0)
    embedding = embedding.unsqueeze(0)

    if cuda:
        embedding = embedding.cuda()
        mixed_spec = mixed_spec.cuda()

    mask = model(mixed_spec, embedding)
    output = mixed_spec * mask

    # inverse spectogram to wav
    est_mag = output[0].cpu().detach().numpy()
    mixed_spec = mixed_spec[0].cpu().detach().numpy()
    # use phase from mixed wav for reconstruct the wave
    est_wav = ap.inv_spectrogram(est_mag, phase=mixed_phase)

    # librosa.output.write_wav(outpath, est_wav, 44100)
    sf.write(outpath, est_wav, 44100)

    if save_img:
        img_path = outpath.replace('predict', 'images').replace(' ', '').replace('.wav','-est.png')
        save_spec(img_path, est_mag)
        target_mag = ap.get_spec_from_audio(target_wav, return_phase=False)
        img_path = outpath.replace('predict', 'images').replace(' ', '').replace('.wav','-target.png')
        save_spec(img_path, target_mag)
        img_path = outpath.replace('predict', 'images').replace(' ', '').replace('.wav','-mixed.png')
        save_spec(img_path, mixed_spec)
        

    return est_wav, target_wav, target_wav2, mixed_wav, emb_wav


import pandas as pd
from IPython.display import Audio, display
from mir_eval.separation import bss_eval_sources
import numpy as np
# create output path
os.makedirs('VoiceSplit/datasets/LibriSpeech/audios_demo/2_speakers/predict/',exist_ok=True)
os.makedirs('VoiceSplit/datasets/LibriSpeech/audios_demo/2_speakers/images/',exist_ok=True)

test_csv = pd.read_csv('VoiceSplit/datasets/LibriSpeech/test_demo.csv', sep=',').values

sdrs_before = []
sdrs_after = []
snrs_before = []
snrs_after = []
for noise_utterance,emb_utterance, clean_utterance, clean_utterance2 in test_csv:
    noise_utterance = os.path.join('VoiceSplit',noise_utterance).replace(' ', '')
    emb_utterance = os.path.join('VoiceSplit',emb_utterance).replace(' ', '')
    clean_utterance = os.path.join('VoiceSplit',clean_utterance).replace(' ', '')
    clean_utterance2 = os.path.join('VoiceSplit',clean_utterance2).replace(' ', '')
    output_path = noise_utterance.replace('noisy', 'predict').replace(' ', '')
    est_wav, target_wav, target_wav2, mixed_wav, emb_wav = predict(encoder, ap, noise_utterance, clean_utterance, clean_utterance2, emb_utterance, outpath=output_path, save_img=True)

    len_est = len(est_wav)
    len_mixed = len(mixed_wav)
    if len_est > len_mixed:
        # mixed need is biggest 
        est_wav = est_wav[:len_mixed]
    else:
        # if mixed is biggest than estimation wav we need pad with zeros because is expected that this part is silence
        est_wav = np.pad(est_wav, (0, len(mixed_wav)-len(est_wav)), 'constant', constant_values=(0, 0))

    # get wav for second voice, its need for SDR calculation
    est_wav2 = mixed_wav-est_wav

    len_est = len(est_wav2)
    len_mixed = len(mixed_wav)
    if len_est > len_mixed:
        # mixed need is biggest
        est_wav2 = est_wav2[:len_mixed]
    else:
        # if mixed is biggest than estimation wav we need pad with zeros because is expected that this part is silence
        est_wav2 = np.pad(est_wav2, (0, len(mixed_wav)-len(est_wav2)), 'constant', constant_values=(0, 0))

    len_est = len(target_wav)
    len_mixed = len(mixed_wav)
    if len_est > len_mixed:
        # mixed need is biggest
        target_wav = target_wav[:len_mixed]
    else:
        # if mixed is biggest than estimation wav we need pad with zeros because is expected that this part is silence
        target_wav = np.pad(target_wav, (0, len(mixed_wav)-len(target_wav)), 'constant', constant_values=(0, 0))

    # get target_wav for second voice, its recomended because google dont provide clean_utterance2 in your demo i need get in LibreSpeech Dataset, but i dont know if they normalised this file..
    target_wav2 = mixed_wav - target_wav
    '''len_est = len(target_wav2)
    len_mixed = len(mixed_wav)
    if len_est > len_mixed:
        # mixed need is biggest
        target_wav2 = target_wav2[:len_mixed]
    else:
        # if mixed is biggest than estimation wav we need pad with zeros because is expected that this part is silence
        target_wav2 = np.pad(target_wav2, (0, len(mixed_wav)-len(target_wav2)), 'constant', constant_values=(0, 0))'''

    # calculate snr and sdr before model
    ests = [torch.from_numpy(mixed_wav), torch.from_numpy(mixed_wav)] # the same voices is mixed_wav
    egs = [torch.from_numpy(target_wav), torch.from_numpy(target_wav2)]
    mix = torch.from_numpy(mixed_wav)
    _snr, per = permute_SI_SNR(ests, egs, mix)
    _sdr = permutation_sdr(ests, egs, mix, per)
    snrs_before.append(_snr)
    sdrs_before.append(_sdr)

    # calculate snr and sdr after model
    ests = [torch.from_numpy(est_wav), torch.from_numpy(est_wav2)]
    egs = [torch.from_numpy(target_wav), torch.from_numpy(target_wav2)]
    mix = torch.from_numpy(mixed_wav)
    _snr, per = permute_SI_SNR(ests, egs, mix)
    _sdr = permutation_sdr(ests, egs, mix, per)
    snrs_after.append(_snr)
    sdrs_after.append(_sdr)

    # show in notebook results
    print('-'*100)
    print('-'*30,os.path.basename(noise_utterance),'-'*30)
    print("Input/Noise Audio")
    display(Audio(mixed_wav,rate=44100))
    print('Predicted Audio')
    display(Audio(est_wav,rate=44100))
    print('Target Audio')
    display(Audio(target_wav,rate=44100))
    print('Predicted2 Audio')
    display(Audio(est_wav2,rate=44100))
    print('Target2 Audio')
    display(Audio(target_wav2,rate=44100))
    print('-'*100)
    del target_wav, est_wav, mixed_wav


print('='*20,"Before Model",'='*20)
print('\nAverage SNRi: {:.5f}'.format(np.array(snrs_before).mean()))
print('Average SDRi: {:.5f}'.format(np.array(sdrs_before).mean()))

print('='*20,"After Model",'='*20)
print('\nAverage SNRi: {:.5f}'.format(np.array(snrs_after).mean()))
print('Average SDRi: {:.5f}'.format(np.array(sdrs_after).mean()))


# Apply VoiceFilter on clean audio (single speaker)
import pandas as pd
from IPython.display import Audio, display
from mir_eval.separation import bss_eval_sources
import numpy as np
# create output path
os.makedirs('VoiceSplit/datasets/LibriSpeech/audios_demo/single_speaker/predict/',exist_ok=True)
os.makedirs('VoiceSplit/datasets/LibriSpeech/audios_demo/single_speaker/images/',exist_ok=True)
test_csv = pd.read_csv('VoiceSplit/datasets/LibriSpeech/test_demo.csv', sep=',').values

sdrs_before = []
sdrs_after = []
snrs_before = []
snrs_after = []
for _ ,emb_utterance, clean_utterance, clean_utterance2 in test_csv:
    emb_utterance = os.path.join('VoiceSplit',emb_utterance).replace(' ', '')
    clean_utterance = os.path.join('VoiceSplit',clean_utterance).replace(' ', '')
    clean_utterance2 = os.path.join('VoiceSplit',clean_utterance2).replace(' ', '')
    output_path = clean_utterance.replace('/clean/', '/single_speaker/predict/').replace(' ', '')

    #  input = clean uterrance
    est_wav, target_wav, target_wav2, mixed_wav, emb_wav = predict(encoder, ap, clean_utterance, clean_utterance, clean_utterance2, emb_utterance, outpath=output_path, save_img=True)

    len_est = len(est_wav)
    len_mixed = len(mixed_wav)
    if len_est > len_mixed:
        # mixed need is biggest 
        est_wav = est_wav[:len_mixed]
    else:
        # if mixed is biggest than estimation wav we need pad with zeros because is expected that this part is silence
        est_wav = np.pad(est_wav, (0, len(mixed_wav)-len(est_wav)), 'constant', constant_values=(0, 0))

    # get wav for second voice, its need for SDR calculation
    est_wav2 = mixed_wav-est_wav

    len_est = len(est_wav2)
    len_mixed = len(mixed_wav)
    if len_est > len_mixed:
        # mixed need is biggest
        est_wav2 = est_wav2[:len_mixed]
    else:
        # if mixed is biggest than estimation wav we need pad with zeros because is expected that this part is silence
        est_wav2 = np.pad(est_wav2, (0, len(mixed_wav)-len(est_wav2)), 'constant', constant_values=(0, 0))

    len_est = len(target_wav)
    len_mixed = len(mixed_wav)
    if len_est > len_mixed:
        # mixed need is biggest
        target_wav = target_wav[:len_mixed]
    else:
        # if mixed is biggest than estimation wav we need pad with zeros because is expected that this part is silence
        target_wav = np.pad(target_wav, (0, len(mixed_wav)-len(target_wav)), 'constant', constant_values=(0, 0))

    # show in notebook results
    print('-'*100)
    print('-'*30,os.path.basename(noise_utterance),'-'*30)
    print("Input/Clean Audio")
    display(Audio(mixed_wav,rate=44100))
    print('Predicted Audio')
    display(Audio(est_wav,rate=44100))
    print('-'*100)
    del target_wav, est_wav, mixed_wav


# in google paper dont is reported SNRi, and not is clean for my when we calculate SNR, for this reason i calculate this
# NOTE: its use other speaker encoder, and other normalization on wavs, for this reason its not directly comparable.
import pandas as pd
from IPython.display import Audio, display
from mir_eval.separation import bss_eval_sources
import numpy as np
# SDR from google paper for this instances
test_csv = pd.read_csv('VoiceSplit/datasets/LibriSpeech/test_demo.csv', sep=',').values
sdrs_before = []
sdrs_after = []
snrs_after = []
snrs_before = []
for noise_utterance, emb_utterance, clean_utterance, clean_utterance2  in test_csv:
    noise_utterance = os.path.join('VoiceSplit',noise_utterance).replace(' ', '')
    emb_utterance = os.path.join('VoiceSplit',emb_utterance).replace(' ', '')
    clean_utterance = os.path.join('VoiceSplit',clean_utterance).replace(' ', '')
    clean_utterance2 = os.path.join('VoiceSplit',clean_utterance2).replace(' ', '')
    est_utterance = noise_utterance.replace('noisy', 'enhanced').replace(' ', '')

    target_wav, _ = librosa.load(clean_utterance, sr=44100)
    target_wav2, _ = librosa.load(clean_utterance2, sr=44100)
    est_wav, _ = librosa.load(est_utterance, sr=44100)
    mixed_wav, _ = librosa.load(noise_utterance, sr=44100)

    len_est = len(est_wav)
    len_mixed = len(mixed_wav)
    if len_est > len_mixed:
        # mixed need is biggest
        est_wav = est_wav[:len_mixed]
    else:
        # if mixed is biggest than estimation wav we need pad with zeros because is expected that this part is silence
        est_wav = np.pad(est_wav, (0, len(mixed_wav)-len(est_wav)), 'constant', constant_values=(0, 0))

    # get wav for second voice, its need for SDR calculation
    est_wav2 = mixed_wav-est_wav

    len_est = len(est_wav2)
    len_mixed = len(mixed_wav)
    if len_est > len_mixed:
        # mixed need is biggest
        est_wav2 = est_wav2[:len_mixed]
    else:
        # if mixed is biggest than estimation wav we need pad with zeros because is expected that this part is silence
        est_wav2 = np.pad(est_wav2, (0, len(mixed_wav)-len(est_wav2)), 'constant', constant_values=(0, 0))

    len_est = len(target_wav)
    len_mixed = len(mixed_wav)
    if len_est > len_mixed:
        # mixed need is biggest
        target_wav = target_wav[:len_mixed]
    else:
        # if mixed is biggest than estimation wav we need pad with zeros because is expected that this part is silence
        target_wav = np.pad(target_wav, (0, len(mixed_wav)-len(target_wav)), 'constant', constant_values=(0, 0))

    # get target_wav for second voice, its recomended because google dont provide clean_utterance2 in your demo i need get in LibreSpeech Dataset, but i dont know if they normalised this file..
    target_wav2 = mixed_wav - target_wav
    '''len_est = len(target_wav2)
    len_mixed = len(mixed_wav)
    if len_est > len_mixed:
        # mixed need is biggest
        target_wav2 = target_wav2[:len_mixed]
    else:
        # if mixed is biggest than estimation wav we need pad with zeros because is expected that this part is silence
        target_wav2 = np.pad(target_wav2, (0, len(mixed_wav)-len(target_wav2)), 'constant', constant_values=(0, 0))'''


    # calculate snr and sdr before model
    ests = [torch.from_numpy(mixed_wav), torch.from_numpy(mixed_wav)] # the same voices is mixed_wav
    egs = [torch.from_numpy(target_wav), torch.from_numpy(target_wav2)]
    mix = torch.from_numpy(mixed_wav)
    _snr, per = permute_SI_SNR(ests, egs, mix)
    _sdr = permutation_sdr(ests, egs, mix, per)
    snrs_before.append(_snr)
    sdrs_before.append(_sdr)

    # calculate snr and sdr after model
    ests = [torch.from_numpy(est_wav), torch.from_numpy(est_wav2)]
    egs = [torch.from_numpy(target_wav), torch.from_numpy(target_wav2)]
    mix = torch.from_numpy(mixed_wav)
    _snr, per = permute_SI_SNR(ests, egs, mix)
    _sdr = permutation_sdr(ests, egs, mix, per)
    snrs_after.append(_snr)
    sdrs_after.append(_sdr)

    # show in notebook results
    print('-'*100)
    print('-'*30,os.path.basename(noise_utterance),'-'*30)
    print("Input/Noise Audio")
    display(Audio(mixed_wav,rate=44100))
    print('Predicted Audio')
    display(Audio(est_wav,rate=44100))
    print('Target Audio')
    display(Audio(target_wav,rate=44100))
    print('Predicted2 Audio')
    display(Audio(est_wav2,rate=44100))
    print('Target2 Audio')
    display(Audio(target_wav2,rate=44100))
    print('-'*100)
    del target_wav, est_wav, mixed_wav


print('='*20,"Before Model",'='*20)
print('\nAverage SNRi: {:.5f}'.format(np.array(snrs_before).mean()))
print('Average SDRi: {:.5f}'.format(np.array(sdrs_before).mean()))

print('='*20,"After Model",'='*20)
print('\nAverage SNRi: {:.5f}'.format(np.array(snrs_after).mean()))
print('Average SDRi: {:.5f}'.format(np.array(sdrs_after).mean()))