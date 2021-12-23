# VASP（Visual-Audio Signal Processing）环境配置

## 0 Preface

这是一份我在完成视听导project时配置环境的总结，其中详细记录了安装所需各种包的方法。我的电脑是M1芯片的 Macbook Air，因此有些安装方法或者出现的问题并不普适。这里使用了py38是因为使用conda-forge创建虚拟环境时py36失败了，并且TensorFlow目前在由适配M1芯片的py38版本，因此就选择了py38。因此在完成project前需要在其它电脑上使用Anaconda创建py36环境验证代码。

## 1 创建虚拟环境

创建python3.8虚拟环境vasp:
```zsh
conda create -n vasp python=3.8 
```

启动虚拟环境:
```zsh
conda activate vasp
```

## 2 安装numpy scipy soundfile ffmpeg-python

安装requirements.txt中的包（numpy scipy nussel matplotlib soundfile ffmpeg-python）:
```zsh
pip3 install -r requirements.txt
```

安装sndfile，否则无法使用soundfile:
```zsh
conda install libsndfile
```

测试numpy,scipy,matplotlib,soundfile,ffmpeg-python安装成功，在终端中输入:
```zsh
python3
```

进入python3交互式解释器，并尝试导入这些包:
```python
import numpy,scipy
import matplotlib.pyplot as plt
import ffmpeg
import soundfile
import nussel
```
没有报错则说明安装成功。

## 3 安装face_recognition

### 3.1 安装dlib

face_recognition需要首先安装dlib，这里使用pip安装。也可以参考事自己[使用cmake编译的方法](https://zhuanlan.zhihu.com/p/296580468)。首先安装dlib的依赖:
```zsh
conda install openblas
conda install opencv
```

此后安装cmake:
```zsh
pip3 install cmake
```

最后安装dilb:
```zsh
pip3 install dlib
```

进入python3交互式解释器，验证安装成功:
```python
import dlib
```

### 3.2 安装face_recognition
在终端中输入:
```zsh
pip3 install face_recognition
```

进入python3交互式解释器，验证安装成功:
```python
import face_recognition
```

## 4 安装resemblyzer

使用pip安装:
```zsh
pip3 install resemblyzer
```

出现报错，发现是llvmlite安装失败。因此先安装llvmlite:
```zsh
conda install llvmlite
```

此后重新使用pip安装，经验证发现安装成功。

## 5 安装nussel
直接使用pip安装出现报错，发现是pandas、sox和grpcio安装失败，因此使用conda安装：
```zsh
conda install pandas
conda install sox
conda install grpcio
```

此后再使用pip安装成功：
```zsh
pip3 install nussel
```

## 6 安装SpeechBrain
直接使用pip安装出现报错，发现是ruamel和sentencepiece安装失败，因此使用conda安装：
```zsh
conda install ruamel
conda install sentencepiece
```

此后再使用pip安装成功：
```zsh
pip install speechbrain
```

## 7 task3

依次输入如下命令即可：

```zsh
! pip install -U tqdm numpy librosa mir_eval matplotlib Pillow  tensorboardX pandas torchaudio PyYAML pysoundfile ffmpeg-normalize
conda install mir_eval
conda install imageio
conda install IPython
conda install -c conda-forge librosa
```

