import logging
from os.path import join, abspath

# Sockets related
END_MSG="%% CLI DISCONNECT"
ACK_SIG="%% ACK TTS REQUEST"
RES_SIG="%% TTS SUCCESS:"
FAIL_SIG="%% TTS FAILURE:"
TTS_PREFIX="%% TTS THAT PLEASE:"
TTS_COPY="%% SEND THIS PLEASE:"
TTS_END_STREAMING="%% END OF STREAMING %%"
BUFFER_SIZE=16384 # 4096
# Potion2 IP is 129.20.220.94
SERVER_HOST="129.20.220.94" # Use "127.0.0.1" to run on the local loop only (not accible from the outside). 
SERVER_PORT=9801 # The port on which to run the server, has to be in [1;65535] (0 is reserved).
MAX_CACHE_SIZE=10000
REQ_TIMEOUT=1 # 1s
ACK_TIMEOUT=10 # 10s to get an answer after sending a portion of the synthesized file. 
LOGGER_LEVEL=logging.DEBUG

# Streaming configuration
streaming_on=True
package_size=200 #200 16 bit PCM values per package sent

# general configuration
backend="pytorch"
n_gpu=1         # number of gpus ("0" uses cpu, otherwise use gpu), peut necessiter un CUDA_VISIBLE_DEVICE=2
debug_mode=1
verbose_mode=1      # verbose option

data2json_script="utils/data2json.sh"
#data2json_script="/vrac/dguennec/dev/ex-e2e-tts/espnet/utils/data2json.sh"

# feature configuration
fs=22050            # sampling frequency
fmax=""             # maximum frequency
fmin=""             # minimum frequency
n_mels=80           # number of mel basis
n_fft=1024          # number of fft points
n_shift=256         # number of shift points
win_length=""       # window length
inter_utt_pause=500 # size of the silence added at the end of each utterance.

# Cache related
# Backup is enabled if variable backup_file is set to something else than None.
# In that case, it's value is the name of the file in which the backup data
# will be saved. At startup, the server will also lookup this file in order to
# restore the cache. 
backup_file="./cache-backup.bak"

# The path where NLTK packages should be downloaded. Please check your
# NLTK_DATA env variable as well and use option download_dir=nltk_path when
# using nltk.download(). E.g. (from the server's original code): 
# nltk.download('punkt', download_dir=consts.nltk_path)
nltk_path="/vrac/software/nltk"


# Languages
lang_en="en"        # English
lang_fr="fr"        # French
lang_brz="brz"      # Breton
lang_aro="aro"      # Occidental Armenian
lang_none="none"    # Used to specify to the server that phonetization should
                    # be bypassed.


#########################
# English voice config. #
#########################

# IMPORTANT. Again, server & client have to be started from the root directory:
default_voice_config="conf/english.ljspeech.v1.conf"

# TTS model
#synth_model="/vrac/dguennec/store/english/espnet/phn.v1.tacotron2.ljspeech/exp/phn_train_no_dev_pytorch_train_pytorch_tacotron2.v1/results/model.last1.avg.best"
#dict_type="/vrac/dguennec/store/english/espnet/phn.v1.tacotron2.ljspeech/data/lang_1phn/phn_train_no_dev_units.txt"
#trans_type="phn"
#cmvn="/vrac/dguennec/store/english/espnet/phn.v1.tacotron2.ljspeech/data/phn_train_no_dev/cmvn.ark"
#decode_config="/vrac/dguennec/store/english/espnet/phn.v1.tacotron2.ljspeech/conf/tuning/decode.yaml"
#model_json="/vrac/dguennec/store/english/espnet/phn.v1.tacotron2.ljspeech/exp/phn_train_no_dev_pytorch_train_pytorch_tacotron2.v1/results/model.json"

# Vocoder model
#vocoder_path="/vrac/dguennec/store/english/vocoder/parallel_wavegan.ljspeech/train_nodev_ljspeech_parallel_wavegan.v1.long_custom/checkpoint-1500000steps.pkl"
#vocoder_conf="/vrac/dguennec/store/english/vocoder/parallel_wavegan.ljspeech/train_nodev_ljspeech_parallel_wavegan.v1.long_custom/config.yml"
#vocoder_stats="/vrac/dguennec/store/english/vocoder/parallel_wavegan.ljspeech/train_nodev_ljspeech_parallel_wavegan.v1.long_custom/stats.h5"

default_lang=lang_en

# Output directories & files
decode_dir_name="./decode/"
decode_dir=abspath(decode_dir_name)
input_txt="input_file.txt"

wav_folder = "wav_wvn"

def main_dir(file_basename:str):
    """ directory where the results of the TTS job will all be put. Requires
    the basename of the file to synthesize as it will be used as the name of 
    the main directory.

    Args:
        file_basename (str): basename of the file to synthesize. 
    """
    return join(decode_dir, file_basename)

def data_dir(file_basename:str):
    return join(main_dir(file_basename), "data")

def dump_dir(file_basename:str):
    return join(main_dir(file_basename), "dump")

def log_dir(file_basename:str):
    return join(main_dir(file_basename), "log")

def outputs_dir(file_basename:str):
    return join(main_dir(file_basename), "outputs")

def outputs_denorm_dir(file_basename:str):
    return join(main_dir(file_basename), "outputs_denorm")

def wav_wnv_dir(file_basename:str):
    return join(main_dir(file_basename), "wav_wvn")


