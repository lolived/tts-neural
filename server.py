""" Code of the TTS server. This is the main file that launches the server app.

Original file on which socket server architecture used in this app is based:
https://github.com/pricheal/python-client-server/blob/master/server.py
"""
import argparse
import array
import click
import codecs
import json
import logging
import nltk
import numpy as np
import pathlib
import psutil
import socket
import soundfile

from enum import Enum
from os import mkdir, system
from os.path import exists, join, abspath
from os.path import split as path_split
from pydub import AudioSegment
from time import time
from threading import Thread, Lock
from uuid import uuid4
from yaml import load, BaseLoader

from tacotron_cleaner.cleaners import custom_english_cleaners

# Local lib
import constants as consts
from model_info import ModelInfo
from phonetize_en import g2p_en, init_g2p_en
from phonetize_brz import g2p_brz
from phonetize_fr import g2p_fr
from phonetize_aro import g2p_aro
from server_pytorch_resources import TTSInferenceModel, VocoderInferenceModel

# Initialize the english phonetizer
f_g2p = init_g2p_en()

# Global variable managing info about the files associated to running models.
model_info = ModelInfo()

# Logger configuration
logger = logging.getLogger("tts_server_log")
logger.setLevel(consts.LOGGER_LEVEL)
handler = logging.FileHandler("./tts_server.log")
handler.setLevel(consts.LOGGER_LEVEL)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Global variables for holding information about connected sockets.
# Each element in connections is an instance of the Client class below.
connections_mutex = Lock()
connections = []
total_connections = 0

# Whether streaming the audio back to the client is activated or not.Activated
# by default.
streaming_audio_enabled = True

# Dictionnary holding information about cached data
# indices are the requested text and associated fields are the location of the synthesized file on disk.
cache_mutex = Lock()
cache = {}
popularity = {}

# Mutex for using inference models (TTS & Vocoder)
inference_tts_mutex = Lock()
inference_vocoder_mutex = Lock()

# Classes managing TTS
inference_tts = TTSInferenceModel()
inference_vocoder = VocoderInferenceModel()

# The following function handle the phonetization of English text sentences.
# ---------------------------------------------------------------------------
def clean_and_phonetize_text(text:str, lang:str=consts.default_lang):
    """ TODO

    Args:
        text (str): TODO

    Returns:
        str: the phonetic string matching the input argument text, ready for
        use with the models.
    """
    clean_content = custom_english_cleaners(text.rstrip())
    clean_text = clean_content.lower()
    if lang == consts.lang_en:
        phonetized_content = g2p_en(clean_text, f_g2p)
    elif lang == consts.lang_fr:
        phonetized_content = g2p_fr(clean_text)
    elif lang == consts.lang_brz:
        phonetized_content = g2p_brz(clean_text)
    elif lang == consts.lang_aro:
        phonetized_content = g2p_aro(clean_text)
    elif lang == consts.lang_none:
        phonetized_content = clean_text
    else:
        raise ValueError(f"Unrecognized language specification for the phonetizer: {lang}") 
    return phonetized_content


# The following functions are for cache handling.
# -----------------------------------------------
def assess_cache_integrity():
    """ Checks the content of the cache links to actual files and deletes 
    occurences where linked path doesn't match with a file on disk.
    """
    cache_mutex.acquire()
    try:
        global cache
        issues_found=[]
        for text, path in cache.items():
            if not exists(path):
                logger.warning(f"assess_cache_integrity found that a (text, path) pair referenced in the cache does not correspond to a real file anymore: [\"{text}\", {path}].")
                issues_found.append(text)
        if len(issues_found) > 0:
            logger.warning(f"assess_cache_integrity found a total of {len(issues_found)} incorrect/outdated pairs in the cache.")
            for key in issues_found:
                del cache[key]
            logger.warning(f"The cache was updated in consequence.")

            if consts.backup_file is not None:
                save_cache_state()
                logger.info(f"assess_cache_integrity has backed-up the state of the cache again.")
    finally:
        cache_mutex.release()


def load_cache(save_file:str=consts.backup_file):
    """ Loads a save file to restore the dictionary keeping the cache from a
    previous save state.

    Args:
        save_file (str): the path to the save file containing the state of
        the cache.
    """
    cache_mutex.acquire()
    try:
        global cache
        global popularity
        if exists(save_file):
            logger.info(f"Recovering previous cache info from file {save_file}")
            try:
                with open(save_file, 'r') as open_save_file:
                    cache = json.load(open_save_file)
                    popularity = {}
                    for key in cache.keys():
                        popularity[key] = 1
                    logger.info("Successfully recovered cache data.")
            except Exception as err:
                logger.error(f"Loading the cache from the backup file failed with the following error: {err}.")
                raise OSError
        else:
            logger.warning(f"Can't find the save file {save_file} so cache was NOT restored to the previous state.")
    finally:
        cache_mutex.release()


def save_cache_state(file_name:str=consts.backup_file):
    """Saves the cache state (ie. the content of dictionnary variable `cache`)
    to a file on disk. For now, the popularity list isn't being saved.

    Args: 
        file_name (str): the path to the file in which the data will be saved.
    """
    cache_mutex.acquire()
    try:
        global cache
        with open(file_name, 'w') as open_save_file:
            json.dump(cache, open_save_file)
    finally:
        cache_mutex.release()


def check_cache(text:str) -> str:
    """By default, the returned uri is set to "", aka. the empty string to
    signify that requested text is not present in the cache. If it gets
    populated, it will be so with the path to the related synthesized file, 
    thus also signifying that the requested text was indeed cached.

    Args:
        text (str): the text to search for in the cache.
    """
    cache_mutex.acquire()
    try:
        global cache
        uri = "" # returns "" if the text isn't in the cache.
        if text in cache:
            uri = cache[text]
        return uri
    finally:
        cache_mutex.release()


def update_cache(text:str, uri:str):
    """Looks for a text utterance in the cache and adds it along with the
    related synthesized audio file if it is not already present. If the cache
    does not have any space left anymore, the first entry in it is deleted.
    As the cache is expanded chronologically, the first entry is also the
    oldest.

    Args:
        text (str): the text that has been synthesized and is considered for
        adding to the cache.
        uri (str): the path to the audio file corresponding to the text.
    """
    global cache

    cache_mutex.acquire()
    try:
        cache_len = len(cache)
    finally:
        cache_mutex.release()

    # Not doing the mutex acquisition here because check_cache() needs it.
    if not check_cache(text):
        # Add the pair to the cache
        cache_mutex.acquire()
        try:
            if cache_len >= consts.MAX_CACHE_SIZE:
                # Delete the cache entry 
                looser = min(popularity, key=popularity.get)
                del popularity[looser]
                del cache[looser]
            cache[text] = uri
            popularity[text] = 1
        finally:
            cache_mutex.release()
    else:
        cache_mutex.acquire()
        try:
            popularity[text] += 1
        finally:
            cache_mutex.release()

    # Each time an update to the cache is made, we check if backup mode is
    # enabled in the main config parameters. If so, the cache is backed-up
    # on disk. This erases the previous backup file.
    if consts.backup_file is not None:
        save_cache_state()


def make_tts_directory_structure(base_filename:str):
    """Creates the entire directory structure needed to synthesize a file
    inside the decode folder. Requires the basename of the file to synthesize
    as the base directory will be named after it.

    Args:
        base_filename (str): base filename of the file to synthesize.
    """
    global model_info

    try:
        # Also checks for necessary files existence:

        # TTS model
        if not exists(model_info.get_synth_model()):
            logger.error(f"No such file: {model_info.get_synth_model()}")
            raise FileNotFoundError
        if not exists(model_info.get_dict_type()):
            logger.error(f"No such file: {model_info.get_dict_type()}")
            raise FileNotFoundError
        # The CMVN file may not be needed
        if model_info.get_cmvn() != "" and not exists(model_info.get_cmvn()):
            logger.error(f"No such file: {model_info.get_cmvn()}")
            raise FileNotFoundError
        if not exists(model_info.get_decode_config()):
            logger.error(f"No such file: {model_info.get_decode_config()}")
            raise FileNotFoundError
        if not exists(model_info.get_model_json()):
            logger.error(f"No such file: {model_info.get_model_json()}")
            raise FileNotFoundError

        # Vocoder model
        if not exists(model_info.get_vocoder_path()):
            logger.error(f"No such file: {model_info.get_vocoder_path()}")
            raise FileNotFoundError
        if not exists(model_info.get_vocoder_conf()):
            logger.error(f"No such file: {model_info.get_vocoder_conf()}")
            raise FileNotFoundError
        if not exists(model_info.get_vocoder_stats()):
            logger.error(f"No such file: {model_info.get_vocoder_stats()}")
            raise FileNotFoundError

        # Deliberately avoiding os.makedirs()
        if not exists(consts.decode_dir):
            print(f"The decoding directory did not exists and is now being created there: {consts.decode_dir}")
            mkdir(consts.decode_dir)
        mkdir(consts.main_dir(base_filename))
        mkdir(consts.data_dir(base_filename))
        mkdir(consts.dump_dir(base_filename))
        mkdir(consts.log_dir(base_filename))
        mkdir(consts.outputs_dir(base_filename))
        mkdir(consts.outputs_denorm_dir(base_filename))
        mkdir(consts.wav_wnv_dir(base_filename))
    except OSError as err:
        logger.error(f"Couldn't make the directory structure: {err}.")
        exit(1)


def make_input_file(base_filename:str, text_to_tts:str):
    with open(join(consts.data_dir(base_filename), consts.input_txt), "w") as input_txt_f:
        input_txt_f.write(text_to_tts)


def data_preparation(base_filename:str, text_to_tts:str, lang:str):
        logger.info("Stage 1: Data preparation")
        global model_info
        if model_info.get_trans_type() == "phn":
            logger.info("Data cleaning and phonetization to be performed as input is set to phn.")
            ready_txt=join(consts.data_dir(base_filename), f"{base_filename}.clean_pho.txt")
            cleaned_txt=clean_and_phonetize_text(text_to_tts, lang)
            logger.info("The text was successfully phonetized.")        
            with open(ready_txt, "w") as ready_f:
                ready_f.write(cleaned_txt)
        else:
            logger.info("No data cleaning as the input is set to char.")
            cleaned_txt=text_to_tts
        

        base_id = f"{base_filename}_1"
        with open(join(consts.data_dir(base_filename), "wav.scp"), "w") as wav_scp_f:
            wav_scp_f.write(f"{base_id} X")
        with open(join(consts.data_dir(base_filename), "spk2utt"), "w") as spk2utt_f:
            spk2utt_f.write(f"X {base_id}")
        with open(join(consts.data_dir(base_filename), "utt2spk"), "w") as utt2spk_f:
            utt2spk_f.write(f"{base_id} X")
        with open(join(consts.data_dir(base_filename), "text"), "w") as text_f:
            text_f.write(f"{base_id} {cleaned_txt}")

        # FIXME use something else than system.
        return_code = system(f"{consts.data2json_script} --trans_type {model_info.get_trans_type()} {consts.data_dir(base_filename)} {model_info.get_dict_type()} > {consts.dump_dir(base_filename)}/data.json")
        if return_code != 0: 
            logger.error(f"data2json.sh returned the following execution code: {return_code}")
            raise OSError


def tts_decoding(base_filename:str):
    global model_info
    logger.info("Stage 2: Decoding")
    logger.info(f"Using model: {model_info.synth_model}.")

    feats_location = join(consts.outputs_dir(base_filename), "feats")
    logger.debug(f"feats_dir = {feats_location}")
    data_json_file = join(consts.dump_dir(base_filename), "data.json")
    logger.debug(f"data_json_file = {data_json_file}")

    # Usage of inference models is protected by a mutex lock.
    inference_tts_mutex.acquire()
    try:
        global inference_tts
        load_inputs_and_targets, js = inference_tts.prepare_decoding(data_json_file, feats_location)
        logger.info("Inference preparation done.")
        inference_tts.decode(feats_location, load_inputs_and_targets, js)
        logger.info("Inference done successfully (it seams :).")
    finally:
        inference_tts_mutex.release()


def vocoder_generation(base_filename:str, nb:int) -> str:
    logger.info("Stage 4: Synthesis with Neural (parallel wavegan) Vocoder")
    
    feats_dir=consts.outputs_dir(base_filename)
    feats_file = join(consts.outputs_dir(base_filename), f"feats.scp")

    # Usage of inference models is protected by a mutex lock.
    inference_vocoder_mutex.acquire()
    try:
        global inference_vocoder
        wav_file = inference_vocoder.decode(feats_file, consts.wav_wnv_dir(base_filename), nb)
    finally:
        inference_vocoder_mutex.release()

    return wav_file


def run_tts(text_to_tts: str, base_filename:str, nb:int=0, lang:str=consts.default_lang) -> str:
    """This function is where we perform the TTS task.

    Args:
        text_to_tts (str): the text to synthesize.
        nb (int): if several files will have to be synthesized, this serves
        to provide a distinct name to each.

    Returns:
        str: the path to the synthesized file. Empty if the synthesis failed.
    """
    uri_synth_file = ""

    make_input_file(base_filename, text_to_tts)
    data_preparation(base_filename, text_to_tts, lang)
    tts_decoding(base_filename)
    uri_synth_file = vocoder_generation(base_filename, nb)
    return uri_synth_file


def concat_synth_files(path_list:list, base_filename:str):
    """ TODO

    Args:
        path_list (list): the list of wav files to concatenate.
        base_filename (str): the unique base filename associated to this
        request.

    Returns:
        str: concatenated wav path.
    """
    concatenated_wav_path = join(consts.wav_wnv_dir(base_filename), f"{base_filename}.wav")
     
    if not path_list:
        logger.error(f"list of files to concatenate is empty, no file to concatenate: {path_list}.")
        raise OSError

    complete_file = None
    for a_path in path_list:
        # Systematically adding a consts.inter_utt_pause ms pause between each
        # utterance and at the end.
        data = AudioSegment.from_wav(a_path) + AudioSegment.silent(duration=consts.inter_utt_pause)
        if complete_file is not None:
            complete_file += data
        else:
            complete_file = data
    
    # soundfile.write(concatenated_wav_path, complete_file, consts.fs, "PCM_16")
    complete_file.export(concatenated_wav_path, format="wav")
    logger.info(f"Final wav has been concatenated in file {concatenated_wav_path}")

    return concatenated_wav_path


def get_raw_frames(wav_path:str) -> array:
    sound = AudioSegment.from_file(wav_path)
    samples = sound.get_array_of_samples()
    return samples


def cleanup():
    global inference_tts
    global inference_vocoder

    inference_tts_mutex.acquire()
    try:
        inference_tts.cleanup()
    finally:
        inference_tts_mutex.release()

    inference_vocoder_mutex.acquire()
    try:
        inference_vocoder.cleanup()
    finally:
        inference_vocoder_mutex.release()


#Client class, new instance created for each connected client
#Each instance has the socket and address that is associated with items
#Along with an assigned ID and a name chosen by the client
class Client(Thread):
    """ Client class, new instance created for each connected client. Each
    instance has the socket and address that is associated with items along with
    an assigned ID and a name chosen by the client.

    Instance variables:
        self.socket (socket.socket): the socket open to communicate with the
        client.

        self.address (str): the address, logically the IP. "localhome" may also
        be used instead of the local loop.

        self.id (str): The index of the client instance in the connections
        global variable.

        self.name (str): A string to help identifying the client.

        self.signal (bool): while true, the loop in the run function will keep
        going.
    """

    def __init__(self, socket: socket.socket, address:str, id:str, name:str, signal:bool, stream_audio:bool, sample_rate:int):
        """ Constructor.
        """
        Thread.__init__(self)
        self.socket = socket
        self.address = address
        self.id = id
        self.name = name
        self.signal = signal
        self.stream_audio = stream_audio
        self.sample_rate = sample_rate
    

    def __str__(self):
        return str(self.id) + " " + str(self.address)


    def disconnect(self):
        """ Disconnects the client and removed the client from the connections global variable.
        """
        logger.warning("Client " + str(self.address) + " has disconnected")
        self.signal = False
        connections_mutex.acquire()
        try:
            connections.remove(self)
        finally:
            connections_mutex.release()
        del self


    def run(self):
        """ Attempt to get data from client. If unable to, assume client has
        disconnected and remove him from server data. If able to and we get
        data back, process it. .decode is used to convert the byte data into a
        printable string.
        Messages are processed and TTS is managed when asked by the client.
        """
        while self.signal:
            try:
                data = self.socket.recv(consts.BUFFER_SIZE)
            except Exception as err:
                logger.error(f"The worker associated to client of id {self.id} encountered an exception: {err} ")
                self.disconnect()
                break
            message_content = str(data.decode("utf-8"))
            if message_content != "":
                logger.info("Received message from ID " + str(self.id) + ": \"" + message_content+"\"")
                if message_content == consts.END_MSG:
                    # In that scenario, the client specified that it is
                    # in the process of disconnecting so we simply
                    # release the handler for this client.
                    self.disconnect()
                    break
                elif consts.TTS_PREFIX in message_content:
                    # This is the main scenario: a TTS request was made by the
                    # client and we have to process it.

                    # First the we tell the client we accept to provide the
                    # synthesized file once we have it.
                    self.socket.sendto(str.encode(consts.ACK_SIG), self.address)
                    text_to_tts = message_content.replace(consts.TTS_PREFIX, "")
                    # Security measure: making sure the first character doesn't
                    # cause an empty utterance during the split if it happens to
                    # be a delimitor. Idem for the last character.
                    if text_to_tts[0] == '|':
                        text_to_tts = text_to_tts[1:]
                    if text_to_tts[-1] == '|':
                        text_to_tts = text_to_tts[:-2]
                    utterances = text_to_tts.split('|')
                    logger.info(f"TTS request acknowledged: {utterances}")

                    synth_paths = []
                    total_synth_time = 0
                    
                    base_filename = str(uuid4())
                    make_tts_directory_structure(base_filename)

                    for nb, utt in enumerate(utterances):
                        # Then we check the text to synthesize isn't already in the
                        # cache.
                        uri_synth_file = check_cache(utt)
                        if uri_synth_file:
                            # If uri_synth_file isn't empty, this means the text to
                            # synthesize is already present in the cache.
                            # In that case, we just log that no time was spent
                            # generating the audio file.
                            time_start = 0
                            time_end = 0
                        else:
                            # If the text wasn't in the cache, we have to generate
                            # it and then we must update the cache.
                            time_start = time()
                            uri_synth_file = run_tts(utt, base_filename, nb, model_info.get_instance_lang())
                            cleanup()
                            time_end = time()
                        if uri_synth_file:
                            synth_paths.append(uri_synth_file)
                            update_cache(utt, uri_synth_file)
                        total_synth_time += (time_end - time_start)
                    
                    if len(synth_paths) == len(utterances):
                        # if  check_cache() / run_tts() returned non-empty
                        # strings, this means synthesis was successful and a 
                        # file was generated. In that case, we forward the path
                        # to the client and update the cache.
                        logger.info(f"Requested file {uri_synth_file} synthesized in {total_synth_time}s.")
                        final_synth_path = concat_synth_files(synth_paths, base_filename)
                        self.socket.sendto(str.encode(consts.RES_SIG+final_synth_path), self.address)
                 
                        if self.stream_audio:
                            # This portion of the code deals with the sending of
                            # the synthesized file to the client via streaming.
                            data_array = get_raw_frames(final_synth_path)
                            n_samples = len(data_array)
                            logger.info(f"There is {n_samples} values to send.")
                            logger.info(f"Sample rate: {self.sample_rate}")
                            i = 0
                            j = consts.package_size
                            the_end = ""
                            while i < n_samples-1:
                                if j >= n_samples:
                                    j = n_samples - 1
                                    the_end=consts.TTS_END_STREAMING
                                logger.info(f"Sending data from {i} to {j} out of {n_samples}.")
                                data = ""
                                for value in data_array[i:j]:
                                    data += f"{value}|"
                                data_to_send = consts.TTS_COPY+f"{i}|{j}|{n_samples}|{self.sample_rate}|"+data+the_end
                                self.socket.sendto(str.encode(data_to_send), self.address)

                                ack_not_received = True
                                ack_timeout_start_time = time()
                                while ack_not_received:
                                    try:
                                        some_data = self.socket.recv(consts.BUFFER_SIZE)
                                    except Exception as err:
                                        logger.error(f"The worker associated to client of id {self.id} encountered an exception: {err} ")
                                        self.disconnect()
                                        break
                                    some_content = str(some_data.decode("utf-8"))
                                    if consts.ACK_SIG in some_content:
                                        ack_not_received = False
                                    if (time() - ack_timeout_start_time) > consts.ACK_TIMEOUT:
                                        logger.error(f"The client was sent a package but failed to respond in time. Aborting and disconnecting from the client.")
                                        self.disconnect()

                                # Sending loop progression 
                                i = j
                                j += consts.package_size

                            # End of the code dealing with the streaming of the
                            # audio file back to the client.
                    else:
                        logger.error(f"TTS attempt lasted {total_synth_time}s but failed. Please check the logs.")
                        self.socket.sendto(str.encode(consts.FAIL_SIG), self.address)
                    # Only one synthesis is allowed for each socket, as a
                    # security mesure.
                    self.disconnect()
                    break
                else:
                    # This is in the case the client sends an unexpected
                    # message. In that case, we report what the message was
                    # and then disconnect as a safety measure.
                    logger.error(f"Client at ID [{self.id}] sent the following unexpected message: {message_content}")
                    self.disconnect()
            del data


def new_connections(socket:socket.socket, sample_rate:int):
    """ Wait for new connections and start the service to handle it when first
    connected.
    """
    while True:
        sock, address = socket.accept()
        connections_mutex.acquire()
        try:
            global total_connections
            # Adding a new client to the connected clients list
            connections.append(Client(sock, address, total_connections, f"Client_{str(len(connections))}", True, streaming_audio_enabled, sample_rate))
            # Starting the thread handling the new client
            connections[len(connections) - 1].start()
            logger.info("New connection at ID " + str(connections[len(connections) - 1]))
            total_connections += 1
        finally:
            connections_mutex.release()

@click.command()
@click.option('--empty-cache/--with-cache', default=False, help='Start the server with an empty cache (--empty-cache) or try to load a previous state if the file exists (--with-cache, default).')
@click.option('--listeners', type=int, default=5, help='How many connexions the server will accept, past the maximum available number.')
@click.option('--force-cpu/--auto', default=False, help='--auto (default) lets the server decide whether it runs in cpu or gpu mode, depending on hardware availability on the machine. --force-cpu forces the server to run on cpu (for testing purposes or if the gpu is doing something else mainly).')
@click.option('--host', type=str, default=consts.SERVER_HOST, help='The host ip address on which to run the server. Default is localhost.')
@click.option('--port', type=int, default=consts.SERVER_PORT, help='The port on which to run the server, has to be in [1;65535] (0 is reserved).')
@click.option('--tts-model', type=str, help="TTS model location")
@click.option('--dict-file', type=str, help="Dictionary location")
@click.option('--trans', type=str, help="\"phn\" for phoneme-based; \"char\" for character-based.")
@click.option('--cmvn', type=str, help="cmvn file location")
@click.option('--decode', type=str, help="Decode configuration file location.")
@click.option('--model-json', type=str, help="model.json file location.")
@click.option('--vocoder-model', type=str, help="Vocoder file location.")
@click.option('--vocoder-conf', type=str, help="Vocoder configuration location.")
@click.option('--vocoder-stats', type=str, help="Vocoder stats file location.")
@click.option('--streaming/--no-streaming', default=consts.streaming_on, help="Enable streaming of the sythesized file back to the client.")
@click.option('--lang', type=str, default=None, help="Language setting for the phonetizer. If option --trans is being set to \"char\" (either through the option itself or with the config file ; see option --conf), then this option MUST be set to \"lang_none\".")
@click.option('--conf', type=str, default=consts.default_voice_config, help="Pass a config file to the server to load models and configure the server.")
def main(empty_cache:bool, listeners:int, force_cpu:bool, host:int, port:int, tts_model:str, dict_file:str, trans:str, cmvn:str, decode:str, model_json:str, vocoder_model:str, vocoder_conf:str, vocoder_stats:str, streaming:bool, lang:str, conf:str):
    """ Main body of the server process. Setups configuration and starts the
    server socket & the associated communication thread.

    Args:
        empty_cache (bool): Command line argument, see the help message above in the function decorators.
        listeners (int): Idem.
        force_cpu (bool): Idem.
        host (int): Idem.
        port (int): Idem.
        TODO: expand.
    """
    global model_info
    if conf:
        model_info.load_config(conf)
    if tts_model:
        model_info.set_synth_model(tts_model)
    if dict_file:
        model_info.set_dict_type(dict_file)
    if trans:
        model_info.set_trans_type(trans)
    if cmvn:
        model_info.set_cmvn(cmvn)
    if decode:
        model_info.set_decode_config(decode)
    if model_json:
        model_info.set_model_json(model_json)
    if vocoder_model:
        model_info.set_vocoder_path(vocoder_model)
    if vocoder_conf:
        model_info.set_vocoder_conf(vocoder_conf)
    if vocoder_stats:
        model_info.set_vocoder_stats(vocoder_stats)

    print(model_info.get_model_json())

    # Find sample rate from vocoder configuration file
    try:
        yaml_conf = open(model_info.get_vocoder_conf(), 'r')
        yaml_conf = load(yaml_conf, Loader=BaseLoader)
        sample_rate = int(yaml_conf.get("sampling_rate"))
    except Exception as e:
        raise ValueError(
            f"Loading configuration file {conf} failed with the following error; please check the file exists and is valid in regard to the yaml standard: {e}")

    if lang:
        if model_info.get_trans_type() == "char" and lang != consts.lang_none:
            raise ValueError(f"Option lang cannot be something other than {consts.lang_none} when trans type is set to char.")
        model.info.set_instance_lang(lang)

    global streaming_audio_enabled
    streaming_audio_enabled = streaming

    try:
        #Create new server socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((host, port))
        sock.listen(listeners)
    
        global inference_tts
        global inference_vocoder
        n_gpu = consts.n_gpu
        if force_cpu:
            n_gpu = 0
        inference_tts.prepare_tts(model_info.get_synth_model(), model_info.get_model_json(), n_gpu)
        inference_vocoder.prepare_vocoder(model_info.get_vocoder_path(), model_info.get_vocoder_conf(), force_cpu)

        # Loading the cache if turned on and checking that contained information
        # is correct.
        if consts.backup_file is not None and not empty_cache:
            load_cache()
            assess_cache_integrity()

        #Create new thread to wait for connections
        new_connections_thread = Thread(target = new_connections, args = (sock,sample_rate))
        new_connections_thread.start()
        logger.info(f"The server is running at IP {host} on port {port}.")
        print(f"The server is running at IP {host} on port {port}.")

    except KeyboardInterrupt:
        print("Stopping the server...")
        sock.shutdown()
        sock.close()
        new_connections_thread.stop()
        print("Done.")
        exit(1)


if __name__ == "__main__":
    main()
