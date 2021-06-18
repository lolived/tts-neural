""" TODO

Original file for the socket client architecture:
https://github.com/pricheal/python-client-server/blob/master/client.py
"""
import click
import logging
import numpy as np
import os
import psutil
import socket
import threading

from os.path import basename, join
from time import sleep
from pydub import AudioSegment
from sys import exit
from uuid import uuid4

# Local lib
import constants as consts
from server_pytorch_resources import TTSInferenceModel, VocoderInferenceModel
from server import make_tts_directory_structure, make_input_file, data_preparation, concat_synth_files
from preprocess_tts import pre_process_text_for_tts

# Logger configuration
logger = logging.getLogger("tts_client_log")
logger.setLevel(consts.LOGGER_LEVEL)
handler = logging.FileHandler("./tts_client.log") #TODO Maybe create a log file per execution and place it in the associated dir in decode?
handler.setLevel(consts.LOGGER_LEVEL)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Global variables
looking_for_tts = False
tts_ack = False
host_name = consts.SERVER_HOST
port_nb = consts.SERVER_PORT
final_fn = ""
sampling_frequency = None

def write_audio_file(file_path:str, data:list, sr:str):
    """ This function is based on the assumption that the data that is passed
    to it is a numpy array of integers.
    """
    sound = AudioSegment(
                data.tobytes(), # Data array
                sample_width=2, # 2 byte (16 bit) samples
                frame_rate=sr, # Sampling frequency
                channels=1 # mono
            )
    file_handle = sound.export(file_path, format="wav")


def end_program():
    """ Kills the whole program.
    """
    logger.info("Quitting the program.")
    current_process_pid = os.getpid()
    this_program = psutil.Process(current_process_pid)
    this_program.terminate()


def receive(socket:socket.socket, signal:bool):
    """ This function manages both the sending/reception of messages through the
    socket and the progress of the TTS job.

    Args:
        socket (socket.socket): an open socket to send and receive messages.
        
        signal (bool): while true, the function will loop and await messages
        form the server.
    """
    global looking_for_tts
    global tts_ack
    global sampling_frequency
    end_process = False
    waiting_for_audio = False
    uri_result = "temp.wav"
    received_data = None
    n_received = 0
    while signal:
        try:
            data = socket.recv(consts.BUFFER_SIZE)
            message_content = str(data.decode("utf-8"))
            if not looking_for_tts or (consts.RES_SIG in message_content and not tts_ack):
                logger.error(f"Received this unawaited response: {message_content}.")
                signal = False
                raise SystemError
            else:
                if message_content == consts.ACK_SIG:
                    logger.info("TTS request acknowledged by the server.")
                    tts_ack = True
                elif tts_ack and not waiting_for_audio and consts.RES_SIG in message_content: 
                    uri_result = message_content.replace(consts.RES_SIG, "")
                    logger.info(f"The server finished the synthesis of requested text and produced the following file:{uri_result}")
                    print(f"The server finished the synthesis of requested text and produced the following file ON THE SERVER: {uri_result}.")
                    if streaming_activated:
                        waiting_for_audio = True
                    else:
                        waiting_for_audio = False # not necessary but done out of caution.
                        looking_for_tts = False
                        end_process = True
                elif waiting_for_audio and consts.TTS_COPY in message_content:
                    data = message_content.replace(consts.TTS_COPY, "")
                    data = data.replace(consts.TTS_END_STREAMING, "")
                    data = data.split("|")
                    if data[-1] == "":
                        del data[-1]
                    i = int(data[0])
                    j = int(data[1])
                    n_samples = int(data[2])
                    sample_rate = int(data[3])
                    
                    # If the user did not specify a sampling_frequency, rely on the one provided by the server
                    if sampling_frequency is None:
                        sampling_frequency = sample_rate
                        
                    logger.debug(f"Received data from {i} to {j} ({len(data)} samples) out of a total of {n_samples}, fs={sample_rate}")
                    del data[3]
                    del data[2]
                    del data[1] # Don't change the deletion order!
                    del data[0]
                    if received_data is None:
                        received_data = np.zeros(n_samples)
                        logger.debug(f"Allocated the Numpy array used to store audio data temporarily. It is now of size {len(received_data)}.")
                    received_data[i:j] = np.array(data)
                    n_received += (j - i)
                    if consts.TTS_END_STREAMING in message_content:
                        final_name = final_fn if final_fn else join(consts.decode_dir, basename(uri_result))
                        print(f"Saving the final result as {final_name} ON THIS MACHINE.")
                        write_audio_file(final_name, received_data.astype(np.int16), sampling_frequency)
                        waiting_for_audio = False
                        looking_for_tts = False
                        end_process = True
                    send(socket, consts.ACK_SIG)
                elif tts_ack and consts.FAIL_SIG in message_content:
                    logger.error("Received news from the server that TTS attempt FAILED. Please check the logs.")
                    print("TTS Error on server side.")
                    looking_for_tts = False
                    end_process = True
                else:
                    logger.error(f"Received this meaningless message from the server: {message_content}")
                    signal = False
                    raise SystemError
            if end_process:
                send(socket, consts.END_MSG)
                print("Now exiting normally...")
                signal = False
                end_program()
                exit(0)
        except Exception as err:
            logger.warning(f"You have been disconnected from the server: {err}")
            print(f"Error there: {err}")
            signal = False
            end_program()
            exit(1)


def send(socket:socket.socket, message:str):
    """ Sending messages using an open socket.

    Args:
        socket (socket.socket): an open socket to send messages with.

        message (str): the message to send. It will be encoded by the function
        before sending.
    """
    socket.sendto(str.encode(message), (host_name, port_nb))
    

def request_tts(socket:socket.socket, utterances:list):
    """ Send the server the message to synthesize.

    Args:
        socket (socket.socket): an open socket to send messages with.

        utterances (list): the list of all utterances to synthesize.
    """
    global looking_for_tts
    logger.info(f"Asking the server to synthesize: {utterances}")
    looking_for_tts = True
    text_to_send = consts.TTS_PREFIX
    for items in utterances:
        text_to_send += f"{items}|"
    send(socket, text_to_send)


def tts_on_cpu(utterances:list):
    """ Fallback plan: TTS on CPU.

    Args:
        utterances (list): The text to be synthesized.
    """
    # Load models and prepare for synthesis
    inference_tts = TTSInferenceModel()
    inference_vocoder = VocoderInferenceModel()
    inference_tts.prepare_tts(consts.synth_model, consts.model_json, 0) # n_gpu=0 => cpu
    inference_vocoder.prepare_vocoder(consts.vocoder_path, consts.vocoder_conf, True) # force_cpu=True

    base_filename = str(uuid4())
    make_tts_directory_structure(base_filename)
    wavs = []
    for n, text_to_tts in enumerate(utterances):
        # Step 0 - Input prep.
        make_input_file(base_filename, text_to_tts)
        # Step 1 - Data prep.
        data_preparation(base_filename, text_to_tts)
        # Step 2 - TTS decoding
        feats_location = join(consts.outputs_dir(base_filename), "feats")
        data_json_file = join(consts.dump_dir(base_filename), "data.json")
        load_inputs_and_targets, js = inference_tts.prepare_decoding(data_json_file, feats_location)
        inference_tts.decode(feats_location, load_inputs_and_targets, js)
        # Step 3/4 - Vocoder
        feats_dir=consts.outputs_dir(base_filename)
        feats_file = join(consts.outputs_dir(base_filename), f"feats.scp")
        wav_file = inference_vocoder.decode(feats_file, consts.wav_wnv_dir(base_filename), n)
        # Recording result location
        wavs.append(wav_file)
    wav_file = concat_synth_files(wavs, base_filename)
    logger.info(f"File {wav_file} has been generated")
    print(f"File {wav_file} has been generated on client side after Server side failed to respond on time.")

# Program input variables - Using click
@click.command()
# @click.option('--test', default=1, help='TODO')
@click.argument('input_utterance', type=str)
@click.option('--host', type=str, default=consts.SERVER_HOST, help='The host ip address at which the server is running. Default is localhost.')
@click.option('--port', type=int, default=consts.SERVER_PORT, help='The port on which the server is listening, has to be in [1;65535] (0 is reserved).')
@click.option('--streaming/--no-streaming', default=consts.streaming_on, help='Enable streaming back the audio to the client. Activate this option if client and server are on different machines or cannot access the same volumes. [Default: Streaming activated].')
@click.option('--save-as', type=str, default="", help='Provide a name/path for the synthesized audio file. Works only if streaming is activated (as otherwise the file is saved by the server app).')
@click.option('--sample-rate', type=int, help='Set a custom sampling frequency when saving audio on client side.')
def main(input_utterance:str, host:str, port:int, streaming:bool, save_as:str, sample_rate:str):
    """ Main body of the client process. Setups configuration and starts the
    server socket & the associated communication thread. Then, the request for
    tts is processed and made.
    """
    global tts_ack
    global host_name
    global port_nb
    global streaming_activated
    global final_fn
    global sampling_frequency
    host_name = host
    port_nb = port
    streaming_activated = streaming
    final_fn = save_as
    sampling_frequency = sample_rate

    #Attempt connection to server
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host_name, port_nb))
    except:
        logger.error("Could not make a connection to the server")
        input("Press enter to quit")
        exit(0)
    
    #Create new thread to wait for data
    receiveThread = threading.Thread(target = receive, args = (sock, True))
    receiveThread.start()
    
    #Send data to server
    #str.encode is used to turn the string message into bytes so it can be sent across the network
    try:
        # Ask the TTS frontend to preprocess the input text utterance.
        logger.debug("Pre-processing the input text to synthesize.")
        preprocessed_utterances = pre_process_text_for_tts(input_utterance)
        logger.debug("Now asking the server for TTS.")
        request_tts(sock, preprocessed_utterances)
        sleep(consts.REQ_TIMEOUT)
        if not tts_ack:
            # TTS on CPU is triggered if the request times out without
            # hearing back from the server.
            send(sock, consts.END_MSG)
            logger.warning("No news from the server, triggering TTS on CPU.")
            tts_on_cpu(preprocessed_utterances, save_as)
            end_program()
    except (KeyboardInterrupt, ValueError):
            send(sock, consts.END_MSG)
            end_program()


if __name__ == "__main__":
    main()

