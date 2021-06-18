# Main imports
from __future__ import print_function
from __future__ import division

import json
import logging
import soundfile
import yaml

from os import makedirs
from argparse import Namespace
from os.path import dirname, exists, join
from time import time

# define device
import torch
import numpy as np
from tqdm import tqdm

# Kaldi resources
import kaldiio

# Espnet resources
# Adding the path needed to import Espnet
import sys
sys.path.append("espnet")
import espnet #TODO, only take what's necessary
from espnet.nets.pytorch_backend.e2e_tts_tacotron2 import Tacotron2
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import torch_load
from espnet.utils.dynamic_import import dynamic_import
from espnet.nets.tts_interface import TTSInterface
from espnet.utils.io_utils import LoadInputsAndTargets

# Parallel Wavegan Resources
import parallel_wavegan.models
from parallel_wavegan.datasets import MelSCPDataset
from parallel_wavegan.layers import PQMF

# Local resources
import clean_text

# Declare logger
logger = logging.getLogger("server_pytorch_resources") #TODO .addHandler(logging.FileHandler("server_pytorch_resources.log")) # .addHandler(logging.NullHandler())

#############################################################################################
#                         Don't use outside the scope of this file                          #
#############################################################################################
# define function to calculate focus rate
# (see section 3.3 in https://arxiv.org/abs/1905.09263)
def _calculate_focus_rete(att_ws):
    if att_ws is None:
        # fastspeech case -> None
        return 1.0
    elif len(att_ws.shape) == 2:
        # tacotron 2 case -> (L, T)
        return float(att_ws.max(dim=-1)[0].mean())
    elif len(att_ws.shape) == 4:
        # transformer case -> (#layers, #heads, L, T)
        return float(att_ws.max(dim=-1)[0].mean(dim=-1).max())
    else:
        logger.error("att_ws should be 2 or 4 dimensional tensor.")
        raise ValueError


# define function to convert attention to duration
def _convert_att_to_duration(att_ws):
    if len(att_ws.shape) == 2:
        # tacotron 2 case -> (L, T)
        pass
    elif len(att_ws.shape) == 4:
        # transformer case -> (#layers, #heads, L, T)
        # get the most diagonal head according to focus rate
        att_ws = torch.cat(
            [att_w for att_w in att_ws], dim=0
        ) # (#heads * #layers, L, T)
        diagonal_scores = att_ws.max(dim=-1)[0].mean(dim=-1)  # (#heads * #layers,)
        diagonal_head_idx = diagonal_scores.argmax()
        att_ws = att_ws[diagonal_head_idx]  # (L, T)
    else:
        logger.error("att_ws should be 2 or 4 dimensional tensor.")
        raise ValueError
    # calculate duration from 2d attention weight
    durations = torch.stack(
        [att_ws.argmax(-1).eq(i).sum() for i in range(att_ws.shape[1])]
    )
    return durations.view(-1, 1).float()
#############################################################################################


class TTSInferenceModel:
    def __init__(self):
        """ Constructor.
        """
        self.model = None
        self.idim = None
        self.odim = None
        self.train_args = None
        self.inference_args = Namespace(**{
            "threshold": 0.5,"minlenratio": 0.0, "maxlenratio": 10.0,
            # Only for Tacotron 2
            "use_attention_constraint": True, "backward_window": 1,"forward_window":3,
            # Only for fastspeech (lower than 1.0 is faster speech, higher than 1.0 is slower speech)
            "fastspeech_alpha": 1.0,
        })

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")


    def cleanup(self):
        """ Simply a function that deleted useless data from memory.
        """
        torch.cuda.empty_cache()


    def prepare_tts(self, model_weights:str, model_conf:str, n_gpu:int):
        """Preprocessing to be done prior to inference with the tts model.

        Args: 
            model_weights (str): the save file with trained weights. 
            model_conf (str):
            n_gpu (int): 
        """
        preprocessing_args = Namespace(**{
            "seed": 1,"debugmode": 1,
        })

        # Instanciate model and load weights
        set_deterministic_pytorch(preprocessing_args)
        self.idim, self.odim, self.train_args = get_model_conf(model_weights, model_conf)
        model_class = dynamic_import(self.train_args.model_module)
        model = model_class(self.idim, self.odim, self.train_args)
        assert isinstance(model, TTSInterface)
        torch_load(model_weights, model)
        model.eval()

        # Forcing cpu mode if requested
        if n_gpu == 0:
            self.device = torch.device("cpu")

        # Loading the final model
        self.model = model.to(self.device)

    def prepare_decoding(self, json_file:str, output:str, preprocess_conf:str=None):
        """Prepare the file structure/data for decoding.

        Args:
            json_file (str): 
            output (str): 
            preprocess_conf (str): 
        """
        # read json data
        with open(json_file, "rb") as f:
            js = json.load(f)["utts"]

        # check directory
        outdir = dirname(output)
        if len(outdir) != 0 and not exists(outdir):
            makedirs(outdir)

        load_inputs_and_targets = LoadInputsAndTargets(
                mode="tts",
                load_input=False,
                sort_in_input_length=False,
                use_speaker_embedding=self.train_args.use_speaker_embedding,
                preprocess_conf=self.train_args.preprocess_conf if preprocess_conf is None else preprocess_conf,
                preprocess_args={"train": False},  # Switch the mode of preprocessing
        )

        return load_inputs_and_targets, js


    def decode(self, output:str, load_inputs_and_targets, js):
        """ Decoding related. Happens once prepare_decoding has been run.

        Args: 
            output (str): TODO
        """
        # define writer instance. In this example, we only really care about output
        # features.
        feat_writer = kaldiio.WriteHelper(f"ark,scp:{output}.ark,{output}.scp")

        if len(js) > 1:
            logger.error(f"There should only be one sentence to produce but the json dictionnary contains multiple ({len(js)}) keys.")
            raise ValueError
        
        utt_id = list(js.keys())[0]
        batch = [(utt_id, js[utt_id])]
        
        data = load_inputs_and_targets(batch)
        
        x = torch.LongTensor(data[0][0]).to(self.device)
        spemb = None
        # no use of speaker embeddings in the current scenario.

        # decode and write
        start_time = time()
        outs, probs, att_ws = self.model.inference(x, self.inference_args, spemb=spemb)
        stop_time = time()
        # NOTE Added a detach(). Need to investigate if this isn't the
        # sign that there is an error elsewhere.
        feat_writer[utt_id] = outs.cpu().detach().numpy() 
        # close file object
        feat_writer.close()
        del feat_writer
        self.cleanup()

    def _alternate_load_tts_model(self, model_path:str, dict_path:str, trans_type:str):
        """ Loads the Tacotron 2 TTS model to memory and returns it, ready for
        predictions.
        
        Args:
            model_path (str):
            dict_path (str):
            trans_type (str):
        
        Returns:
            Tacotron2 instance.
        """
        idim, odim, train_args = get_model_conf(model_path)
        model_class = dynamic_import(train_args.model_module)
        model = model_class(idim, odim, train_args)
        
        torch_load(model_path, model)
        model = model.eval().to(self.device)
        inputs = np.random.randint(1, size=(30))
        inputs = torch.from_numpy(inputs)
        inference_args = Namespace(**{
            "threshold": 0.5,"minlenratio": 0.0, "maxlenratio": 10.0,
            # Only for Tacotron 2
            "use_attention_constraint": True, "backward_window": 1,"forward_window":3,
        })
        return model


class VocoderInferenceModel:

    def __init__(self):
        """ Constructor.
        """
        self.vocoder = None
        self.use_noise_input = False # Not very important to manage that
                                     # variable but I integrate it nonetheless
                                     # in case future models require it to be
                                     # turned on/off.
        self.config=None

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.inference_args = None
        self.pad_fn = None


    def cleanup(self):
        """ Simply a function that deleted useless data from memory.
        """
        torch.cuda.empty_cache()


    def prepare_vocoder(self, vocoder_path:str, vocoder_conf:str, force_cpu:bool=False):
        """ Preprocessing to be done prior to inference with the vocoder model.

        Args:
            vocoder_path (str): TODO
            vocoder_conf (str): TODO
        """
        
        # Overriding device settings if requested
        if force_cpu:
            self.device = torch.device("cpu")

        inference_args = Namespace(**{
            "feats-scp": None, "dumpdir": None,"outdir": None,
            "config": None, "checkpoint": vocoder_path, "verbose": 1
        })

        # Loading the model configuration file
        with open(vocoder_conf) as vocoder_conf_file:
            self.config = yaml.load(vocoder_conf_file, Loader=yaml.Loader)
            self.config.update(vars(inference_args))
        

        # Setting up the environment for generation
        model_class = getattr(
            parallel_wavegan.models,
            self.config.get("generator_type", "ParallelWaveGANGenerator")
        )
        model = model_class(**self.config["generator_params"])
        model.load_state_dict(
            torch.load(vocoder_path, map_location="cpu")["model"]["generator"]
        )
        model.remove_weight_norm()
        model = model.eval().to(self.device)
        self.use_noise_input = not isinstance(model, 
                                         parallel_wavegan.models.MelGANGenerator
        )
        self.pad_fn = torch.nn.ReplicationPad1d(
                self.config["generator_params"].get("aux_context_window", 0)
        )
        if self.config["generator_params"]["out_channels"] > 1:
            pqmf = PQMF(self.config["generator_params"]["out_channels"]).to(self.device)

        self.vocoder = model


    def decode(self, scp_feats_file:str, outdir_path:str, nb:int) -> str:
        """ Decode function. Reads input features from the scp file generated
        by the tts model.
        
        Args:
            scp_feats_file (str): feats.scp file location.
        
        Returns:
        str: path to the generated file.
        """
        total_rtf = 0

        dataset = MelSCPDataset(
            feats_scp=scp_feats_file,
            return_utt_id=True,
        )

        with torch.no_grad(), tqdm(dataset, desc="[decode]") as pbar:
            for idx, (utt_id, c) in enumerate(pbar, 1):
                # setup input
                x = ()
                if self.use_noise_input:
                    z = torch.randn(1, 1, len(c) * self.config["hop_size"]).to(self.device)
                    x += (z,)
                c = self.pad_fn(torch.tensor(c, dtype=torch.float).unsqueeze(0).transpose(2, 1)).to(self.device)
                x += (c,)

                # generate
                start = time()
                if self.config["generator_params"]["out_channels"] == 1:
                    y = self.vocoder(*x).view(-1).cpu().numpy()
                else:
                    y = pqmf.synthesis(self.vocoder(*x)).view(-1).cpu().numpy()
                rtf = (time() - start) / (len(y) / self.config["sampling_rate"])
                pbar.set_postfix({"RTF": rtf})
                total_rtf += rtf
                # save audio as PCM 16 bit wav file, sample rate is that used
                # to train the model (typically 16 or 22.05kHz).
                wav_name = join(outdir_path, f"{utt_id}_{nb}_gen.wav")
                soundfile.write(wav_name, y, self.config["sampling_rate"], "PCM_16")
        logger.info(f"Total rtf: {total_rtf}")

        self.cleanup()
        return wav_name


    def _alternate_load_vocoder_model(self, vocoder_path:str, vocoder_conf:str):
        """ Loads the vocoder model to memory and returns the loaded model, ready to
        do predictions.
    
        Args:
            vocoder_path (str): the path to to the model weights file.
            vocoder_conf (str): the path to the model configuration file.
    
        Returns:
            vocoder_class instance.
        """
        with open(vocoder_conf) as f:
            config = yaml.load(f, Loader=yaml.Loader)
        vocoder_class = config.get("generator_type", "ParallelWaveGANGenerator")
        vocoder = getattr(parallel_wavegan.models, vocoder_class)(**config["generator_params"])
        vocoder.load_state_dict(torch.load(vocoder_path, map_location="cpu")["model"]["generator"])
        vocoder.remove_weight_norm()
        vocoder = vocoder.eval().to(device)
        if config["generator_params"]["out_channels"] > 1:
            pqmf = PQMF(config["generator_params"]["out_channels"]).to(device)
        
        return vocoder

