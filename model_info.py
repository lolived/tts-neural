"""
"""

import pathlib
from yaml import load, BaseLoader

from os.path import exists, join, abspath

# Local lib
import constants as consts

class ModelInfo():
    def __init__(self):
        """ Constuctor
        """

        # The roof of the repository is one level above the location of this
        # script (./script/server.py)
        #         ^
        self.root = join(pathlib.Path(__file__).parent.absolute(), "..")

        self.synth_model = None
        self.dict_type = None
        self.trans_type = None
        self.cmvn = None
        self.decode_config = None
        self.model_json = None
        self.vocoder_path = None
        self.vocoder_conf = None
        self.vocoder_stats = None
        self.instance_lang = None


    def load_config(self, conf:str):
        try:
            yaml_conf = open(conf, 'r')
            yaml_conf = load(yaml_conf, Loader=BaseLoader)
        except Exception as e:
            raise ValueError(f"Loading configuration file {conf} failed with the following error; please check the file exists and is valid in regard to the yaml standard: {e}")
        
        synth_model = join(self.root, yaml_conf.get("synth_model"))
        if synth_model:
            self.synth_model = synth_model
        dict_type = join(self.root, yaml_conf.get("dict_type"))
        if dict_type:
            self.dict_type = dict_type
        trans_type = yaml_conf.get("trans_type")
        if trans_type:
            self.trans_type = trans_type
        cmvn = join(self.root, yaml_conf.get("cmvn"))
        if cmvn:
            self.cmvn = cmvn
        decode_config = join(self.root, yaml_conf.get("decode_config"))
        if decode_config:
            self.decode_config = decode_config
        model_json = join(self.root, yaml_conf.get("model_json"))
        if model_json:
            self.model_json = model_json
        vocoder_path = join(self.root, yaml_conf.get("vocoder_path"))
        if vocoder_path:
            self.vocoder_path = vocoder_path
        vocoder_conf = join(self.root, yaml_conf.get("vocoder_conf"))
        if vocoder_conf:
            self.vocoder_conf = vocoder_conf
        vocoder_stats = join(self.root, yaml_conf.get("vocoder_stats"))
        if vocoder_stats:
            self.vocoder_stats = vocoder_stats
        instance_lang = yaml_conf.get("lang")
        if instance_lang:
            self.instance_lang = instance_lang


    def get_root(self):
        """ Returns the root directory of the repository.

        Returns:
            str: absolute path of the repository root directory.
        """
        return self.root


    def get_synth_model(self):
        return self.synth_model

    def set_synth_model(self, value:str):
        self.synth_model = value
    

    def get_dict_type(self):
        return self.dict_type

    def set_dict_type(self, value:str):
        self.dict_type = value


    def get_trans_type(self):
        return self.trans_type

    def set_trans_type(self, value:str):
        self.trans_type = value


    def get_cmvn(self):
        return self.cmvn

    def set_cmvn(self, value:str):
        self.cmvn = value


    def get_decode_config(self):
        return self.decode_config

    def set_decode_config(self, value:str):
        self.decode_config = value


    def get_model_json(self):
        return self.model_json

    def set_model_json(self, value:str):
        self.model_json = value


    def get_vocoder_path(self):
        return self.vocoder_path

    def set_vocoder_path(self, value:str):
        self.vocoder_path = value


    def get_vocoder_conf(self):
        return self.vocoder_conf

    def set_vocoder_conf(self, value:str):
        self.vocoder_conf = value


    def get_vocoder_stats(self):
        return self.vocoder_stats

    def set_vocoder_stats(self, value:str):
        self.vocoder_stats = value


    def get_instance_lang(self):
        return self.instance_lang

    def set_instance_lang(self, value:str):
        self.instance_lang = value


