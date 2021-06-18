#!/usr/bin/env python3

# Modified by David Guennec in oct-nov 2020 for IRISA
# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import codecs
import nltk

import constants as consts
from tacotron_cleaner.cleaners import custom_english_cleaners

try:
    # For phoneme conversion, use https://github.com/Kyubyong/g2p.
    from g2p_en import G2p
    
    f_g2p = G2p()
    f_g2p("")
except ImportError:
    raise ImportError(
        "g2p_en is not installed. please run `. ./path.sh && pip install g2p_en`."
    )
except LookupError:
    # NOTE: we need to download dict in initial running
    nltk.data.path.append(consts.nltk_path)
    nltk.download('punkt', download_dir=consts.nltk_path)


def g2p(text):
    """Convert grapheme to phoneme."""
    tokens = filter(lambda s: s != " ", f_g2p(text))
    return " ".join(tokens)

def clean_and_phonetize_text(text:str):
    clean_content = custom_english_cleaners(text.rstrip())
    clean_text = clean_content.lower()
    phonetized_content = g2p(clean_text)
    return phonetized_content

def clean_text(text, txt, trans_type):
    clean_data = {}
    with codecs.open(text, "r", "utf-8") as fid:
        for line in fid.readlines():
            if txt:
                id = ""
                content = line
            else:
                id, _, content = line.split("|")
            clean_content = custom_english_cleaners(content.rstrip())
            if trans_type == "phn":
                text = clean_content.lower()
                clean_content = g2p(text)

            clean_data[id]=clean_content
    return clean_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=str, help="text to be cleaned")
    parser.add_argument(
            "trans_type",
            type=str,
            default="kana",
            choices=["char", "phn"],
            help="Input transcription type",
            )
    parser.add_argument(
            "--txt",
            default=False,
            action='store_true',
            help="If true, considers the input as text and not csv",
            )
    args = parser.parse_args()
    
    cleaned_data = clean_text(args.text, args.txt, args.trans_type)
    for id in cleaned_data:
        print("%s %s" % (id, cleaned_data[id]))

