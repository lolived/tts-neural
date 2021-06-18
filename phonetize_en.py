""" Code of the english phoentization block.
If an English phonetizer ends up being implemented in a sub-repository of
Tools, this code should be replaced by it.
"""

import constants as consts

def init_g2p_en():

    # Preparing NLTK for phonetization of English text.
    try:
        # For phoneme conversion, use https://github.com/Kyubyong/g2p.
        from g2p_en import G2p
    
        f_g2p = G2p()
        f_g2p("")

        return f_g2p
    except ImportError:
        raise ImportError(
            "g2p_en is not installed. please check that you are running the virtual environment and run pip install g2p_en."
        )
    except LookupError:
        # Needed only the first time. This downloads punkt English dictionnary.
        # Warning! Pay attention to where this is downloaded and check the content
        # of the NLTK_DATA environment variable in your local env.
        nltk.data.path.append(consts.nltk_path)
        nltk.download('punkt', download_dir=consts.nltk_path)


# The following functions handle the phonetization of English text sentences.
# ---------------------------------------------------------------------------
def g2p_en(text:str, f_g2p):
    """ Convert a grapheme sequence of English script to the equivalent phoneme
    sequence using NLTK.

    Args:
        text (str): the English text to phonetize.

    Returns:
        str: the phonetic string matching the input argument text.
    """
    tokens = filter(lambda s: s != " ", f_g2p(text))
    return " ".join(tokens)


