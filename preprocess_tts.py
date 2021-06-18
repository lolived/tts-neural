""" This python script reads some text on stdin, preprocesses it so that it can
be used as input for the Tacotron2 TTS script. The preprocessed text is output
on stdout.
"""
import logging
import nltk
import string

from sys import stdin

logger = logging.getLogger("preprocess_tts")

# The typical sentence is about 15 words long in modern English prose and shorter in spoken language. Very old books (pre-19th century) may contain much longer sentences up to 60 words in average. So about 100 words per sentence really constitutes an extremely long sentence.
MAX = 100

def pre_process_text_for_tts(utterance:str) -> list:
    """ TODO

    Args:
        utterance (str): TODO 

    Returns:
        list: TODO
    """
    processed_utt = []

    # Upper case everything
    utterance = utterance.upper()

    # NOTE Force add weak punctuation in long utterances that have none?

    # Break the utterance based on strong punctuation.
    sentences = nltk.sent_tokenize(utterance)

    # Normalize each text utterance separately.
    for sentence in sentences:
        # TODO Do a (much) better job at replacing problematic text characters.
        sentence = sentence.replace("-"," ")
        sentence = sentence.replace("_"," ")
        sentence = sentence.replace("|"," ")
        sentence = sentence.replace("\\"," ")
        sentence = sentence.replace("/"," ")
        sentence = sentence.replace("\'",", ")
        sentence = sentence.replace("\"",", ")
        sentence = sentence.replace("—", ", ")
        sentence = sentence.replace(" ,", ",")
        sentence = sentence.strip()
        sentence = sentence.replace("  ", " ")
        # Force add (strong) punctuation at the end.
        # We make the STRONG assumption that if there is no punctuation, the
        # sentence is declarative.
        if len(sentence.split(" ")) > MAX:
            nb_words = len(sentence.split(" "))
            logger.error(f"Error! You are asking for a very long sentence to be synthesized: {nb_words} words. This is most probably incorrect. Bear in mind that most TTS models are not designed to synthesize this type of unusually extreme sentences so the result might be incorrect.")
            raise ValueError(f"Error! The requested sentence is much too long to correspond to any plausible use case at {nb_words} words. Please enter meaningful sentences with correct punctuation.")
        if not sentence[-1] in string.punctuation:
            sentence = sentence+"."

        processed_utt.append(sentence)

    return processed_utt


def main(data:list):
    """ TODO

    Args:
        data (str): TODO
    """
    for line in data:
        line_to_process = line.rstrip()

        processed_text = pre_process_text_for_tts(line_to_process)

        for p_line in processed_text:
            print(p_line)


if __name__ == "__main__":
    main(stdin.readlines())

