# This file conducts the pre-processing. It assumes that speeches from the BIS website have been downloaded from the BIS website and converted to txt files using the code here:
# https://github.com/HanssonMagnus/scrape_bis
# There should be a folder called speeches in the root of the project with one folder per central bank.

# Text processing:
import os
import pandas as pd
from pathlib import Path
import spacy
import datetime as dt
import re
from nltk.tokenize import sent_tokenize, word_tokenize

files = [x for x in Path("../speeches/").glob("**/*") if x.is_file()]

speeches = []

for filename in files:
    speech = filename.read_text(encoding='UTF-8')
    speech_name = filename.name
    central_bank = filename.parent.name

    speeches.append(
        {
            "speech_identifier" : speech_name,
            "central_bank" : central_bank,
            "content" : speech
        }
    )

all_speeches = pd.DataFrame(speeches)
all_speeches

# Attempt to remove metadata at beginning of speech
all_speeches["content"] = all_speeches["content"].str.split(r"\*\n\n\*\n\n\*\n\n").str[1].combine_first(all_speeches["content"])
all_speeches["content"] = all_speeches["content"].str.split(r"\* \* \*\n").str[1].combine_first(all_speeches["content"])

all_speeches["content"] = (
    all_speeches["content"].
    replace(r"\n\d+\n\n", "" ,regex= True). # Page numbers
    replace(r"(\n\d\n\n(.+\n)+)+\n", r"\n" ,regex= True). # Footnotes
    replace(r"\n.*\f", "" ,regex= True). # New page characters
    replace(r"\f", "" ,regex= True). # Delete remaininig \f entries
    replace(r"\d+\[\d+\]", "" ,regex= True). # Footnote
    replace(r"(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b", "" ,regex= True). # URLs
    replace(r"BIS Review \d+/\d+", "", regex= True).
    replace(r"BIS central bankers' speeches", "", regex= True).
    replace(r"BIS central bankers. speeches", "", regex= True). 

    replace(r"(?<=[^0-9]\.)\d+", "", regex= True). # letter followed by dot followed by number. This is often a footnote
    replace(r"(?<=\.)\[\d+\]", "", regex= True). # Also footnotes like this .[digit]
    replace(r"(?<=[\!\.\?])\n\d+", r"\n", regex= True). # Sentence after line break starting with number is likely a footnote.
    replace(r"[\!\.\?]\n\d+", r"\n", regex= True).

    replace(r"\n[^\!\.\?]{1,15}\n", r"\n" ,regex= True). # Less than 15 characters in a line and nothing that termiantes a sentence.
    replace(r"\n+", r"\n" ,regex= True). # More than one line break in sequence
    replace(r"\n\.\n", r"\n" ,regex= True). # Lines that only contain a dot
    replace(r"\n\d+\.\n", r"\n", regex= True). # Lines that only contain a number followed by a dot
    replace(r"\n\d+\n", r"\n", regex= True) # Lines that only contain a digit
)

# List all central banks and loop over central banks to extract metadata
list_central_banks = all_speeches['central_bank'].unique()
list_central_banks

metadata = []

for central_bank in list_central_banks:
    metadata.append(
        {
            "metadata" : Path(f"../metadata/{central_bank}_meta.txt").read_text(encoding='UTF-8')
        } 
    )

metadata = pd.DataFrame(metadata)

# Some functions require either a spacy or nltk model to be loaded
spacy_model = spacy.load('en_core_web_lg')
nltk.download('punkt')
# Non-vectorized.
def extract_entity(text, entity):
    """Named entity recognition using spacey"""
    labelled_output = spacy_model(text)
    for ent in labelled_output.ents:
        if ent.label_ == entity:
            return(ent.text)
    return(None)

def extract_sentences(text):
    """Spacy equivalent of nltk.tokenize.sent_tokenize. Much slower however ~ 130 times"""
    labelled_output = spacy_model(text)
    return(
        [sentence.text for sentence in labelled_output.sents]
    )

#enc = tiktoken.get_encoding("cl100k_base")
def count_tokens(text):
    "Count tokens. Requires tiktoken and a enc loaded"
    try:
      length = len(enc.encode(text))
    except:
      print(f"something went wrong with the input {text}")
      length = 0
    return(length)


def convert_identifier(indent_pdf):
    order = indent_pdf[0]
    date = indent_pdf[1:] # in format "%y%m%d"
    date = dt.datetime.strptime(date, "%y%m%d") # datetime object
    date = dt.datetime.strftime(date, "%Y-%m-%d") # convert to "%Y-%m-%d"
    return(f"{date}_{order}.txt")

# Vectorized
def remove_before_pattern(series, pattern):
    """Removes anything that comes before the pattern. If the pattern is not found it preserves the series"""
    return(series.str.split(pattern).str[1].combine_first(series))

def frequency_count(series, pattern):
    """Counts the frequency of a regex pattern or word"""
    return(series.str.split(pattern).map(len) -1)

sentence_level = all_speeches.assign(
    sentence = lambda x: x["content"].astype(str).map(extract_sentences)
).drop(["content"], axis = 1).explode("sentence").dropna( #There are 34 senteces which are na.
    subset=["sentence"]
).assign(
    sentence = lambda x: x["sentence"].replace(r"\s+", " " ,regex= True),
    token_count = lambda x: x["sentence"].map(count_tokens),len = lambda x: x["sentence"].str.len(),
    ascii_letter_count = lambda x: x["sentence"].str.count('[a-zA-Z]'),
    share_ascii_letter = lambda x: x["ascii_letter_count"] / x["len"],
    fiscal = lambda x: frequency_count(x["sentence"], "fiscal"),
    financial = lambda x: frequency_count(x["sentence"], "financial"),
    monetary = lambda x: frequency_count(x["sentence"], "monetary")
)[
    lambda x : 
    (x["share_ascii_letter"] > 0.66) & 
    (x["token_count"] > 5)  & 
    (x["token_count"] < 200)  & 
    (x["len"] > 20) 
].reset_index(drop = True)


sentence_level = sentence_level.drop(["index"], axis = 1)

# Sample 1000 sentences. This is still not entirely determinstic as the sentence tokenizer seems to be non deterministic
sentence_level.drop(["content"], axis = 1).explode("sentence").sample(n = 1000, random_state=442)

# Export sentence level data:
sentence_level.to_pickle('../inputdata/sentences.pkl')




