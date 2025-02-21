import os
import spacy
from spacy_layout import spaCyLayout


def process_file(file: str, words_per_chunk_seconds=32):
    nlp = spacy.load("en_core_web_trf")

    # Extract content
    if file.endswith(".txt") and os.path.isfile(file):
        with open(file, "r") as f:
            content = f.read()
    elif (file.endswith(".pdf") or file.endswith(".xlsx") or file.endswith("docx")) and os.path.isfile(file):
        parser = spaCyLayout(nlp)
        doc = parser(file)
        content = doc.text
    elif not os.path.isfile(file):
        content = file
    else:
        raise Exception(
            "Unsupported file type, Come on there! why not, have your content in txt or pdf? it's simple and convinient isn't it?"
        )

    num_words = 0
    current_split = 0
    chunks = []

    # split te extracted content
    # later: https://github.com/segment-any-text/wtpsplit
    doc = nlp(content)
    splits = [sent.text.strip() for sent in doc.sents]

    # generate chunks with words less the word count limit
    for i in range(len(splits)):
        sent = splits[i]
        sentence_per_words = len(sent.split())
        if num_words + sentence_per_words >= words_per_chunk_seconds:
            text = " ".join(splits[current_split:i])
            chunks.append(text)
            num_words = 0
            current_split = i
        num_words += sentence_per_words

    chunks.append(" ".join(splits[current_split:]))  # flush the remaining splits
    return chunks
