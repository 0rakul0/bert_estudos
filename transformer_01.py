import numpy as np
import pandas as pd
import re
import time
import random
import tensorflow as tf
from keras import layers
import tensorflow_datasets as tfds
from util.limpeza import limpeza
import multiprocessing as mp

## base de dados
with open('./tradutor/europarl-v7.pt-en.en', mode='r', encoding='utf-8') as f:
    europarl_en = f.read()

with open('./tradutor/europarl-v7.pt-en.pt', mode='r', encoding='utf-8') as f:
    europarl_pt = f.read()

corpus_en = europarl_en
corpus_pt = europarl_pt

corpus_en = limpeza(corpus_en)
corpus_pt = limpeza(corpus_pt)

#### tokenizer ####
#
# build_tokenize_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(corpus_en, target_vocab_size=2**13)
# build_tokenize_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(corpus_pt, target_vocab_size=2**13)
#
# build_tokenize_en.save_to_file("./vocab_tokenizer_en")
# build_tokenize_pt.save_to_file("./vocab_tokenizer_pt")

print("carregando o vocab")

tokenize_en = tfds.deprecated.text.SubwordTextEncoder.load_from_file('./vocab_tokenizer_en')
tokenize_pt = tfds.deprecated.text.SubwordTextEncoder.load_from_file('./vocab_tokenizer_pt')

print("vocab carregado")

vocab_size_en = tokenize_en.vocab_size + 2
inputs_en = [[vocab_size_en - 2] + tokenize_en.encode(sentence) + [vocab_size_en - 1] for sentence in corpus_en]

print(inputs_en[random.randint(0, len(inputs_en) - 1)])


vocab_size_pt = tokenize_pt.vocab_size + 2
outputs_pt = [[vocab_size_pt - 2] + tokenize_pt.encode(sentence) + [vocab_size_pt - 1] for sentence in corpus_pt]

print(outputs_pt[random.randint(0, len(outputs_pt) - 1)])

#### Melhorando o processamento #####
print("etapa de melhora do processamento")
"""
o Objetivo é tirar frases com mais de 15 palavras
"""

max_length = 15
idx_to_remove = [count for count, sent in enumerate(inputs_en) if len(sent) > max_length]

### mantem a lista com os mesmos indices
for idx in reversed(idx_to_remove):
    del inputs_en[idx]
    del outputs_pt[idx]

print(len(idx_to_remove))


max_length = 15
idx_to_remove = [count for count, sent in enumerate(outputs_pt) if len(sent) > max_length]

### mantem a lista com os mesmos indices
for idx in reversed(idx_to_remove):
    del inputs_en[idx]
    del outputs_pt[idx]

print(len(idx_to_remove))

#### tamanho total
print("tamanhos das entradas e saidas")
print(len(inputs_en))
print(len(outputs_pt))