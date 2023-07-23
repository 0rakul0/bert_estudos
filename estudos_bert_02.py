"""

Esse modelo tem o objetivo de preencher lacunas [MASK] dentro da frase

"""
import transformers
from transformers import pipeline

prexer_lacuna = pipeline('fill-mask', model='neuralmind/bert-base-portuguese-cased')

texto = prexer_lacuna('Tinha uma [MASK] no meio do caminho.')

for x in range(len(texto)):
    print(texto[x])

texto_lsita = prexer_lacuna(['Tinha uma [MASK] no meio do caminho.','Rio de janeiro é uma [MASK]'])

for y in range(len(texto_lsita)):
    print(texto_lsita[y])