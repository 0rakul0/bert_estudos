"""

Esse modelo tem o objetivo de gerar textos

"""
import transformers
from transformers import pipeline

gerador = pipeline('text-generation', model='pierreguillou/gpt2-small-portuguese')
