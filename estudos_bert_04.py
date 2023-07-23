"""

Esse modelo tem o objetivo de gerar textos

"""
import transformers
from transformers import pipeline

gerador = pipeline('text-generation', model='pierreguillou/gpt2-small-portuguese')

texto = "Em sentido estrito, ciência refere-se ao sistema de adiquirir conhecimento baseado em método cientifico"

resultado = gerador(texto, max_length=100, min_length=50)

print(resultado)