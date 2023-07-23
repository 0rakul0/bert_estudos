"""

Esse modelo tem o objetivo de fazer resumos

"""
import transformers
from transformers import pipeline

sumarizador = pipeline('summarization')

texto = """PARTE GERAL LIVRO I DAS NORMAS PROCESSUAIS CIVIS    TÍTULO ÚNICO    DAS NORMAS FUNDAMENTAIS E DA APLICAÇÃO DAS NORMAS PROCESSUAIS   CAPÍTULO I  DAS NORMAS FUNDAMENTAIS DO PROCESSO CIVIL   Art. 1º O processo civil será ordenado, disciplinado e interpretado conforme os valores e as normas fundamentais estabelecidos na Constituição da República Federativa do Brasil , observando-se as disposições deste Código.Art. 2º O processo começa por iniciativa da parte e se desenvolve por impulso oficial, salvo as exceções previstas em lei.Art. 3º Não se excluirá da apreciação jurisdicional ameaça ou lesão a direito.§ 1º É permitida a arbitragem, na forma da lei.§ 2º O Estado promoverá, sempre que possível, a solução consensual dos conflitos.§ 3º A conciliação, a mediação e outros métodos de solução consensual de conflitos deverão ser estimulados por juízes, advogados, defensores públicos e membros do Ministério Público, inclusive no curso do processo judicial.Art. 4º As partes têm o direito de obter em prazo razoável a solução integral do mérito, incluída a atividade satisfativa.Art. 5º Aquele que de qualquer forma participa do processo deve comportar-se de acordo com a boa-fé.Art. 6º Todos os sujeitos do processo devem cooperar entre si para que se obtenha, em tempo razoável, decisão de mérito justa e efetiva.Art. 7º É assegurada às partes paridade de tratamento em relação ao exercício de direitos e faculdades processuais, aos meios de defesa, aos ônus, aos deveres e à aplicação de sanções processuais, competindo ao juiz zelar pelo efetivo contraditório.Art. 8º Ao aplicar o ordenamento jurídico, o juiz atenderá aos fins sociais e às exigências do bem comum, resguardando e promovendo a dignidade da pessoa humana e observando a proporcionalidade, a razoabilidade, a legalidade, a publicidade e a eficiência.Art. 9º Não se proferirá decisão contra uma das partes sem que ela seja previamente ouvida.Parágrafo único. O disposto no caput não se aplica:    I - à tutela provisória de urgência;II - às hipóteses de tutela da evidência previstas no art. 311, incisos II e III ;III - à decisão prevista no art. 701  
"""

resumo = sumarizador(texto, max_length=100, min_length=50)

print(resumo)