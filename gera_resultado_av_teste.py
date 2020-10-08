import pandas as pd
from competicao_am.gerar_resultado_teste import gerar_saida_teste

#altere aqui para o número correspondente ao seu grupo
num_grupo = "11223344" 

#leia o dataset fornecido pelo professor (coloquei apenas um exemplo, na entrega, será outro)
df_amostra_teste = pd.read_csv("/home/profhasan/git/aulas/machine-learning/datasets_competicoes/movies/datasets/movies_amostra_teste_2.csv")

gerar_saida_teste(df_amostra_teste,"genero", num_grupo)