from base_am.resultado import Fold
from base_am.avaliacao import Experimento, ExperimentoBOW
from competicao_am.metodo_competicao import MetodoHierarquico, MetodoTradicional
from competicao_am.avaliacao_competicao import OtimizacaoObjetivoSVMCompeticao
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def gera_experimento(df_amostra, scikit_method, classe_objetivo, n_trials):
    arr_folds = Fold.gerar_k_folds(df_amostra, val_k=5, col_classe="original_genre",
                                num_repeticoes=1, num_folds_validacao=4,num_repeticoes_validacao=1)
    print("gera_experimento")
    # arr_to_predict, arr_predictions = ml_method.eval(arr_folds[0].df_treino, arr_folds[0].df_data_to_predict, "original_genre")

#     scikit_method = LinearSVC(random_state=2)
    ml_method = MetodoHierarquico(scikit_method, "genre")

    ClasseObjetivo = classe_objetivo
    #colocamos apenas 5 trials para ir mais rápido. Porém, algumas vezes precisamos de dezenas, centenas - ou milhares - de trials para conseguir uma boa configuração
    #Isso depende muito da caracteristica do problema, da quantidade de parametros e do impacto desses parametros no resultado
    experimento = Experimento(arr_folds, ml_method=ml_method,
                        ClasseObjetivoOtimizacao=ClasseObjetivo,
                        num_trials=n_trials)# num_trial até 3 pra depois ir aumentando: 10, 100 etc 
    experimento.calcula_resultados()
    #salva
    return experimento

def gera_experimento_metodo_tradicional(df_amostra, scikit_method, classe_objetivo, n_trials):
    arr_folds = Fold.gerar_k_folds(df_amostra, val_k=5, col_classe="original_genre",
                                num_repeticoes=1, num_folds_validacao=4,num_repeticoes_validacao=1)
    print("gera_experimento")
    # arr_to_predict, arr_predictions = ml_method.eval(arr_folds[0].df_treino, arr_folds[0].df_data_to_predict, "original_genre")

#     scikit_method = LinearSVC(random_state=2)
    ml_method = MetodoTradicional(scikit_method, "original_genre")

    ClasseObjetivo = classe_objetivo
    #colocamos apenas 5 trials para ir mais rápido. Porém, algumas vezes precisamos de dezenas, centenas - ou milhares - de trials para conseguir uma boa configuração
    #Isso depende muito da caracteristica do problema, da quantidade de parametros e do impacto desses parametros no resultado
    experimento = Experimento(arr_folds, ml_method=ml_method,
                        ClasseObjetivoOtimizacao=ClasseObjetivo,
                        num_trials=n_trials)# num_trial até 3 pra depois ir aumentando: 10, 100 etc 
    experimento.calcula_resultados()
    #salva
    return experimento