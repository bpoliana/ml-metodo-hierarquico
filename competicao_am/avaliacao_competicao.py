from base_am.avaliacao import OtimizacaoObjetivo
from base_am.metodo import MetodoAprendizadoDeMaquina
from base_am.resultado import Fold, Resultado
from competicao_am.metodo_competicao import MetodoHierarquico
import optuna
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


class OtimizacaoObjetivoSVMCompeticao(OtimizacaoObjetivo):
    def __init__(self, fold:Fold):
        super().__init__(fold)

    def obtem_metodo(self,trial: optuna.Trial)->MetodoAprendizadoDeMaquina:
        #Um custo adequado para custo pode variar muito, por ex, para uma tarefa 
        #o valor de custo pode ser 10, para outra, 32000. 
        #Assim, normalmente, para conseguir valores mais distintos,
        #usamos c=2^exp_cost
        exp_cost = trial.suggest_uniform('COST', -5, 15)

        scikit_method = LinearSVC(C=2**exp_cost, random_state=2)

        return MetodoHierarquico(scikit_method, "genre")

    def resultado_metrica_otimizacao(self,resultado: Resultado) -> float:
        return resultado.macro_f1

    
class OtimizacaoObjetivoRF(OtimizacaoObjetivo):
    def __init__(self, fold:Fold):
        super().__init__(fold)

    def obtem_metodo(self,trial: optuna.Trial)->MetodoAprendizadoDeMaquina:
        #Um custo adequado para custo pode variar muito, por ex, para uma tarefa 
        #o valor de custo pode ser 10, para outra, 32000. 
        #Assim, normalmente, para conseguir valores mais distintos,
        #usamos c=2^exp_cost
        min_samples = trial.suggest_uniform('min_samples_split', 0.1, 0.5)
        max_features = trial.suggest_uniform('max_features', 0.1, 0.9)
        num_arvores = trial.suggest_int('num_arvores', 5, 100)
        scikit_method = RandomForestClassifier(min_samples_split=min_samples,n_estimators=num_arvores,
                                            max_features=max_features,random_state=2)

        return MetodoHierarquico(scikit_method, "genre")

    def resultado_metrica_otimizacao(self,resultado: Resultado) -> float:
        return resultado.macro_f1

class OtimizacaoObjetivoKNN(OtimizacaoObjetivo):
    TAM_AMOSTRA = 100
    
    def __init__(self, fold:Fold):
        super().__init__(fold)

    def obtem_metodo(self,trial: optuna.Trial)->MetodoAprendizadoDeMaquina:
        #Um custo adequado para custo pode variar muito, por ex, para uma tarefa 
        #o valor de custo pode ser 10, para outra, 32000. 
        #Assim, normalmente, para conseguir valores mais distintos,
        #usamos c=2^exp_cost
        
        n_neighbors = trial.suggest_uniform('n_neighbors', 0.1, 0.4)
        scikit_method = KNeighborsClassifier(n_neighbors=n_neighbors*TAM_AMOSTRA)

        return MetodoHierarquico(scikit_method, "genre")

    def resultado_metrica_otimizacao(self,resultado: Resultado) -> float:
        return resultado.macro_f1