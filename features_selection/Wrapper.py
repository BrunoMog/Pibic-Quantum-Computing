import numpy as np
import pandas as pd
import ast
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score
from typing import List, Union, Callable
from joblib import load


class ForwardFeatureSelector:
    def __init__(
        self,
        model: Union[BaseEstimator, nn.Module],
        model_type: str = 'sklearn',  # 'sklearn' or 'pytorch'
        n_features_to_select: int = None,
        scoring: Union[str, Callable] = 'accuracy',
        cv: int = 5,
        verbose: int = 0,
        # Parâmetros específicos para PyTorch
        pytorch_train_func: Callable = None,
        pytorch_eval_func: Callable = None,
        pytorch_criterion: Callable = nn.CrossEntropyLoss(),
        pytorch_optimizer: Callable = None,
        epochs: int = 100,
        batch_size: int = 32,
        classification = 1,
        maior_score = 1,
        device: str = 'cpu'
    ):

        self.model_type = model_type
        self.n_features_to_select = n_features_to_select
        self.scoring = scoring
        self.kf = KFold(n_splits=cv, shuffle=True, random_state=49)
        self.verbose = verbose
        self.selected_features = []
        self.best_scores = []
        
        # Configurações específicas para PyTorch
        self.pytorch_train_func = pytorch_train_func
        self.pytorch_eval_func = pytorch_eval_func
        self.pytorch_criterion = pytorch_criterion
        self.pytorch_optimizer = pytorch_optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.classification = classification
        self.maior_score = maior_score

        if model_type == 'pytorch':
            if not callable(model):
                raise ValueError("Para modelos PyTorch, o argumento 'model' deve ser uma função (factory) que recebe input_dim.")
            self.model_factory = model
        else:
            self.model = model  # sklearn: já é instância

    def _evaluate_model(
        self, 
        X: Union[np.ndarray, torch.Tensor], 
        y: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """Avalia o modelo usando validação cruzada"""
        if self.model_type == 'sklearn':
            scores = cross_val_score(
                self.model, X, y, 
                scoring=self.scoring, cv=self.kf
            )
            return np.mean(scores)
        
        elif self.model_type == 'pytorch':

            # Implementação customizada para PyTorch
            fold_scores = []

            
            # X deve ser um tensor caso não, será transformado
            if isinstance(X, np.ndarray):
                X_tensor = torch.FloatTensor(X)

            # y também deve ser um tensor
            if isinstance(y, np.ndarray):
                if self.classification:
                    y_tensor = torch.LongTensor(y)
                else:
                    y_tensor = torch.FloatTensor(y)
            
            for fold, (train_idx, val_idx) in enumerate(self.kf.split(X)):
                
                model = self.model_factory(len(X[0]))

                #separando o X e o y no conjunto de treino e validação
                X_train, X_val = X_tensor[train_idx], X_tensor[val_idx]
                y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]

                # criando o dataset
                train_dataset = TensorDataset(X_train, y_train)
                val_dataset = TensorDataset(X_val, y_val)

                #criando o dataloader
                train_loader = DataLoader(train_dataset, batch_size=self.batch_size)
                val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

                model.to(self.device)
                
                criterion = self.pytorch_criterion
                optimizer = self.pytorch_optimizer(model.parameters(), lr = 0.01)

                for epoch in range(self.epochs):

                    #treinamento do modelo
                    model.train()
                    train_loss = 0.0

                    for X_batch, y_batch in train_loader:
                        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                        y_batch = y_batch.squeeze()

                        optimizer.zero_grad()
                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item()
                                    
                # Validação
                model.eval()
                with torch.no_grad():

                    y_pred = []
                    y_true = []
                    for X_val, y_val in val_loader:
                        X_val, y_val = X_val.to(self.device), y_val.to(self.device)
                        y_val = y_val.squeeze()  # Garante que y_val é 1D

                        outputs = model(X_val)
                        probabilities = torch.softmax(outputs, dim=1)

                        # se for classificação o meu y_pred será um valor fixo indicando o melhor ansatz
                        if self.classification:
                            y_pred.extend(probabilities.argmax(dim=1).cpu().numpy())

                        else:
                            y_pred.extend(outputs.cpu().numpy())

                        y_true.extend(y_val.cpu().numpy())


                score = self.scoring(y_true=y_true, y_pred=y_pred)
                fold_scores.append(score)

            return np.mean(fold_scores)
        
        else:
            raise ValueError("Model type must be 'sklearn' or 'pytorch'")
        
    def _get_best_feature(
        self, 
        X: Union[np.ndarray, torch.Tensor], 
        y: Union[np.ndarray, torch.Tensor], 
        candidate_features: List[int]
    ) -> int:
        """Encontra a melhor feature para adicionar"""
        if self.maior_score:
            best_score = -np.inf
        else:
            best_score = np.inf
        best_feature = None
        
        for feature in candidate_features:
            current_features = self.selected_features + [feature]
            
            
            score = self._evaluate_model(X[:, current_features], y)

            if self.verbose > 0:
                print(f"Testing feature set: {current_features}, score: {score}")
            
            #tem métricas como MSE que desejam ser minimizadas para regressão
            if self.maior_score:
                if score > best_score:
                    best_score = score
                    best_feature = feature

            else:
                if score < best_score:
                    best_score = score
                    best_feature = feature                

        print(best_feature)
        print(best_score)    
        return best_feature, best_score

    def fit(
        self, 
        X: Union[np.ndarray, torch.Tensor], 
        y: Union[np.ndarray, torch.Tensor]
    ):
        """
        Executa o algoritmo de Forward Selection
        Testar os subconjuntos de features até preencher o número de features desejadas
        Mas só seleciona a feature testada se o desempenho dela for maior do que o desempenho anterior
        """
        n_samples, n_features = X.shape

        if self.n_features_to_select is None:
            self.n_features_to_select = n_features
            
        remaining_features = list(range(n_features))
        
        #controlar a parada do loop para casos em quem nenhuma feature restante melhore o desempenho do modelo
        _flag = True

        for _ in range(self.n_features_to_select):

            if _flag:

                best_feature, best_score = self._get_best_feature(X, y, remaining_features)
                
                remaining_features.remove(best_feature)

                # maximizar a métrica de desempenho
                if self.maior_score:

                    if len(self.best_scores) > 0 and best_score > self.best_scores[-1]:

                        self.selected_features.append(best_feature)
                        self.best_scores.append(best_score)

                        
                        if self.verbose > 0:
                            print(f"Selected feature: {best_feature} | Score: {best_score:.4f}")
                    
                    elif len(self.best_scores) == 0:

                        self.selected_features.append(best_feature)
                        self.best_scores.append(best_score)
                        
                        if self.verbose > 0:
                            print(f"Selected feature: {best_feature} | Score: {best_score:.4f}")

                    else:
                        _flag = False

                # minimizar a métrica de desempenho
                else:

                    if len(self.best_scores) > 0 and best_score < self.best_scores[-1]:

                        self.selected_features.append(best_feature)
                        self.best_scores.append(best_score)

                        
                        if self.verbose > 0:
                            print(f"Selected feature: {best_feature} | Score: {best_score:.4f}")
                    
                    elif len(self.best_scores) == 0:

                        self.selected_features.append(best_feature)
                        self.best_scores.append(best_score)
                        
                        if self.verbose > 0:
                            print(f"Selected feature: {best_feature} | Score: {best_score:.4f}")

                    else:
                        _flag = False


        return self

    def transform(self, X: Union[np.ndarray, torch.Tensor]):
        """Seleciona as features escolhidas"""
        return X[:, self.selected_features]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


class BackwardFeatureSelector(ForwardFeatureSelector):
    def __init__(self, features_to_remove = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_features_to_remove = features_to_remove

    def _get_worst_feature(
        self, 
        X: Union[np.ndarray, torch.Tensor], 
        y: Union[np.ndarray, torch.Tensor], 
        remaining_features: List[int]
    ) -> int:
        
        """Encontra a pior feature para remover"""
        if self.maior_score:
            best_score = -np.inf
        else:
            best_score = np.inf
        worst_feature = None

        for feature in remaining_features:
            # Cria subconjunto sem a feature atual
            current_features = [f for f in remaining_features if f != feature]
            
            score = self._evaluate_model(X[:, current_features], y)
            
            if self.verbose > 0:
                print(f"Testando subconjunto: {current_features}, score: {score}")

            # Lógica de maximização/minimização
            if self.maior_score:
                if score > best_score:
                    best_score = score
                    worst_feature = feature
            else:
                if score < best_score:
                    best_score = score
                    worst_feature = feature

        return worst_feature, best_score

    def fit(
        self, 
        X: Union[np.ndarray, torch.Tensor], 
        y: Union[np.ndarray, torch.Tensor]
    ):
        n_samples, n_features = X.shape
        
        #remove todas as features que pioram o desepenho do modelo ou deixam ele igual
        if self.n_features_to_remove is None:
            self.n_features_to_remove = n_features
            
        remaining_features = list(range(n_features))
        self.selected_features = remaining_features.copy()

        #avalia o modelo com todas as features e salva o resultado em best_scores
        score = self._evaluate_model(X, y)
        self.best_scores.append(score)

        _flag = True
        
        for _ in range(self.n_features_to_remove):
            if _flag:
                worst_feature, best_score = self._get_worst_feature(X, y, remaining_features)
                
                # Atualiza lista
                remaining_features.remove(worst_feature)                

                # maximizar a métrica de desempenho
                if self.maior_score:

                    if  best_score > self.best_scores[-1]:

                        self.selected_features.remove(worst_feature)
                        self.best_scores.append(best_score)
                        
                        if self.verbose > 0:
                            print(f"Removed feature: {worst_feature} | Score: {best_score:.4f}")

                    else:
                        _flag = False
                
                #minimizar a métrica de desempenho
                else:

                    if  best_score < self.best_scores[-1]:

                        self.selected_features.remove(worst_feature)
                        self.best_scores.append(best_score)
                        
                        if self.verbose > 0:
                            print(f"Removed feature: {worst_feature} | Score: {best_score:.4f}")

                    else:
                        _flag = False

        return self



"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Exemplo de uso para sklearn
if __name__ == "__main__":
 
    
    # abrindo os dados de treinamento
    df = pd.read_csv("val/data_val.csv")
    X = df.drop(columns=["target"]).to_numpy()
    y = pd.DataFrame(df['target'].apply(ast.literal_eval).tolist()).to_numpy()

    # Para cada amostra, identificar o ansatz com maior acurácia
    y_best_ansatz = np.argmax(y, axis=1)  # Retorna índices 0-29

    dt_clf = load('models/models_salvos/dt_classifier.joblib')
    gb = load('models/models_salvos/gb_clf.joblib')

    # Seleção de features
    selector = ForwardFeatureSelector(
        model=dt_clf,
        model_type='sklearn',
        cv=3,
        verbose=1
    )
    
    X_new = selector.fit_transform(X, y_best_ansatz)
"""

