# PROJETO FINAL
# André Bernardo Martin Ramos
# N_USP: 3314580


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression

class Modelo():
    def __init__(self):
        pass

    def CarregarDataset(self, path):
        """
        Carrega o conjunto de dados a partir de um arquivo CSV.

        Parâmetros:
        - path (str): Caminho para o arquivo CSV contendo o dataset.
        
        O dataset é carregado com as seguintes colunas: SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm e Species.
        """
        self.names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
        self.df = pd.read_csv(path, names=self.names)

    def TratamentoDeDados(self):
        """
        Realiza o pré-processamento dos dados carregados.

        Sugestões para o tratamento dos dados:
            * Utilize `self.df.head()` para visualizar as primeiras linhas e entender a estrutura.
            * Verifique a presença de valores ausentes e faça o tratamento adequado.
            * Considere remover colunas ou linhas que não são úteis para o treinamento do modelo.
        
        Dicas adicionais:
            * Explore gráficos e visualizações para obter insights sobre a distribuição dos dados.
            * Certifique-se de que os dados estão limpos e prontos para serem usados no treinamento do modelo.
        """
        
        print("--- Iniciando tratamento de dados")
        # Vou verificar se os dados necessitam de tratamento
        count_invalidos = 0
        for col in self.names:
            count_nan = self.df[col].isna().sum()
            count_zero = (self.df[self.df[col]==0]).shape[0]
            count_null = self.df[col].isna().sum()
            if (col != 'Species'):   
                # As colunas diferentes de Species são numéricas. Vamos contar valores não numéricos ou iguais a zero             
                print(f"Coluna: {col:13}  -  NaN:  {count_nan}  -  Zeros: {count_zero}")
                count_invalidos = count_invalidos + count_nan + count_zero
            else:
                # A coluna Species tem texto da espécie. Vamos apenas verificar se o conteúdo é não nulo       
                print(f"Coluna: {col:13}  -  Null: {count_null}")
                count_invalidos = count_invalidos + count_null
        
        # Concluindo...
        if count_invalidos > 0:
            # Se há valores inválidos, gero exceção
            raise Exception("Erro! É preciso tratar dados")
        else:
            # Se valores estão válidos, informo isso
            print("--- Dados Ok! Prosseguindo...")




    def Treinamento(self):
        """
        Treina o modelo de machine learning.

        Detalhes:
            * Utilize a função `train_test_split` para dividir os dados em treinamento e teste.
            * Escolha o modelo de machine learning que queira usar. Lembrando que não precisa ser SMV e Regressão linear.
            * Experimente técnicas de validação cruzada (cross-validation) para melhorar a acurácia final.
        
        Nota: Esta função deve ser ajustada conforme o modelo escolhido.
        """

        print("--- Iniciando divisão entre dados de treino e de testes, e treinando 2 modelos")

        # Dividindo dados entre conjuntos de treinamento e de testes
        X = self.df[self.names[:-1]]   # Todas as colunas antes da última
        y = self.df['Species']         # A coluna Species (última)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.1, random_state=99) # 10% dos dados separados para teste
        
        
        print("Iniciando treinamento com modelo linear")
        self.classificador_linear = SVC(kernel='linear')           # modelo linear
        self.classificador_linear.fit(self.X_train, self.y_train)  # classificando com conjuntos de treinamento
        

        print("Iniciando treinamento com modelo rbf")
        self.classificador_rbf = SVC(kernel='rbf')                 # modelo rfb
        self.classificador_rbf.fit(self.X_train, self.y_train)     # classificando com conjuntos de treinamento
        



    def Teste(self):
        """
        Avalia o desempenho do modelo treinado nos dados de teste.

        Esta função deve ser implementada para testar o modelo e calcular métricas de avaliação relevantes, 
        como acurácia, precisão, ou outras métricas apropriadas ao tipo de problema.
        """

        # Avaliando pelo Score
        print("--- Iniciando testes dos dois modelos")        
        print(f'Taxa de acerto do modelo linear: {round(100*self.classificador_linear.score(self.X_test, self.y_test), 2)}%')
        print(f'Taxa de acerto do modelo rbf: {round(100*self.classificador_rbf.score(self.X_test, self.y_test), 2)}%')

        
        # Vou montar novo dataframe incluindo colunas geradas pelos classificadores
        print(f"--- Montando novo dataframe com a espécie real e aquela projetada pelo algoritmo, e comparando")       
        df = self.df.copy()
        df['Classificador_Linear'] = df.apply(lambda row: self.classificador_linear.predict(pd.DataFrame([[row[self.names[0]], row[self.names[1]], row[self.names[2]], row[self.names[3]]]], columns=self.names[0:4]))[0], axis=1)
        df['Classificador_Rbf'] = df.apply(lambda row: self.classificador_rbf.predict(pd.DataFrame([[row[self.names[0]], row[self.names[1]], row[self.names[2]], row[self.names[3]]]], columns=self.names[0:4]))[0], axis=1)

        # Imprimo discrepâncias
        print("--- Discrepâncias do classificador linear")
        print(df[df['Species']!=df['Classificador_Linear']].drop(columns=['Classificador_Rbf']))        
        print("--- Discrepâncias do classificador rbf")
        print(df[df['Species']!=df['Classificador_Rbf']].drop(columns=['Classificador_Linear']))        
        print("--- Discrepâncias do classificador linear")
        
        # Conclusão: os dois classificadores erraram 3 vezes cada. A linha 83 foi problemática nos dois classificadores


    def Train(self):
        """
        Função principal para o fluxo de treinamento do modelo.

        Este método encapsula as etapas de carregamento de dados, pré-processamento e treinamento do modelo.
        Sua tarefa é garantir que os métodos `CarregarDataset`, `TratamentoDeDados` e `Treinamento` estejam implementados corretamente.
        
        Notas:
            * O dataset padrão é "iris.data", mas o caminho pode ser ajustado.
            * Caso esteja executando fora do Colab e enfrente problemas com o path, use a biblioteca `os` para gerenciar caminhos de arquivos.
        """
        self.CarregarDataset("iris.data")  # Carrega o dataset especificado.

        self.TratamentoDeDados() # Já sei que é desnecessário. Na função mostra como cheguei nessa conclusão

        self.Treinamento()  # Executa o treinamento do modelo

        self.Teste()


# EXECUTA
Modelo().Train()
