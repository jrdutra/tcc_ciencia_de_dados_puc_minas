import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import requests
import json
from datetime import date, datetime
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
import Orange
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from scipy import stats
from sklearn.model_selection import cross_val_score, KFold
from yellowbrick.classifier import ConfusionMatrix

mes_maio = '05'
dias_maio = ['01', '02', '03','04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31']
mes_junho = '06'
dias_junho = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30']
horas_dia = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']

df_final = pd.DataFrame(columns=['ID EQP', 'DATA HORA', 'MILESEGUNDO', 'FAIXA', 'ID DE ENDEREÇO', 'VELOCIDADE DA VIA', 'VELOCIDADE AFERIDA', 'CLASSIFICAÇÃO', 'TAMANHO', 'NUMERO DE SÉRIE', 'LATITUDE', 'LONGITUDE', 'ENDEREÇO', 'SENTIDO'])
for dia in dias_maio:
    for hora in horas_dia:
        caminho_arquivo = f'dados/fluxo_moto/2022{mes_maio}{dia}/2022{mes_maio}{dia}_{hora}.json'
        with open(caminho_arquivo, 'r', encoding="utf8") as f:
            str_json = f.read()
            dicionario_dados = json.loads(str_json)
            df_lido = pd.json_normalize(dicionario_dados)
            df_lido = df_lido.loc[df_lido['CLASSIFICAÇÃO'] == 'MOTO']
            df_lido['LATITUDE'] = df_lido['LATITUDE'].str.replace(' ', '').astype(float)
            df_lido['LONGITUDE'] = df_lido['LONGITUDE'].str.replace(' ', '').astype(float)
            df_lido = df_lido.loc[df_lido['LATITUDE'] == -19.9377]
            df_lido = df_lido.loc[df_lido['LONGITUDE'] == -43.92711]
            df_final = pd.concat([df_final, df_lido])
for dia in dias_junho:
    for hora in horas_dia:
        caminho_arquivo = f'dados/fluxo_moto/2022{mes_junho}{dia}/2022{mes_junho}{dia}_{hora}.json'
        with open(caminho_arquivo, 'r', encoding="utf8") as f:
            str_json = f.read()
            dicionario_dados = json.loads(str_json)
            df_lido = pd.json_normalize(dicionario_dados)
            df_lido = df_lido.loc[df_lido['CLASSIFICAÇÃO'] == 'MOTO']
            df_lido['LATITUDE'] = df_lido['LATITUDE'].str.replace(' ', '').astype(float)
            df_lido['LONGITUDE'] = df_lido['LONGITUDE'].str.replace(' ', '').astype(float)
            df_lido = df_lido.loc[df_lido['LATITUDE'] == -19.9377]
            df_lido = df_lido.loc[df_lido['LONGITUDE'] == -43.92711]
            df_final = pd.concat([df_final, df_lido])
df_final.rename(columns={'ID EQP': 'ID_EQP', 'DATA HORA': 'DATA_HORA', 'ID DE ENDEREÇO': 'ID_DE_ENDEREÇO', 'VELOCIDADE DA VIA':'VELOCIDADE_DA_VIA', 'VELOCIDADE AFERIDA':'VELOCIDADE_AFERIDA', 'CLASSIFICAÇÃO':'CLASSIFICACAO', 'NUMERO DE SÉRIE':'NUMERO_DE_SERIE', 'ENDEREÇO':'ENDERECO'}, inplace = True)
df_final.to_csv("dados/fluxo_moto/base_fluxo_motos.csv", index=False)

df_final.head(5)

def avalia_tipo_dia(data_atual):
    tipo_dia = "DIA_UTIL"
    if data_atual.weekday() == 5:
        tipo_dia = 'SABADO'
    elif data_atual.weekday() == 6:
        tipo_dia = 'DOMINGO'
    elif format(data_atual, "%Y-%m-%d") == '2022-06-16':
        tipo_dia = 'FERIADO'
    return tipo_dia

datas = pd.date_range(start ='2022-05-01 00:00:00', end ='2022-06-30 23:00:00', periods = 1464)   
base_datas = pd.DataFrame(datas, columns=['DATA'])
base_datas['DATA_HORA_COMPARADOR'] = base_datas['DATA'].apply(lambda data_atual: format(data_atual, "%Y-%m-%dT%H:00:00"))
base_datas['DATA_COMPARADOR'] = base_datas['DATA'].apply(lambda data_atual: format(data_atual, "%Y-%m-%d"))
base_datas['TIPO_DIA'] = base_datas['DATA'].apply(lambda data_atual: avalia_tipo_dia(data_atual))

base_datas.to_csv("dados/fluxo_moto/base_datas.csv", index=False)
base_datas

datas = pd.date_range(start ='2022-05-01 00:00:00', end ='2022-06-30 23:00:00', periods = 1464)
base_clima = pd.DataFrame(datas, columns=['DATA'])
base_clima['DATA_HORA_COMPARADOR'] = base_clima['DATA'].apply(lambda data_atual: format(data_atual, "%Y-%m-%dT%H:00:00"))
base_clima

def retorna_dados_metereologicos(data_hora, latitude, longitude):
    dataDate = datetime.strptime(data_hora, '%Y-%m-%dT%H:%M:%S')
    dataStrRequest = dataDate.strftime('%Y-%m-%d')
    
    #chamada
    url = "https://archive-api.open-meteo.com/v1/era5?latitude="+str(latitude)+"&longitude="+str(longitude)+"&start_date="+str(dataStrRequest)+"&end_date="+str(dataStrRequest)+"&hourly=temperature_2m,precipitation"
    response = requests.get(url)
    jsonObj = response.json()
    
    #pegando atributos
    dataStrComparador = dataDate.strftime('%Y-%m-%dT%H:00')
    indiceDataHora = jsonObj['hourly']['time'].index(dataStrComparador)
    
    temperature_2m = jsonObj['hourly']['temperature_2m'][indiceDataHora]
    precipitation = jsonObj['hourly']['precipitation'][indiceDataHora]
    
    return dict({'temperature_2m': temperature_2m, 'precipitation': precipitation})

teste = retorna_dados_metereologicos("2022-05-01T00:00:00", -19.9377, -43.92711)
teste

base_clima['TEMPERATURA'] = np.nan
base_clima['PRECIPITACAO'] = np.nan
for i in base_clima.index:
    dados_clima = retorna_dados_metereologicos(base_clima['DATA_HORA_COMPARADOR'][i], -19.9377, -43.92711)
    base_clima.loc[i,'TEMPERATURA'] = dados_clima['temperature_2m']
    base_clima.loc[i,'PRECIPITACAO'] = dados_clima['precipitation']
base_clima.to_csv("dados/fluxo_moto/base_clima.csv", index=False)

base_clima

base_fluxo_motocicleta = pd.read_csv('dados/fluxo_moto/base_fluxo_motos.csv')

base_fluxo_motocicleta.head(3)

def data_hora_minuto(data_hora):
    dataDate = datetime.strptime(data_hora, '%Y-%m-%dT%H:%M:%S')
    dataStrRetorno = dataDate.strftime('%Y-%m-%dT%H:00:00')
    return dataStrRetorno

base_fluxo_motocicleta['HORA_PASSAGEM_MOTO'] = base_fluxo_motocicleta['DATA_HORA'].apply(lambda data_hora_atual: data_hora_minuto(data_hora_atual))

base_fluxo_motocicleta.head(3)

frequencia_fluxo_motos_data = base_fluxo_motocicleta['HORA_PASSAGEM_MOTO'].value_counts()
frequencia_fluxo_motos_data

type(frequencia_fluxo_motos_data)

frequencia_fluxo_motos_data['2022-05-01T02:00:00']

def recupera_frequencia(indice, frequencia_fluxo_motos_data):
    freq = 0
    try:
        freq = frequencia_fluxo_motos_data[indice]
    except:
        freq = 0
    return freq

datas = pd.date_range(start ='2022-05-01 00:00:00', end ='2022-06-30 23:00:00', periods = 1464)
base_frequencia_fluxo_moto = pd.DataFrame(datas, columns=['DATA'])
base_frequencia_fluxo_moto['DATA_HORA_COMPARADOR'] = base_frequencia_fluxo_moto['DATA'].apply(lambda data_atual: format(data_atual, "%Y-%m-%dT%H:00:00"))
base_frequencia_fluxo_moto['HORA'] = base_frequencia_fluxo_moto['DATA'].apply(lambda data_atual: format(data_atual, "%H"))

base_frequencia_fluxo_moto['FREQUENCIA'] = np.nan

for i in base_frequencia_fluxo_moto.index:
    try:
        base_frequencia_fluxo_moto.loc[i,'FREQUENCIA'] = frequencia_fluxo_motos_data[base_frequencia_fluxo_moto['DATA_HORA_COMPARADOR'][i]]
    except:
        base_frequencia_fluxo_moto.loc[i,'FREQUENCIA'] = 0

base_frequencia_fluxo_moto

sns.boxplot(x=base_frequencia_fluxo_moto['HORA'].apply(lambda x : int(x)))

sns.boxplot(x=base_frequencia_fluxo_moto['FREQUENCIA'])

base_frequencia_fluxo_moto.loc[base_frequencia_fluxo_moto['FREQUENCIA'] > 119]

base_frequencia_fluxo_moto.drop(base_frequencia_fluxo_moto[base_frequencia_fluxo_moto['FREQUENCIA'] > 119].index, axis=0, inplace=True)

sns.boxplot(x=base_frequencia_fluxo_moto['FREQUENCIA'])

base_frequencia_fluxo_moto['FREQUENCIA'].describe()

sns.countplot(x = base_frequencia_fluxo_moto['FREQUENCIA'])

media_freq = base_frequencia_fluxo_moto['FREQUENCIA'].mean()
media_freq

base_frequencia_fluxo_moto['ACIMA_MEDIA_FREQUENCIA'] = base_frequencia_fluxo_moto['FREQUENCIA'].apply(lambda freq_atual : 1 if freq_atual > media_freq else 0)

base_frequencia_fluxo_moto.head(5)

base_frequencia_fluxo_moto.to_csv("dados/fluxo_moto/base_frequencia_moto.csv", index=False)

base = ""
base_tipo_dia = pd.read_csv("dados/fluxo_moto/base_datas.csv")
base_tipo_dia.head(3)

base_clima = pd.read_csv("dados/fluxo_moto/base_clima.csv")
base_clima.head(3)

base_frequencia_fluxo_moto = pd.read_csv("dados/fluxo_moto/base_frequencia_moto.csv")

base = base_frequencia_fluxo_moto

base.insert(loc = 3, column = 'TEMPERATURA', value = base['DATA_HORA_COMPARADOR'].map(lambda x : base_clima.loc[base_clima.DATA_HORA_COMPARADOR == x].iloc[0]['TEMPERATURA']))

base.insert(loc = 3, column = 'PRECIPITACAO', value = base['DATA_HORA_COMPARADOR'].map(lambda x : base_clima.loc[base_clima.DATA_HORA_COMPARADOR == x].iloc[0]['PRECIPITACAO']))

base.insert(loc = 3, column = 'TIPO_DIA', value = base['DATA_HORA_COMPARADOR'].map(lambda x : base_tipo_dia.loc[base_tipo_dia.DATA_HORA_COMPARADOR == x].iloc[0]['TIPO_DIA']))

base = base.drop(['DATA_HORA_COMPARADOR', 'DATA', 'FREQUENCIA'], axis=1)

base.columns[base.isna().any()].tolist()

base

base.isnull().sum()

base.duplicated().sum()

base = base.drop_duplicates()

base

base.to_csv("dados/fluxo_moto/base_fluxo_motos_tratada.csv", index=False)

def plota_countplot(base_principal, variavel, titulo_x, titulo_y, titulo_do_grafico):
    plt.clf()
    grafico = sns.countplot(x=base_principal[variavel])
    grafico.set_ylabel('Quantidade de Ocorrências'); 
    grafico.set_title(f'{titulo_do_grafico}\n'); 
    plt.show()
    
def plota_histplot(base_principal, variavel, titulo_x, titulo_y, titulo_do_grafico):
    plt.clf()
    grafico = sns.histplot(x=base_principal[variavel])
    grafico.set_ylabel('Quantidade de Ocorrências'); 
    grafico.set_title(f'{titulo_do_grafico}\n'); 
    plt.show()

base = pd.read_csv("dados/fluxo_moto/base_fluxo_motos_tratada.csv")
base

base.isna().sum()

base.isnull().sum()

np.unique(base['ACIMA_MEDIA_FREQUENCIA'], return_counts = True) 

plota_countplot(base, 'ACIMA_MEDIA_FREQUENCIA', 'Acima da média da frequência', 'Quantidade de Ocorrências', 'Visualização dos registros que estão acima ee abaixo da média da frequência')

plota_countplot(base, 'HORA', 'Hora', 'Quantidade de Ocorrências', 'Visualização da quantidade de amostras por hora')

plota_histplot(base, 'TEMPERATURA', 'Temperatura durante a hora de registros', 'Quantidade de Ocorrências', 'Visualização dos registros de temperatura')

sns.boxplot(x=base['TEMPERATURA'])

plota_countplot(base, 'TIPO_DIA', 'Tipo do dia do registro', 'Quantidade de Ocorrências', 'Visualização da quantidade de tipos de dia da ocorrência')

plt.clf()
grafico = sns.histplot(data=base, x="TIPO_DIA", hue="ACIMA_MEDIA_FREQUENCIA", multiple="dodge", shrink=0.8)
grafico.set_title(f'Ocorrencia de fluxos acima e abaixo da média por tipo de dia de registro\n'); 
grafico.set_ylabel('Quantidade de Ocorrências'); 
plt.show()

grafico = px.scatter_matrix(base, dimensions=['TIPO_DIA', 'PRECIPITACAO','TEMPERATURA','HORA'], color = 'ACIMA_MEDIA_FREQUENCIA')
grafico.show()

grafico = px.scatter_matrix(base, dimensions=['TEMPERATURA','PRECIPITACAO'], color = 'ACIMA_MEDIA_FREQUENCIA')
grafico.show()

grafico = px.scatter_matrix(base, dimensions=['TIPO_DIA','HORA'], color = 'ACIMA_MEDIA_FREQUENCIA')
grafico.show()

def plota_heatmap_1(rotulo_y, rotulo_x, rotulo_preenchimento, foco_preenchimento, largura = 5, altura = 5, titulo =''):
    plt.clf()
    df = base.loc[:, [rotulo_y, rotulo_x, rotulo_preenchimento]]
    colunas = df[rotulo_x].unique()
    indices = df[rotulo_y].unique()
    mapa = pd.DataFrame(columns=colunas, index = indices)
    for i in mapa.columns:
        for j in mapa.index:
            mapa[i][j] = pd.to_numeric(df.loc[df[rotulo_y]==j].loc[df[rotulo_x]==i].loc[df[rotulo_preenchimento] == foco_preenchimento].shape[0])
        mapa[i] = pd.to_numeric(mapa[i],errors = 'coerce')
    plt.figure(figsize = (largura, altura))
    grafico = sns.heatmap(mapa, annot=True, linewidths=0)
    grafico.set_title(f'{titulo}\n')
    grafico.set_ylabel(rotulo_y);
    grafico.set_xlabel(rotulo_x); 
    plt.show()
plota_heatmap_1('TIPO_DIA', 'HORA', 'ACIMA_MEDIA_FREQUENCIA', 1, 14, 2, 'Mapa de Calor Indicando Registro de Frequência de Fluxo Acima da Média')

def plota_heatmap_2(rotulo_y, rotulo_x, rotulo_preenchimento, foco_preenchimento, largura = 5, altura = 5, titulo =''):
    plt.clf()
    base[rotulo_x] = base[rotulo_x].apply(lambda x : round(int(x)))
    base['PRECIPITACAO_CATEGORICO'] = base[rotulo_y].apply(lambda x : "NAO_CHOVE" if x == 0.0 else "CHOVE")
    rotulo_y = 'PRECIPITACAO_CATEGORICO'
    df = base.loc[:, [rotulo_y, rotulo_x, rotulo_preenchimento]]
    colunas = sorted(df[rotulo_x].unique())
    indices =  sorted(df[rotulo_y].unique())
    mapa = pd.DataFrame(columns=colunas, index = indices)
    for i in mapa.columns:
        for j in mapa.index:
            mapa[i][j] = pd.to_numeric(df.loc[df[rotulo_y]==j].loc[df[rotulo_x]==i].loc[df[rotulo_preenchimento] == foco_preenchimento].shape[0])
        mapa[i] = pd.to_numeric(mapa[i],errors = 'coerce')
    plt.figure(figsize = (largura, altura))
    grafico = sns.heatmap(mapa, annot=True, linewidths=0)
    grafico.set_title(f'{titulo}\n')
    grafico.set_ylabel(rotulo_y);
    grafico.set_xlabel(rotulo_x); 
    plt.show()
plota_heatmap_2('PRECIPITACAO', 'HORA', 'ACIMA_MEDIA_FREQUENCIA', 1, 15, 3, 'Mapa de Calor Indicando Registro de Frequência de Fluxo Acima da Média')

def plota_heatmap_3(rotulo_y, rotulo_x, rotulo_preenchimento, foco_preenchimento, largura = 5, altura = 5, titulo =''):
    plt.clf()
    base[rotulo_x] = base[rotulo_x].apply(lambda x : round(x))
    base[rotulo_y] = base[rotulo_y].apply(lambda y : round(y))
    df = base.loc[:, [rotulo_y, rotulo_x, rotulo_preenchimento]]
    colunas = sorted(df[rotulo_x].unique())
    indices =  sorted(df[rotulo_y].unique(), reverse=True)
    mapa = pd.DataFrame(columns=colunas, index = indices)
    for i in mapa.columns:
        for j in mapa.index:
            mapa[i][j] = pd.to_numeric(df.loc[df[rotulo_y]==j].loc[df[rotulo_x]==i].loc[df[rotulo_preenchimento] == foco_preenchimento].shape[0])
        mapa[i] = pd.to_numeric(mapa[i],errors = 'coerce')
    plt.figure(figsize = (largura, altura))
    grafico = sns.heatmap(mapa, annot=True, linewidths=0)
    grafico.set_title(f'{titulo}\n')
    grafico.set_ylabel(rotulo_y);
    grafico.set_xlabel(rotulo_x); 
    plt.show()
plota_heatmap_3('TEMPERATURA', 'HORA', 'ACIMA_MEDIA_FREQUENCIA', 1, 9, 6, 'Mapa de Calor Indicando Registro de Frequência de Fluxo de Motocicletas Acima da Média')

base_ml = pd.read_csv('dados/fluxo_moto/base_fluxo_motos_tratada.csv')
base_ml

base_ml = pd.get_dummies(base_ml)
base_ml

X = base_ml.drop('ACIMA_MEDIA_FREQUENCIA', axis=1)
y = base_ml['ACIMA_MEDIA_FREQUENCIA']

plt.clf()
grafico = sns.countplot(x=y)
grafico.set_ylabel('Quantidade de Ocorrências'); 
grafico.set_title(f'Visualização dos registros que estão acima ee abaixo da média da frequência\n'); 
plt.show()

np.unique(y, return_counts = True) 

smote = SMOTE(sampling_strategy='minority')
X, y = smote.fit_resample(X, y)

plt.clf()
grafico = sns.countplot(x=y)
grafico.set_ylabel('Quantidade de Ocorrências'); 
grafico.set_title(f'Visualização dos registros que estão acima ee abaixo da média da frequência\n'); 
plt.show()

np.unique(y, return_counts = True) 

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X

base_majority = Orange.data.Table('dados/fluxo_moto/base_fluxo_motos_tratada_regras.csv')

base_majority.domain

majority = Orange.classification.MajorityLearner()

previsoes = Orange.evaluation.testing.TestOnTestData(base_majority, base_majority, [majority])

Orange.evaluation.CA(previsoes)

parametros = {'criterion': ['gini', 'entropy'],
              'n_estimators': range(35, 45),
              'min_samples_split': range(2, 10),
              'min_samples_leaf': range(2, 10)}

grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parametros)
grid_search.fit(X, y)
melhores_parametros = grid_search.best_params_
melhor_resultado = grid_search.best_score_
print(melhores_parametros)
print(melhor_resultado)

parametros = {'n_neighbors': range(10, 15),
              'p': [1, 2],
              'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
              'metric': ['cityblock','minkowski','euclidean']}
grid_search = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=parametros)
grid_search.fit(X, y)
melhores_parametros = grid_search.best_params_
melhor_resultado = grid_search.best_score_
print(melhores_parametros)
print(melhor_resultado)

parametros = {'tol': [0.1, 0.01, 0.001, 0.0001, 0.00001],
              'random_state': [1,2,3,4,5,6],
              'C': [2.5,3.0,3.5,4.0,4.5,5.0],
              'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}

grid_search = GridSearchCV(estimator=SVC(), param_grid=parametros)
grid_search.fit(X, y)
melhores_parametros = grid_search.best_params_
melhor_resultado = grid_search.best_score_
print(melhores_parametros)
print(melhor_resultado)

resultados_ramdom_forest = []
for i in range(30):
    kfold = KFold(n_splits=10, shuffle=True, random_state=i)
    random_forest = RandomForestClassifier(n_estimators=36, criterion='gini', min_samples_leaf= 6, min_samples_split= 5, random_state = 0)
    pontos = cross_val_score(random_forest, X, y, cv = kfold, scoring='accuracy')
    resultados_ramdom_forest.append(pontos.mean())
    
df_resultados_ramdom_forest = pd.DataFrame(resultados_ramdom_forest,columns=['RESULTADOS_RANDOM_FOREST'])
df_resultados_ramdom_forest.describe()

resultados_knn = []
for i in range(30):
    kfold = KFold(n_splits=10, shuffle=True, random_state=i)
    knn = KNeighborsClassifier(algorithm='auto', n_neighbors=14, metric='cityblock', p = 1)
    pontos = cross_val_score(knn, X, y, cv = kfold, scoring='accuracy')
    resultados_knn.append(pontos.mean())

df_resultados_knn = pd.DataFrame(resultados_knn,columns=['RESULTADOS_KNN'])
df_resultados_knn.describe()

resultados_svm = []
for i in range(30):
    kfold = KFold(n_splits=10, shuffle=True, random_state=i)
    svm = SVC(kernel='rbf', random_state=1, C = 3.0, tol = 0.01)
    pontos = cross_val_score(svm, X, y, cv = kfold, scoring='accuracy')
    resultados_svm.append(pontos.mean())
    
df_resultados_svm = pd.DataFrame(resultados_svm,columns=['RESULTADOS_SVM'])
df_resultados_svm.describe()

resultados = pd.DataFrame({'RANDOM_FOREST': resultados_ramdom_forest, 'KNN': resultados_knn, 'SVM': resultados_svm})
resultados.describe()

(resultados.std() / resultados.mean())*100

#Recupera e trata colunas da base
base_ml = pd.read_csv('dados/fluxo_moto/base_fluxo_motos_tratada.csv')
base_ml = base_ml.drop(['TIPO_DIA', 'HORA'], axis=1)

#Separa a classe alvo
X = base_ml.drop('ACIMA_MEDIA_FREQUENCIA', axis=1)
y = base_ml['ACIMA_MEDIA_FREQUENCIA']

#Escalona os dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

#Faz o tunning dos parametros
# -> Random Forest
parametros = {'criterion': ['gini', 'entropy'],
              'n_estimators': range(38, 41),
              'min_samples_split': range(6, 10),
              'min_samples_leaf': range(6, 10)}
grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parametros)
grid_search.fit(X, y)
melhores_parametros = grid_search.best_params_
melhor_resultado = grid_search.best_score_
print("==========================\n Tunning Random Forest\n==========================")
print(melhores_parametros)
print(melhor_resultado)
# -> Random KNN
parametros = {'n_neighbors': [35,40, 45],
              'p': [1, 2],
              'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
              'metric': ['cityblock','minkowski','euclidean']}
grid_search = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=parametros)
grid_search.fit(X, y)
melhores_parametros = grid_search.best_params_
melhor_resultado = grid_search.best_score_
print("==========================\n     Tunning KNN\n==========================")
print(melhores_parametros)
print(melhor_resultado)
# -> Random SVM
parametros = {'tol': [0.1, 0.01, 0.001, 0.0001, 0.00001],
              'random_state': [1,2,3,4,5,6],
              'C': [0.5, 1.0, 1.5, 2.0],
              'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}
grid_search = GridSearchCV(estimator=SVC(), param_grid=parametros)
grid_search.fit(X, y)
melhores_parametros = grid_search.best_params_
melhor_resultado = grid_search.best_score_
print("==========================\n     Tunning SVM\n==========================")
print(melhores_parametros)
print(melhor_resultado)

# Avaliacao dos algoritimos com cross-validation
resultados_ramdom_forest = []
for i in range(30):
    kfold = KFold(n_splits=10, shuffle=True, random_state=i)
    random_forest = RandomForestClassifier(n_estimators=38, criterion='entropy', min_samples_leaf=6, min_samples_split=7, random_state = 0)
    pontos = cross_val_score(random_forest, X, y, cv = kfold, scoring='accuracy')
    resultados_ramdom_forest.append(pontos.mean())
    
resultados_knn = []
for i in range(30):
    kfold = KFold(n_splits=10, shuffle=True, random_state=i)
    knn = KNeighborsClassifier(algorithm='ball_tree', n_neighbors=40, metric='cityblock', p = 1)
    pontos = cross_val_score(knn, X, y, cv = kfold, scoring='accuracy')
    resultados_knn.append(pontos.mean())
    
resultados_svm = []
for i in range(30):
    kfold = KFold(n_splits=10, shuffle=True, random_state=i)
    svm = SVC(kernel='rbf', random_state=1, C = 1.5, tol = 0.1)
    pontos = cross_val_score(svm, X, y, cv = kfold, scoring='accuracy')
    resultados_svm.append(pontos.mean())
resultados_dados_climaticos = pd.DataFrame({'RANDOM_FOREST': resultados_ramdom_forest, 'KNN': resultados_knn, 'SVM': resultados_svm})
resultados_dados_climaticos.describe()

#Recupera e trata colunas da base
base_ml = pd.read_csv('dados/fluxo_moto/base_fluxo_motos_tratada.csv')
base_ml = base_ml.drop(['PRECIPITACAO', 'TEMPERATURA'], axis=1)
base_ml = pd.get_dummies(base_ml)

#Separa a classe alvo
X = base_ml.drop('ACIMA_MEDIA_FREQUENCIA', axis=1)
y = base_ml['ACIMA_MEDIA_FREQUENCIA']

#Escalona os dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

#Faz o tunning dos parametros
# -> Random Forest
parametros = {'criterion': ['gini', 'entropy'],
              'n_estimators': range(42, 45),
              'min_samples_split': range(6, 10),
              'min_samples_leaf': range(4, 8)}
grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parametros)
grid_search.fit(X, y)
melhores_parametros = grid_search.best_params_
melhor_resultado = grid_search.best_score_
print("==========================\n Tunning Random Forest\n==========================")
print(melhores_parametros)
print(melhor_resultado)
# -> Random KNN
parametros = {'n_neighbors': [4,6,8,10,12,14,15,16],
              'p': [1, 2],
              'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
              'metric': ['cityblock','minkowski','euclidean']}
grid_search = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=parametros)
grid_search.fit(X, y)
melhores_parametros = grid_search.best_params_
melhor_resultado = grid_search.best_score_
print("==========================\n     Tunning KNN\n==========================")
print(melhores_parametros)
print(melhor_resultado)
# -> Random SVM
parametros = {'tol': [0.1, 0.01, 0.001, 0.0001, 0.00001],
              'random_state': [1,2],
              'C': [0.5, 1.0, 1.5, 2.0],
              'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}
grid_search = GridSearchCV(estimator=SVC(), param_grid=parametros)
grid_search.fit(X, y)
melhores_parametros = grid_search.best_params_
melhor_resultado = grid_search.best_score_
print("==========================\n     Tunning SVM\n==========================")
print(melhores_parametros)
print(melhor_resultado)

# Avaliacao dos algoritimos com cross-validation
resultados_ramdom_forest = []
for i in range(30):
    kfold = KFold(n_splits=10, shuffle=True, random_state=i)
    random_forest = RandomForestClassifier(n_estimators=42, criterion='entropy', min_samples_leaf=5 , min_samples_split=7, random_state = 0)
    pontos = cross_val_score(random_forest, X, y, cv = kfold, scoring='accuracy')
    resultados_ramdom_forest.append(pontos.mean())
    
resultados_knn = []
for i in range(30):
    kfold = KFold(n_splits=10, shuffle=True, random_state=i)
    knn = KNeighborsClassifier(algorithm='brute', n_neighbors=15, metric='minkowski', p = 2)
    pontos = cross_val_score(knn, X, y, cv = kfold, scoring='accuracy')
    resultados_knn.append(pontos.mean())
    
resultados_svm = []
for i in range(30):
    kfold = KFold(n_splits=10, shuffle=True, random_state=i)
    svm = SVC(kernel='rbf', random_state=1, C = 1.0, tol = 0.1)
    pontos = cross_val_score(svm, X, y, cv = kfold, scoring='accuracy')
    resultados_svm.append(pontos.mean())
resultados_dados_dia_hora = pd.DataFrame({'RANDOM_FOREST': resultados_ramdom_forest, 'KNN': resultados_knn, 'SVM': resultados_svm})
resultados_dados_dia_hora.describe()

print("==============================================================================")
print("Dados Fluxo + Dados Clima + Dados de Tipos de Dias e Horários")
print("==============================================================================")
print(resultados.describe())
print("..............................................................................")
print(" ")
print("==============================================================================")
print("Dados Fluxo + Dados Clima")
print("==============================================================================")
print(resultados_dados_climaticos.describe())
print("..............................................................................")
print(" ")
print("==============================================================================")
print("Dados Fluxo + Dados de Tipos de Dias e Horários")
print("==============================================================================")
print(resultados_dados_dia_hora.describe())

dados_medias = {'ALGORITMO': ['Random Forest','Random Forest','Random Forest', 'KNN', 'KNN', 'KNN', 'SVM', 'SVM', 'SVM'],
                'CONFIGURACAO': ['Configuração 1','Configuração 3','Configuração 2','Configuração 1','Configuração 3','Configuração 2','Configuração 1','Configuração 3','Configuração 2'],
                'MEDIA': [resultados.describe()['RANDOM_FOREST']['mean'],resultados_dados_dia_hora.describe()['RANDOM_FOREST']['mean'],resultados_dados_climaticos.describe()['RANDOM_FOREST']['mean'],
                         resultados.describe()['KNN']['mean'],resultados_dados_dia_hora.describe()['KNN']['mean'],resultados_dados_climaticos.describe()['KNN']['mean'],
                         resultados.describe()['SVM']['mean'],resultados_dados_dia_hora.describe()['SVM']['mean'],resultados_dados_climaticos.describe()['SVM']['mean']]}
df_medias = pd.DataFrame(dados_medias)

plt.clf()
plt.figure(figsize = (8, 6))
grafico = sns.barplot(data=df_medias, x="ALGORITMO", y="MEDIA", hue="CONFIGURACAO")
grafico.set_title(f'Média da Acurácia por Algoritmo e Configuração de Dados do Modelo\n'); 
grafico.set_ylabel('Média da Acurácia');
grafico.set_xlabel('Algoritmos');
plt.legend(loc='upper right')
plt.ylim(0.6, 0.9)
plt.show()

dados_desvio_padrao = {'ALGORITMO': ['Random Forest','Random Forest','Random Forest', 'KNN', 'KNN', 'KNN', 'SVM', 'SVM', 'SVM'],
                'CONFIGURACAO': ['Configuração 1','Configuração 3','Configuração 2','Configuração 1','Configuração 3','Configuração 2','Configuração 1','Configuração 3','Configuração 2'],
                'DESVIO_PADRAO': [resultados.describe()['RANDOM_FOREST']['std'],resultados_dados_dia_hora.describe()['RANDOM_FOREST']['std'],resultados_dados_climaticos.describe()['RANDOM_FOREST']['std'],
                         resultados.describe()['KNN']['std'],resultados_dados_dia_hora.describe()['KNN']['std'],resultados_dados_climaticos.describe()['KNN']['std'],
                         resultados.describe()['SVM']['std'],resultados_dados_dia_hora.describe()['SVM']['std'],resultados_dados_climaticos.describe()['SVM']['std']]}
df_desvio_padrao = pd.DataFrame(dados_desvio_padrao)

plt.clf()
plt.figure(figsize = (8, 6))
grafico = sns.barplot(data=df_desvio_padrao, x="ALGORITMO", y="DESVIO_PADRAO", hue="CONFIGURACAO")
grafico.set_title(f'Desvio Padrão da Acurácia por Algoritmo e Configuração de Dados do Modelo\n'); 
grafico.set_ylabel('Desvio Padrão da Acurácia');
grafico.set_xlabel('Algoritmos');
plt.legend(loc='upper right')
plt.ylim(0.0005, 0.005)
plt.show()

base_ml = pd.read_csv('dados/fluxo_moto/base_fluxo_motos_tratada.csv')
base_ml = pd.get_dummies(base_ml)
X = base_ml.drop('ACIMA_MEDIA_FREQUENCIA', axis=1)
y = base_ml['ACIMA_MEDIA_FREQUENCIA']
X, y = smote.fit_resample(X, y)
X = scaler.fit_transform(X)
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size = 0.15, random_state = 0)
X_treinamento.shape, y_treinamento.shape, X_teste.shape, y_teste.shape, X.shape, y.shape

random_forest = RandomForestClassifier(n_estimators=37, criterion='entropy', min_samples_leaf=7, min_samples_split=9, random_state=0)
random_forest.fit(X_treinamento, y_treinamento)

previsoes = random_forest.predict(X_teste)

accuracy_score(y_teste, previsoes)

cm = ConfusionMatrix(random_forest)
cm.fit(X_treinamento, y_treinamento)
cm.score(X_teste, y_teste)

print(classification_report(y_teste, previsoes))

knn = KNeighborsClassifier(algorithm='auto', n_neighbors=20, metric='cityblock', p = 1)
knn.fit(X_treinamento, y_treinamento)

previsoes = knn.predict(X_teste)

accuracy_score(y_teste, previsoes)

cm = ConfusionMatrix(knn)
cm.fit(X_treinamento, y_treinamento)
cm.score(X_teste, y_teste)

print(classification_report(y_teste, previsoes))

svm = SVC(kernel='rbf', random_state=1, C =2.5, tol=0.01)
svm.fit(X_treinamento, y_treinamento)

previsoes = svm.predict(X_teste)

accuracy_score(y_teste, previsoes)

cm = ConfusionMatrix(svm)
cm.fit(X_treinamento, y_treinamento)
cm.score(X_teste, y_teste)

print(classification_report(y_teste, previsoes))
