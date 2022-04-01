##executar no terminal: streamlit run App.py

import pandas as pd
import numpy as numpy
import matplotlib.pyplot as plt 
import streamlit as st
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#=================================================================================================


st.title("Web App Para Previsões com Machine Learning em Python:")
st.write("**Mad By:[Ednei Cunha Vicente](https://www.linkedin.com/in/ednei-cunha-vicente-551b64187/")
st.write("**Mad By:[Ednei Cunha Vicente](https://github.com/ednei-code")
#subtitulo
st.subheader("Este conjunto de Dados contém registros de Pacientes que desenvolveram ou não Doenças Cardiovasculares")

#texto
st.write("Tabela com Dados Originais")


#carrega os dados
df = pd.read_csv('heart.csv')
names = ['age','sex','slope','cp','fbs','target']
to_drop = ['trestbps','chol','restecg','thalach','exang','oldpeak','ca','thal']
df.drop(to_drop, inplace=True, axis=1)
df.columns = names
#imprime o dataframe
st.dataframe(df)

#resumo estatistico
st.write("Tabela de Resumo Estatistico:")
st.write(df.describe())

#texto
graf = st.selectbox("Escolha a variavel para a construção do grafico:",
	options=['age','sex','slope','cp','fbs'])

if graf == 'age':
	st.bar_chart(df['age'].value_counts())
elif graf == 'sex':
	st.bar_chart(df['sex'].value_counts())
elif graf == 'slope':
	st.bar_chart(df['slope'].value_counts())
elif graf == 'cp':
	st.bar_chart(df['cp'].value_counts())
elif graf == 'fbs':
	st.bar_chart(df['fbs'].value_counts())


#divisao de dados de entrada e saida
x = df.iloc[: , 0:5].values
y = df.iloc[: , -1].values

#dividir em treino e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.25)

#funcao para obter novos dados de usuarios
def novos_dados():
	age = st.sidebar.slider(' Idade do paciente',29,77)
	sex = st.sidebar.slider('Sexo do paciente',0,1)
	slope = st.sidebar.slider('Obstrução dos vasos Teste de normalidade',0,2)
	cp = st.sidebar.slider('Tipo dor no peito',0, 3)
	fbs = st.sidebar.slider('Açucar no sangue em jejum',0,2)
  
	#cria um dicionario de dados
	user_data = {'age' : age,
				 'sex' : sex,
				 'slope': slope,
				 'cp': cp,
				 'fbs': fbs,
			
				 }

	features = pd.DataFrame(user_data, index = [0])

	return features

#Armazena input dos usuarios
user_input = novos_dados()

#sub-titulo
st.subheader('Input do usuario(novos dados) :')
st.write(user_input)

#cria o modelo
modelo = RandomForestClassifier()

#treina o modelo
modelo.fit(x_treino, y_treino)

#imprime a accuracia do modelo
st.subheader("Accuracia do modelo: " + str(accuracy_score(y_teste, modelo.predict(x_teste))*100)+ '%')

#faz as previsoes
prediction = modelo.predict(user_input.values)

#imprime o resultado
if prediction[0] == 1:
	st.subheader("Esse paciente, provavelmente, desenvolverá doença Cardiovasculares! ")
else:
	st.subheader("Esse paciente, provavelmente, não desenvolverá doença Cardiovasculares!")
