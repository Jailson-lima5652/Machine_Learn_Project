from sklearn.linear_model import LogisticRegression

# TRANSFORMANDO V.A CATEGÓRICA 

replace = {'MAU PAGADOR': 1, 'BOM PAGADOR': 0}
df['TARGET'] = df['TARGET'].map(replace)

# TRATANDO DF PARA CONSTRUÇÃO DO MODELO
cols = [x for x in df.columns if df.dtypes[x]==float or df.dtypes[x]==int]
df2 = df[cols]
df2.drop([df2.columns[0]],axis = 1, inplace = True)

# EXTRAINDO AMOSTRAS VARIÁVEIS PARA MODELO
df_treino = df2.sample(250000)
df_teste = df2.sample(125000)

Xs_trein = np.array(df_treino[[x for x in df_treino.columns[1:]]],ndmin=2)
y_trein = df_treino.iloc[:,0]

Xs_test = df_teste[[x for x in df_teste.columns[1:]]] 
y_test = df_teste.iloc[:,0]

# APLICANDO MODELO
logit_regression = linear_model.LogisticRegression()
logit_regression.fit(Xs_trein,y_trein)

y_predict = np.round(logit_regression.predict(Xs_test),0)
y_predict

# MATRIZ DE CONFUZÃO
cm = metrics.confusion_matrix(y_test, y_predict, labels=[True, False])/len(y_test)
Matrix_confusion = pandas.DataFrame(cm,index = ['False','True'],columns = ['False','True'])

accuracy = metrics.accuracy_score(y_test,y_predict)

# ESTIMADORES DE QUALIDADE
print('  Matrix Confusion\n',Matrix_confusion)
print('\n Accuracy: ',accuracy)

# Estimativas de probabilidade
pandas.DataFrame(prob.reshape(-1,1))[0]


