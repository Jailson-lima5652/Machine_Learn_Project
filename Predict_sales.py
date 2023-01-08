
"""# Predict Sale - Previsão com Séries temporais

## Importando dados
"""

import pandas; from zipfile import ZipFile
import pandas as pd
import numpy as np
import scipy 

import matplotlib.pyplot as plt

zip = ZipFile("/content/store-sales-time-series-forecasting.zip");zip.namelist()
dir = zip.namelist()
data_sets = {dir[:-4]: pd.read_csv(zip.open(dir),
                         encoding = ('ISO-8859-1'),
                         low_memory = False) for dir in dir }

holidays_events = data_sets[dir[0][:-4]]
oil = data_sets[dir[1][:-4]]
sample_submission = data_sets[dir[2][:-4]]
stores = data_sets[dir[3][:-4]]
test = data_sets[dir[4][:-4]]
train = data_sets[dir[5][:-4]]
transactions = data_sets[dir[6][:-4]]

"""## Análise 1 - Tratamento

Holidays Events
"""

'''
============= DICIONÁRIO DE VARIÁVEIS ==============
Tab -> Tabela de eventos e feriados 
TYPE: Tipo de evento ou feriado (Se é transferido,adicionado, evento etc )
LOCALE: Nível do evento ou feriado (nacional, internanional etc)
LOCALE_NAME: Região/local do evento ou feriado
DESCRIPTION: Descrição de evento ou feriado 
TRANSFERRED: Situação de transferência de feriado ou evento 
'''
analise(holidays_events)
unique_values(holidays_events)



# EXTRAINDO DIAS DE FERIADOS
holidays = holidays_events[(holidays_events.transferred == False) & (holidays_events.type != 'Work Day')]
analise(holidays); '''VEJA QUE HÁ DUPLICATA EM DATAS DE FERIADOS'''
holidays = holidays.drop_duplicates(subset= ['date'])


# SEPARANDO FERIADOS NACIONAL/REGIONAL E LOCAL PARA VERIFICAR INPACTOS DE VENDAS
holidays_national_regional = holidays[holidays.locale.isin(['Regional','National'])]
holidays_local = holidays[holidays.locale == 'Local']


# FERIADOS TRANSFERIDOS
holidays_trans = holidays_events[holidays_events.transferred]
translado = holidays_events[holidays_events.type=='Transfer']

"""Stores - Lojas """

analise(stores)
unique_values(stores)

"""Train"""

analise(train)
unique_values(train)

"""Test"""

analise(test)

"""## Análise 2 - Gráficos

"""

train_eda = train.copy()
train_eda.date = pd.to_datetime(train_eda.date)
train_eda.set_index('date', inplace = True)

train_eda.resample('d').sum()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 



# ====================== resultados de vendas por período =======================

#sns.distplot(train['sales'])
#plt.show()

daily_sales=train_eda.resample('d').sales.mean().to_frame()  ## Resample sales by day
weekly_sales=train_eda.resample('w').sales.mean().to_frame()  ## Resample sales by week
monthly_sales=train_eda.resample('m').sales.mean().to_frame()  ## Resample sales by month

sns.relplot(x = daily_sales.index ,y= daily_sales.sales,
            kind= 'line',aspect = 4,
            hue=daily_sales.index.year)
plt.title('Média de vendas diárias')
plt.show();print()


sns.relplot(x = weekly_sales.index ,y= weekly_sales.sales,
            kind= 'line',aspect = 4,
            hue=weekly_sales.index.year)
plt.title('Média de vendas semanais')
plt.show();print()


sns.relplot(x = monthly_sales.index ,y= monthly_sales.sales,
            kind= 'line',aspect = 4,
            hue=monthly_sales.index.year)
plt.title('Média de vendas mensais')
plt.show();print()



# ================= MÉDIAS DE VENDAS POR DIA DA SEMANA =========================

## Set Plot_Parameters
sns.set(color_codes=True)        
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 5))
plt.rc("axes", labelweight="bold", labelsize="LARGE",
       titleweight="bold", titlesize=14, titlepad=10)

plot_params = dict(color="0.75", 
                   style=".-",
                   markeredgecolor="0.25",
                   markerfacecolor="0.25",
                   legend=False)
# ______________________________________________________________________________

temp = train_eda.groupby(train_eda.index.day_of_week)['sales'].mean().to_frame()
 
temp.plot(**plot_params)
plt.title('Média de vendas por dia da semana')
plt.show()


'''
- vemos que há uma tendência crescente de vendas com o passar dos anos 
 
- parece haver uma sazonalidade (repetições de tendências num período de tempo)
  visto que a venda sempre é maior aos finais de semana e finais de ano
'''

# ______________________________________________________________________________
# PODE HAVER UMA CORRELAÇÃO ENTRE VENDEAS(SALES) E PROMOÇÃO(ONPROMOTION) 
# VAMOS VERIFICAR ISSO 

avg_sales=train_eda.groupby(['date'])['sales','onpromotion'].mean().reset_index() 
array_corr = avg_sales[['sales','onpromotion']].corr()


plt.figure(figsize=(20,10))
sns.regplot(data=avg_sales,
            x='onpromotion',y='sales',
            ci=None,
            scatter_kws={'color':'0.4'},
            line_kws={'color':'red','linewidth':3})
plt.title("Gráfico de Vendas em função das Promoções")
plt.show()

print('Correlação: {}'.format(array_corr.iloc[0,1]))


'''
Há também uma boa correlação entre vendas e promoções, visto que quando há promoções 
há também maiores resultados de vendas
'''

# VERIFICANDO RESULTADOS DE VENDAS POR FAMÍLIA 
temp = train_eda.groupby('family').mean()[['sales']].sort_values(by = 'sales',ascending=False)


plt.figure(figsize=(20,10))
sns.barplot(data=temp,
            x = temp.sales, y = temp.index,
            ci=None,
            order=list(temp.index))
plt.show()

# RESULTADOS DE VENDAS POR LOJAS 
temp = train_eda.groupby('store_nbr').mean()[['sales']].sort_values(by = 'sales',ascending = False)

plt.figure(figsize =(20,10))
sns.barplot(data = temp,
            x = temp.index, y = temp.sales,
            ci = None,
            order = list(temp.index))
plt.show()

"""## Análise 3 - Tendências

### Tendências de vendas por período (dia, semana, mes, ano, dias da semana)

Para aplicação do modelo, verificaremos a tendência de vendas usando média móvel com uma janela de 365 dias para suavizar quaisquer mudanças de curto prazo num determinado ano
"""

avg_sales = train_eda.groupby('date').sales.mean()
moving_avg = avg_sales.rolling(  window=365,
                                 min_periods=183,
                                 center=True).mean()

plt.figure( figsize = (20,10) )
ax = avg_sales.plot(**plot_params)
ax = moving_avg.plot(color='red',linewidth=3)
plt.title("Tendência de Vendas por Média Móvel")


'''
Esse gráfico corrobora a ideia de tendência crescente de vendas com o passar dos anos
'''

"""Determinando Sazonalidade (Tendência de venda num determinado paríodo)"""

avg_sales = train_eda.groupby('date')['sales'].mean().to_frame()

# days within a week
avg_sales["day"] = avg_sales.index.dayofweek  # the x-axis (freq)
avg_sales["week"] = avg_sales.index.week  # the seasonal period (period)

# days within a year
avg_sales["dayofyear"] = avg_sales.index.dayofyear
avg_sales["dayofweek"] = avg_sales.index.dayofweek
avg_sales["year"] = avg_sales.index.year

sns.relplot( data = avg_sales,
             x = 'dayofweek',
             y = 'sales',
             kind = 'line',
             aspect = 3
             )
plt.show()

sns.relplot( data = avg_sales,
             x = 'dayofyear',
             y = 'sales',
             kind = 'line',
             aspect = 3,
             hue = 'year'
             )
plt.show()


'''
Vemos que há de fato uma sazonalidade

'''

var_year = train_eda.groupby(train_eda.index.year).var().reset_index()
var_month = train_eda.groupby(train_eda.index.month).var().reset_index()
var_week = train_eda.groupby(train_eda.index.week).var().reset_index()
var_dayofweek = train_eda.groupby(train_eda.index.dayofweek).var().reset_index()


sns.relplot(x = var_year.date,
            y = var_year.sales / var_year.sales.max(),
            kind = 'line',
            aspect = 4)
plt.title("Variabilidade Anual de Vendas")
plt.xlabel("Year");plt.show(); print()

sns.relplot(x = var_month.date,
            y = var_month.sales / var_month.sales.max(),
            kind = 'line',
            aspect = 4 )
plt.title("Variabilidade Mensal de Vendas")
plt.xlabel("Month");plt.show(); print()

sns.relplot(x = var_week.date,
            y = var_week.sales / var_week.sales.max(),
            kind = 'line',
            aspect = 4 )
plt.title("Variabilidade Semanal de Vendas")
plt.xlabel("Week");plt.show(); print()


sns.relplot(x = var_dayofweek.date,
            y = var_dayofweek.sales / var_dayofweek.sales.max(),
            kind = 'line',
            aspect = 4 )
plt.title("Variabilidade Semanal de Vendas")
plt.xlabel("Day of Week");plt.show(); print()

"""### Tendências de vendas em Feriados

Acredita-se que há mais vendas em período dias de feriado. Desse modo, parece haver uma relação entre a base de treino (vendas) e a base de feriados. Verifiquemos
"""

avg_SalesForDay = train_eda.resample('d').mean()[['sales']]
avg_SalesInHolidays =  train[train.date.isin(holidays_national_regional.date.tolist())].groupby('date').mean()[['sales']]
avg_SalesInHolidays2 = train[train.date.isin(holidays_local.date.tolist())].groupby('date').mean()[['sales']]

avg_SalesForDay.plot(**plot_params)
plt.scatter(x = avg_SalesInHolidays.index,
                 y = avg_SalesInHolidays.sales,
                 color = 'red')
plt.title('Average sales and holiday sales regional/national  for days')
plt.show(); print()

avg_SalesForDay.plot(**plot_params)
plt.scatter(x = avg_SalesInHolidays2.index,
            y = avg_SalesInHolidays2.sales,
            color = 'red')
plt.title('Average sales and holiday sales local for days')
plt.show()

'''
Parece que há mais vendas en dias de feriados regionais/nacionais
Consideremos então esses dias para nosos modelo de previsão (e os outros 
descartamos do modelo - basta verificar que obteremos um melhor resultado)
'''

holidaysNR = holidays_national_regional.copy()

# UNINDO FERIADOS A BASE DE VENDAS 
df = pd.merge(train,holidaysNR[['date','type']],
              on = 'date',
              how = 'left')

# SUBSTITUIREMOS OS TIPOS DE FERIADOS PELO NUMERO 1, APENAS PARA IDENTIFICAR ESSE DIA COMO FERIADO 

dicReplace = {'Additional':1,
              'Bridge':1,
              'Event':1,
              'Holiday':1,
              'Transfer':1}

df.type = df.type.map(dicReplace) 
df.type = df.type.fillna(0)

df.date = pd.to_datetime(df.date)
df.set_index('date',inplace = True)

# ANALISAREMOS AQUI AS VENDAS NOS FERIADOS E FINAIS DE SEMANA
# OBS: LEMBRANDO QUE O PRIMEIRO DIA DO TODOS OS FUNCIONÁRIOS TRABALHAM!

# TEMOS ENTÃO UMA COLUNA IDENTIFICADORA DE TRABALHO QUE RECEBERÁ 1 PARA INATIVIDADE E 0 PARA ATIVIDADES NORMAIS
df['dayofweek'] = df.index.dayofweek
df['dayofyear'] = df.index.dayofyear
df.loc[ (df.dayofweek == 5 ) | (df.dayofweek == 6), 'type'] = 1
df.loc[ df.dayofyear == 1, 'type'] = 0

df.rename(columns={'type': 'work'},inplace = True)



# AVALIANDO VENDAS
avg_SalesHolidays = df[df.work == 1].groupby('date').mean()[['sales']]
avg_workdaysales = df[df.work != 1].groupby('date').mean()[['sales']]



plt.figure(figsize= (20,6) )
plt.scatter(x = avg_workdaysales.index,
         y = avg_workdaysales.sales,
         color = 'grey'
         )

plt.scatter(x = avg_SalesHolidays.index,
            y = avg_SalesHolidays.sales,
            color = 'red')
legenda = ['feriados e fins de semana','dia de semana']
plt.title('Média de vendas')
plt.legend(legenda,loc = 2)
plt.show()


'''
VEMOS ENTÃO DADOS QUE CORROBORAM MAIOR VENDAS EM FINAIS DE SEMANA E FERIADOS 
REGIONAIS OU NACIONAIS

O indicador de férias será utilizado como variável para treinamento.
'''

"""### Análise de preço do petróleo

O preço do petróleo é um influenciador significativo nas vendas dos produtos uma vez que afeta diretamente o poder de compra dos clientes havendo assim uma correlação entre ambos
"""

oil.date = pd.to_datetime(oil.date)
oil.set_index('date',inplace = True)

df = pd.merge( left = train_eda,
          right = oil,
         left_index = True,
         right_index = True, 
         how='left')
df.rename(columns = {'dcoilwtico': 'oilprice'},inplace = True)

from scipy import stats
ValueAnalysis = stats.scoreatpercentile(oil.dcoilwtico, 65)

avg_salesforoilprice = df.groupby('oilprice').mean()[['sales']]
cor = avg_salesforoilprice.reset_index().corr().iloc[0,1]

plt.figure(figsize = (20,6))
sns.regplot(x = avg_salesforoilprice.index,
            y = avg_salesforoilprice.sales,
            ci=None,
            scatter_kws={'color':'0.5'},
            line_kws={'color':'red','linewidth':3})
plt.show()
print('Correlação = {}'.format(cor))

'''
Com isso, vemos que, de fato, quanto maior o preço do petróleo, menor são as vendas
Portanto, o preço do petróleo será usado como variável de treinamento.
'''

"""### Análise de Lójas"""

stores.sample(10)

df = pd.merge(train_eda, stores,
              on = 'store_nbr',
              how = 'left')

#____________________________________________________________________________________________________
avg_salesforcity = df.groupby('city').mean()[['sales']].sort_values(by = 'sales',ascending = False)

plt.figure(figsize = (20,10))
sns.barplot(x = avg_salesforcity.sales,
            y = avg_salesforcity.index)
plt.show()
#____________________________________________________________________________________________________

avg_salesfortype = df.groupby('type').mean()[['sales']].sort_values(by = 'sales',ascending = False)

plt.figure(figsize = (20,10))
plt.pie(avg_salesfortype, labels=avg_salesfortype.index,
        autopct="%1.1f%%")
plt.axis('equal')
plt.title('Avg. Sales by Store Type')
plt.show()

#____________________________________________________________________________________________________

avg_salesforcluster = df.groupby('cluster').mean()[['sales']].sort_values(by = 'sales',ascending = False)

plt.figure(figsize = (20,10))
plt.pie(avg_salesforcluster, labels=avg_salesforcluster.index,
        autopct="%1.1f%%")
plt.axis('equal')
plt.title('Avg. Sales by Store cluster')
plt.show()

#____________________________________________________________________________________________________

avg_salesforstate = df.groupby('state').mean()[['sales']].sort_values(by = 'sales',ascending = False)

plt.figure(figsize = (20,10))
sns.barplot(x = avg_salesforstate.sales, 
            y = avg_salesforstate.index)

plt.title('Avg. Sales by Store state')
plt.show()

#____________________________________________________________________________________________________

avg_salesforstore = df.groupby('store_nbr').mean()[['sales']].sort_values(by = 'sales',ascending = False)

plt.figure(figsize = (20,10))
sns.barplot(x = avg_salesforstore.index, 
            y = avg_salesforstore.sales,
            order = avg_salesforstore.index 
            )

plt.title('Avg. Sales by Store state')
plt.show()

"""## Aplicando Modelo de Previsão"""

from prophet import Prophet
# NORMALIZANDO PARA APLICAÇÃO DO MODELO
df = train[['date','sales']]
df = df.groupby('date').mean()[['sales']].reset_index()
df = df.rename(columns = {'date': 'ds',
                          'sales': 'y'})

df_holidays= holidays[['date','type']]
df_holidays.rename(columns = {'date': 'ds',
                              'type': 'holiday'},inplace= True)
df_holidays
df_holidays_prior = holidays[holidays.locale.isin(['Regional', 'National'])][['date','type']]
df_holidays_prior.rename(columns = {'date': 'ds',
                              'type': 'holiday'},inplace= True)

# GERANDO MODELO 

model = Prophet(seasonality_mode ='multiplicative',
                holidays= df_holidays_prior)
model.fit(df)

# GERANDO PREDIÇÕES
newdates = model.make_future_dataframe(periods = 12, freq = 'm') # GERA PREDIÇÃO DE 12 MESES 
predicted = model.predict(newdates)[['ds',
                                     'trend', # TENDENCIA
                                     'trend_lower', # MENOR VALOR DE TENDÊNCIA POSSÍVEL
                                     'trend_upper', # MAIOR VALOR DE TENDÊNCIA POSSÍVEL
                                     'yhat',# VALOR DE VENDA PREDITADO
                                     'yhat_lower',# MENOR VENDA POSSÍVEL
                                     'yhat_upper'# MAIOR VENDA POSSÍVEL
                                     ]] 

predicted2 = model.predict(df) 
fig1 = model.plot(predicted)
fig2 = model.plot_components(predicted)

# MEDINDO QUALIDADE 
from sklearn.metrics import mean_squared_log_error

avaliar_predicted = predicted.loc[:'2017-08-15']

erro_medio = np.sqrt(mean_squared_error(df.y,predicted2.yhat))
erro_medio
