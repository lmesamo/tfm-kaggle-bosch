#!/usr/bin/env python
# coding: utf-8

# <div style="width: 100%; clear: both;">
# <div style="float: left; width: 50%;">
# <img src="http://www.uoc.edu/portal/_resources/common/imatges/marca_UOC/UOC_Masterbrand.jpg" align="left">
# </div>
# <div style="float: right; width: 50%;">
# <p style="margin: 0; padding-top: 22px; text-align:right;"><strong>M2.879 · TFM - Área 2- Machine Learning</strong></p>
# <p style="margin: 0; padding-top: 22px; text-align:right;"><strong>Predicción de errores en producción industrial de piezas</strong></p>
#     <p style="margin: 0; text-align:right;"><strong>Lorenzo Mesa Morales</strong></p>
#     <p style="margin: 0; text-align:right;">2019-2 · Máster universitario en Ciencia de datos (Data science)</p>
#     <p style="margin: 0; text-align:right;">Nombre Consultor/a: Jerónimo Hernández González</p>
#     <p style="margin: 0; text-align:right;">Nombre Profesor/a responsable de la asignatura: Jordi Casas Roma</p>
# </div>
# </div>
# <div style="width:100%;">&nbsp;</div>
# 
# 
# # Implementación
# 
# El documento se estructura siguiendo la metodología CRISP-DM en las siguientes secciones:
# 
#  <ol start="1">
#   <li>Carga del conjuntos de datos</li>
#   <li>Análisis de los datos
#   <li>Preparación de los datos</li>
#     3.1 Valores nulos
#     <br>3.2 Reduccion de dimensionalidad
#     <br>3.3 Técnicas de muestreo
#   <li>Modelado</li>
#     4.1 Random Forest
#     <br>4.2 eXtreme Gradient Boosting (XGBoost)
#   <li>Evaluación</li>
#     5.1 Combinación secuencial de modelos
#  </ol>
#    
# Para ello vamos a necesitar las siguientes librerías:

# In[1]:


import time
import pickle
import datetime
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
pd.set_option('display.max_columns', None)
import gc
from collections import Counter
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
from xgboost import plot_importance
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.ensemble import StackingClassifier


# # 1. Carga del conjunto de datos

# <div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
# Dada la cantidad de recursos que consume la carga de los datos se procede a mostrar un resumen de los datos que contienen cada uno de los ficheros 
# <a href="https://www.kaggle.com/c/bosch-production-line-performance/discussion/22908" title="Kaggle Discusion 22908">[Kaggle Discusion 22908]</a>
# </div>

# In[2]:


gc.collect()

def muestra_resumen(filename): 
    rowCount=0
    numberEmptyValues=0

    with open(filename, "rt") as csvfile:
        filereader = csv.reader(csvfile)
        for curRow in filereader:
            if rowCount == 0 :
                headerRow=curRow
                numberColumns=len(headerRow)
                emptyList=['']*numberColumns  # creamos una lista para las entradas vacías
                emptyCounter=Counter(emptyList)

            else:
                curCounter=Counter(curRow)
                diff = curCounter-emptyCounter  # creamos una lista de valores no vacíos

                numberNotEmpty=len(diff) # calculamos el tamaño de la lista
                numEmpty=numberColumns-numberNotEmpty 
                numberEmptyValues=numberEmptyValues+numEmpty


            rowCount=rowCount+1


    totalnumber=rowCount*numberColumns
    pctEmpty=100*numberEmptyValues/totalnumber

    print("fichero analizado       :",filename)
    print("número de filas         :",rowCount)
    print("número de columnas      :",numberColumns)
    print("número de valores vacíos:",numberEmptyValues)
    print("% de valores vacíos     :",pctEmpty)
   


# In[3]:


filename="train_numeric.csv"
muestra_resumen(filename)


# In[4]:


filename="train_categorical.csv"
muestra_resumen(filename)


# In[5]:


filename="train_date.csv"
muestra_resumen(filename)


# In[6]:


# Para simplificar el trabajo y por la limitación de recursos se utilizan únicamente con los datos de tipo numérico

# Se carga el conjunto de datos de entrenamiento de tipo numérico

# Se toma una muestra de 100 filas para determinar los dtypes.
df_sample = pd.read_csv('train_numeric.csv', nrows=100)

# Se convierten a float32 para reducir el tamaño del dataset y optimizar los recursos
float_cols = [c for c in df_sample if df_sample[c].dtype == "float64"]
float32_cols = {c: np.float32 for c in float_cols}

df_num = pd.read_csv('train_numeric.csv', engine='c', dtype={c: np.float32 for c in float_cols})


# In[7]:


# Se muestra información del conjunto de datos cargados
df_num.head()


# In[8]:


df_num.info()


# In[9]:


df_num.shape


# In[10]:


df_num.dtypes


# In[11]:


summary = df_num.describe()
summary = summary.transpose()
summary


# In[12]:


# Dada la cantidad de recursos que consume la carga del conjunto de datos de entrenamiento de tipo categórico
# solo se leen los 10000 primeros registros para realizar un análisis rápido

df_categ = pd.read_csv('train_categorical.csv', nrows=10000 ,low_memory=False)


# In[13]:


df_categ.head()


# In[14]:


df_categ.info()


# In[15]:


summary = df_categ.describe()
summary = summary.transpose()
summary


# In[16]:


# Dada la cantidad de recursos que consume la carga de del conjunto de datos de entrenamiento de tipo categórico
# solo se leen los 10000 primeros registros para realizar un análisis rápido

df_date = pd.read_csv('train_date.csv', nrows=10000 ,low_memory=False)


# In[17]:


df_date.head()


# In[18]:


df_date.info()


# In[19]:


summary = df_date.describe()
summary = summary.transpose()
summary


# # 2. Análisis de los datos

# In[20]:


# Se muestra un análisis estadístico para los atributos numéricos
df_num.describe()


# In[21]:


# Se calcula el porcentaje de resultados de piezas correctas e incorrectas en los datos:
df_num["Response"].value_counts(normalize=True)


# In[22]:


# Se muestra en formato gráfico la distribución de los resultados para evidenciar el desbalanceo en los datos

ax = df_num["Response"].value_counts().to_frame().plot(kind='bar')

totals = []

for i in ax.patches:
    totals.append(i.get_height())

total = sum(totals)

for i in ax.patches:
    ax.text(i.get_x()+.05, i.get_height()+.5,             str(round((i.get_height()/total)*100, 2))+'%', fontsize=10)


# # 3. Preparación de los datos

# ## 3.1 Valores nulos

# In[23]:


# Se sustituyen los valores nulos por la media en los datos
df_num.fillna(df_num.mean(), inplace=True)


# ## 3.2 Reducción de dimensionalidad

# In[24]:


# Se separan los datos entre las variables independientes y la variable dependiente
X = df_num.drop('Response',1)
y = df_num['Response']

# Se separan el conjunto en datos de entrenamiento (80%) y datos de test (20%) 
# Debido a la baja representación de casos de fallos, se utiliza el parámetro stratify para asegurar que 
# ambas clases están representadas en el conjunto de datos de test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2020)


# In[25]:


# A continuación, se normalizan ambos conjuntos de datos para que estén representados en la misma escala 
# y por lo tanto no tomen más importancia unos que otros.
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[26]:


X_train.shape


# In[27]:


X_test.shape


# In[28]:


y_train.shape


# In[29]:


y_test.shape


# ### 3.2.1. PCA

# In[30]:


# Se aplica la reducción de dimensionalidad tanto a los datos de entrenamiento como a los de test
# Se utiliza como parámetro un 98% de explicación de la varianza en lugar de determinar los componentes
pca = PCA(0.98)


# In[31]:


# Se aplica el mapeo al conjunto de datos de entrenamiento
X_train_pca = pca.fit_transform(X_train)


# In[32]:


# Se muestra el número de componentes calculado para al 98% de explicación de la varianza
print ( "Componentes para un 98% de explicación de varianza:   ", pca.n_components_ )


# In[33]:


# Se aplica el mapeo al conjunto de datos de test
X_test_pca = pca.transform(X_test)


# In[34]:


X_train_pca.shape


# In[35]:


X_test_pca.shape


# ### 3.2.2. Feature Importance

# In[36]:


# Se utiliza el modelo XGBoost para obtener las características importantes
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)


# In[37]:


# Se muestra gráficamente el top 20 de características más importantes
plot_importance(xgb_model,max_num_features = 20)
plt.show()


# In[38]:


# Se toman las 500 características más importantes para el conjunto de datos
selection = SelectFromModel(xgb_model, threshold=-np.inf, max_features=500, prefit=True)
X_train_xgb = selection.transform(X_train)
X_test_xgb = selection.transform(X_test)


# In[39]:


X_train_xgb.shape


# In[40]:


X_test_xgb.shape


# ## 3.3 Técnicas de muestreo

# <div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
# Se aplican las técnicas de muestreo sobre los datos obtenidos con la aplicación del PCA
# </div>

# In[41]:


rus_pca = RandomUnderSampler(random_state=2020)
X_train_rus_pca, y_train_rus_pca = rus_pca.fit_resample(X_train_pca, y_train)


# In[42]:


sm_pca = SMOTE(sampling_strategy='minority',random_state=2020)
X_train_sm_pca, y_train_sm_pca = sm_pca.fit_resample(X_train_pca, y_train)


# In[43]:


y_vals, counts = np.unique(y_test, return_counts=True)

y_vals_rus_pca, counts_rus_pca = np.unique(y_train_rus_pca, return_counts=True)
y_vals_sm_pca, counts_sm_pca = np.unique(y_train_sm_pca, return_counts=True)

print('Clases en conjunto de entrenamiento:',dict(zip(y_vals, counts)),'\n',
      'Clases en conjunto de entrenamiento con PCA y RandomUnderSampler:',dict(zip(y_vals_rus_pca, counts_rus_pca)),'\n',
      'Clases en conjunto de entrenamiento con PCA y SMOTE:',dict(zip(y_vals_sm_pca, counts_sm_pca)),'\n',
     )


# In[44]:


# Formato gráfico para evidenciar la corrección del desbalanceo en los datos de PCA y uno de los métodos

ax = y_train_rus_pca.value_counts().to_frame().plot(kind='bar')

totals = []

for i in ax.patches:
    totals.append(i.get_height())

total = sum(totals)

for i in ax.patches:
    ax.text(i.get_x()+.05, i.get_height()+.5,             str(round((i.get_height()/total)*100, 2))+'%', fontsize=10)


# In[45]:


X_train_rus_pca.shape


# In[46]:


X_train_sm_pca.shape


# <div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
# Se aplican las técnicas de muestreo sobre los datos obtenidos con la aplicación del XGBoost
# </div>

# In[47]:


rus_xgb = RandomUnderSampler(random_state=2020)
X_train_rus_xgb, y_train_rus_xgb = rus_xgb.fit_resample(X_train_xgb, y_train)


# In[48]:


sm_xgb = SMOTE(sampling_strategy='minority',random_state=2020)
X_train_sm_xgb, y_train_sm_xgb = sm_xgb.fit_resample(X_train_xgb, y_train)


# In[49]:


y_vals_rus_xgb, counts_rus_xgb = np.unique(y_train_rus_xgb, return_counts=True)
y_vals_sm_xgb, counts_sm_xgb = np.unique(y_train_sm_xgb, return_counts=True)

print('Clases en conjunto de entrenamiento:',dict(zip(y_vals, counts)),'\n',
      'Clases en conjunto de entrenamiento con XGB y RandomUnderSampler:',dict(zip(y_vals_rus_xgb, counts_rus_xgb)),'\n',
      'Clases en conjunto de entrenamiento con XGB y SMOTE:',dict(zip(y_vals_sm_xgb, counts_sm_xgb)),'\n',
     )


# In[50]:


# Formato gráfico para evidenciar la corrección del desbalanceo en los datos de XGBoost y uno de los métodos

ax = y_train_rus_xgb.value_counts().to_frame().plot(kind='bar')

totals = []

for i in ax.patches:
    totals.append(i.get_height())

total = sum(totals)

for i in ax.patches:
    ax.text(i.get_x()+.05, i.get_height()+.5,             str(round((i.get_height()/total)*100, 2))+'%', fontsize=10)


# In[51]:


X_train_rus_xgb.shape


# In[52]:


X_train_sm_xgb.shape


# <div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
# Dada la gran cantidad de registros obtenidos en los metodos de sobremuestreo, antes de pasar al modelado, vamos a analizar si realmente necesitamos tantos datos.
# 
# Para ello se realiza el siguiente análisis:
# <ul>
# <li>Submuestreo aleatorio de la clase negativa con diferentes tamaños respecto a los casos positivos (1:2, 1:5, 1:10, 1:15)</li>
# <li>A continuación, aplicar sobremuestreo de la clase negativa y aprender un modelo</li>
# </ul>    
#     
# Este análisis se realiza con 10 submuestreos aleatorios diferentes para evitar sesgo en las muestras.
# </div>

# In[53]:


#TO-DO. Revisar y mejorar disposición del código
start_time = time.time()
ratios=[1/2, 1/5, 1/10, 1/15]
for ratio in ratios:
    acc_scores    = []
    prec_scores   = []
    recall_scores = []
    f1_scores     = []
    mcc_scores    = [] 
    for i in range(10):
        rus = RandomUnderSampler(sampling_strategy=ratio, random_state=i)
        X_train_rus, y_train_rus = rus.fit_resample(X_train_pca, y_train)
        print("Ratio: {:.2f}x). Iteración: {}. Tamaño resultado submuestreo: {}\n".format(1/ratio, i, X_train_rus.shape))
        sm = SMOTE(sampling_strategy='minority',random_state=i)
        X_train_rus_sm, y_train_rus_sm = sm.fit_resample(X_train_rus, y_train_rus)
        print("Ratio: {:.2f}x). Iteración: {}. Tamaño resultado sobremuestreo: {}\n".format(1/ratio, i, X_train_rus_sm.shape))
        rf_clf_rus_sm = ensemble.RandomForestClassifier(n_estimators=50, random_state=i, n_jobs=3)
        rf_clf_rus_sm_pca = rf_clf_rus_sm.fit(X_train_rus_sm, y_train_rus_sm)
        y_pred_rf_rus_sm_pca = rf_clf_rus_sm_pca.predict(X_test_pca)
        acc = accuracy_score(y_test,y_pred_rf_rus_sm_pca)*100
        prec = precision_score(y_test,y_pred_rf_rus_sm_pca)*100
        rec = recall_score(y_test,y_pred_rf_rus_sm_pca)*100
        f1s = f1_score(y_test,y_pred_rf_rus_sm_pca)*100
        mcc = matthews_corrcoef(y_test,y_pred_rf_rus_sm_pca)*100
        print("Ratio: {:.2f}x). Iteración: {}. Accuracy:  {:.2f}\n".format(1/ratio, i, acc))
        print("Ratio: {:.2f}x). Iteración: {}. Precision: {:.2f}\n".format(1/ratio, i, prec))
        print("Ratio: {:.2f}x). Iteración: {}. Recall:    {:.2f}\n".format(1/ratio, i, rec))
        print("Ratio: {:.2f}x). Iteración: {}. F1 score:  {:.2f}\n".format(1/ratio, i, f1s))
        print("Ratio: {:.2f}x). Iteración: {}. MCC:       {:.2f}\n".format(1/ratio, i, mcc))
        acc_scores.append(acc)
        prec_scores.append(prec)
        recall_scores.append(rec)
        f1_scores.append(f1s)
        mcc_scores.append(mcc)
    print("Ratio: {:.2f}x). Accuracy promedio:  {:.2f}\n".format(1/ratio, np.mean(acc_scores)))
    print("Ratio: {:.2f}x). Precision promedio: {:.2f}\n".format(1/ratio, np.mean(prec_scores)))
    print("Ratio: {:.2f}x). Recall promedio:    {:.2f}\n".format(1/ratio, np.mean(recall_scores)))
    print("Ratio: {:.2f}x). F1 score promedio:  {:.2f}\n".format(1/ratio, np.mean(f1_scores)))
    print("Ratio: {:.2f}x). MCC promedio:       {:.2f}\n\n\n".format(1/ratio, np.mean(mcc_scores)))
    
print("--- %s segundos ---" % (time.time() - start_time))


# <div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
# Se amplia el experimiento a analizar los resultados para los ratios 1:20, 1:30 y 1:50
# </div>

# In[54]:


#TO-DO. Revisar y mejorar disposición del código
start_time = time.time()
ratios=[1/20, 1/30, 1/50]
for ratio in ratios:
    acc_scores    = []
    prec_scores   = []
    recall_scores = []
    f1_scores     = []
    mcc_scores    = [] 
    for i in range(10):
        rus = RandomUnderSampler(sampling_strategy=ratio, random_state=i)
        X_train_rus, y_train_rus = rus.fit_resample(X_train_pca, y_train)
        print("Ratio: {:.2f}x). Iteración: {}. Tamaño resultado submuestreo: {}\n".format(1/ratio, i, X_train_rus.shape))
        sm = SMOTE(sampling_strategy='minority',random_state=i)
        X_train_rus_sm, y_train_rus_sm = sm.fit_resample(X_train_rus, y_train_rus)
        print("Ratio: {:.2f}x). Iteración: {}. Tamaño resultado sobremuestreo: {}\n".format(1/ratio, i, X_train_rus_sm.shape))
        rf_clf_rus_sm = ensemble.RandomForestClassifier(n_estimators=50, random_state=i, n_jobs=3)
        rf_clf_rus_sm_pca = rf_clf_rus_sm.fit(X_train_rus_sm, y_train_rus_sm)
        y_pred_rf_rus_sm_pca = rf_clf_rus_sm_pca.predict(X_test_pca)
        acc = accuracy_score(y_test,y_pred_rf_rus_sm_pca)*100
        prec = precision_score(y_test,y_pred_rf_rus_sm_pca)*100
        rec = recall_score(y_test,y_pred_rf_rus_sm_pca)*100
        f1s = f1_score(y_test,y_pred_rf_rus_sm_pca)*100
        mcc = matthews_corrcoef(y_test,y_pred_rf_rus_sm_pca)*100
        print("Ratio: {:.2f}x). Iteración: {}. Accuracy:  {:.2f}\n".format(1/ratio, i, acc))
        print("Ratio: {:.2f}x). Iteración: {}. Precision: {:.2f}\n".format(1/ratio, i, prec))
        print("Ratio: {:.2f}x). Iteración: {}. Recall:    {:.2f}\n".format(1/ratio, i, rec))
        print("Ratio: {:.2f}x). Iteración: {}. F1 score:  {:.2f}\n".format(1/ratio, i, f1s))
        print("Ratio: {:.2f}x). Iteración: {}. MCC:       {:.2f}\n".format(1/ratio, i, mcc))
        acc_scores.append(acc)
        prec_scores.append(prec)
        recall_scores.append(rec)
        f1_scores.append(f1s)
        mcc_scores.append(mcc)
    print("Ratio: {:.2f}x). Accuracy promedio:  {:.2f}\n".format(1/ratio, np.mean(acc_scores)))
    print("Ratio: {:.2f}x). Precision promedio: {:.2f}\n".format(1/ratio, np.mean(prec_scores)))
    print("Ratio: {:.2f}x). Recall promedio:    {:.2f}\n".format(1/ratio, np.mean(recall_scores)))
    print("Ratio: {:.2f}x). F1 score promedio:  {:.2f}\n".format(1/ratio, np.mean(f1_scores)))
    print("Ratio: {:.2f}x). MCC promedio:       {:.2f}\n\n\n".format(1/ratio, np.mean(mcc_scores)))
    
print("--- %s segundos ---" % (time.time() - start_time))


# <div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
# Al observar que el resultado sigue mejorando, se amplia el experimiento a analizar los resultados para los ratios 1:100, 1:150 y 1:170
# </div>

# In[55]:


#TO-DO. Revisar y mejorar disposición del código
start_time = time.time()
ratios=[1/100, 1/150, 1/170]
for ratio in ratios:
    acc_scores    = []
    prec_scores   = []
    recall_scores = []
    f1_scores     = []
    mcc_scores    = [] 
    for i in range(10):
        rus = RandomUnderSampler(sampling_strategy=ratio, random_state=i)
        X_train_rus, y_train_rus = rus.fit_resample(X_train_pca, y_train)
        print("Ratio: {:.2f}x). Iteración: {}. Tamaño resultado submuestreo: {}\n".format(1/ratio, i, X_train_rus.shape))
        sm = SMOTE(sampling_strategy='minority',random_state=i)
        X_train_rus_sm, y_train_rus_sm = sm.fit_resample(X_train_rus, y_train_rus)
        print("Ratio: {:.2f}x). Iteración: {}. Tamaño resultado sobremuestreo: {}\n".format(1/ratio, i, X_train_rus_sm.shape))
        rf_clf_rus_sm = ensemble.RandomForestClassifier(n_estimators=50, random_state=i, n_jobs=3)
        rf_clf_rus_sm_pca = rf_clf_rus_sm.fit(X_train_rus_sm, y_train_rus_sm)
        y_pred_rf_rus_sm_pca = rf_clf_rus_sm_pca.predict(X_test_pca)
        acc = accuracy_score(y_test,y_pred_rf_rus_sm_pca)*100
        prec = precision_score(y_test,y_pred_rf_rus_sm_pca)*100
        rec = recall_score(y_test,y_pred_rf_rus_sm_pca)*100
        f1s = f1_score(y_test,y_pred_rf_rus_sm_pca)*100
        mcc = matthews_corrcoef(y_test,y_pred_rf_rus_sm_pca)*100
        print("Ratio: {:.2f}x). Iteración: {}. Accuracy:  {:.2f}\n".format(1/ratio, i, acc))
        print("Ratio: {:.2f}x). Iteración: {}. Precision: {:.2f}\n".format(1/ratio, i, prec))
        print("Ratio: {:.2f}x). Iteración: {}. Recall:    {:.2f}\n".format(1/ratio, i, rec))
        print("Ratio: {:.2f}x). Iteración: {}. F1 score:  {:.2f}\n".format(1/ratio, i, f1s))
        print("Ratio: {:.2f}x). Iteración: {}. MCC:       {:.2f}\n".format(1/ratio, i, mcc))
        acc_scores.append(acc)
        prec_scores.append(prec)
        recall_scores.append(rec)
        f1_scores.append(f1s)
        mcc_scores.append(mcc)
    print("Ratio: {:.2f}x). Accuracy promedio:  {:.2f}\n".format(1/ratio, np.mean(acc_scores)))
    print("Ratio: {:.2f}x). Precision promedio: {:.2f}\n".format(1/ratio, np.mean(prec_scores)))
    print("Ratio: {:.2f}x). Recall promedio:    {:.2f}\n".format(1/ratio, np.mean(recall_scores)))
    print("Ratio: {:.2f}x). F1 score promedio:  {:.2f}\n".format(1/ratio, np.mean(f1_scores)))
    print("Ratio: {:.2f}x). MCC promedio:       {:.2f}\n\n\n".format(1/ratio, np.mean(mcc_scores)))
    
print("--- %s segundos ---" % (time.time() - start_time))


# <div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
# Para los modelos, se tomara la técnica de muestreo con ratio 1:15 y ratio 1:50 de números positivos sobre los datos obtenidos con la aplicación del PCA y XGBoost
# </div>

# In[56]:


ratio = 1/15

rus_pca_15 = RandomUnderSampler(sampling_strategy=ratio, random_state=2020)
X_train_rus_pca_15, y_train_rus_pca_15 = rus_pca_15.fit_resample(X_train_pca, y_train)
        
rus_sm_pca_15 = SMOTE(sampling_strategy='minority',random_state=2020)
X_train_rus_sm_pca_15, y_train_rus_sm_pca_15 = rus_sm_pca_15.fit_resample(X_train_rus_pca_15, y_train_rus_pca_15)


# In[57]:


rus_xgb_15 = RandomUnderSampler(sampling_strategy=ratio, random_state=2020)
X_train_rus_xgb_15, y_train_rus_xgb_15 = rus_xgb_15.fit_resample(X_train_xgb, y_train)
        
rus_sm_xgb_15 = SMOTE(sampling_strategy='minority',random_state=2020)
X_train_rus_sm_xgb_15, y_train_rus_sm_xgb_15 = rus_sm_xgb_15.fit_resample(X_train_rus_xgb_15, y_train_rus_xgb_15)


# In[58]:


ratio = 1/50

rus_pca_50 = RandomUnderSampler(sampling_strategy=ratio, random_state=2020)
X_train_rus_pca_50, y_train_rus_pca_50 = rus_pca_50.fit_resample(X_train_pca, y_train)
        
rus_sm_pca_50 = SMOTE(sampling_strategy='minority',random_state=2020)
X_train_rus_sm_pca_50, y_train_rus_sm_pca_50 = rus_sm_pca_50.fit_resample(X_train_rus_pca_50, y_train_rus_pca_50)


# In[59]:


rus_xgb_50 = RandomUnderSampler(sampling_strategy=ratio, random_state=2020)
X_train_rus_xgb_50, y_train_rus_xgb_50 = rus_xgb_50.fit_resample(X_train_xgb, y_train)
        
rus_sm_xgb_50 = SMOTE(sampling_strategy='minority',random_state=2020)
X_train_rus_sm_xgb_50, y_train_rus_sm_xgb_50 = rus_sm_xgb_50.fit_resample(X_train_rus_xgb_50, y_train_rus_xgb_50)


# In[60]:


#Actualizamos los datos incluyendo el nuevo método de muestreo
y_vals_rus_sm_pca_15, counts_rus_sm_pca_15 = np.unique(y_train_rus_sm_pca_15, return_counts=True)
y_vals_rus_sm_pca_50, counts_rus_sm_pca_50 = np.unique(y_train_rus_sm_pca_50, return_counts=True)

print('Clases en conjunto de entrenamiento:',dict(zip(y_vals, counts)),'\n',
      'Clases en conjunto de entrenamiento con PCA y RandomUnderSampler (1:1):',dict(zip(y_vals_rus_pca, counts_rus_pca)),'\n',
      'Clases en conjunto de entrenamiento con PCA y SMOTE (1:171):',dict(zip(y_vals_sm_pca, counts_sm_pca)),'\n',
      'Clases en conjunto de entrenamiento con PCA y RandomUnderSampler+SMOTE (1:15):',dict(zip(y_vals_rus_sm_pca_15, counts_rus_sm_pca_15)),'\n',
      'Clases en conjunto de entrenamiento con PCA y RandomUnderSampler+SMOTE (1:50):',dict(zip(y_vals_rus_sm_pca_50, counts_rus_sm_pca_50)),'\n',
     )

y_vals_rus_sm_xgb_15, counts_rus_sm_xgb_15 = np.unique(y_train_rus_sm_xgb_15, return_counts=True)
y_vals_rus_sm_xgb_50, counts_rus_sm_xgb_50 = np.unique(y_train_rus_sm_xgb_50, return_counts=True)

print('Clases en conjunto de entrenamiento:',dict(zip(y_vals, counts)),'\n',
      'Clases en conjunto de entrenamiento con XGB y RandomUnderSampler (1:1):',dict(zip(y_vals_rus_xgb, counts_rus_xgb)),'\n',
      'Clases en conjunto de entrenamiento con XGB y SMOTE (1:171):',dict(zip(y_vals_sm_xgb, counts_sm_xgb)),'\n',
      'Clases en conjunto de entrenamiento con XGB y RandomUnderSampler+SMOTE (1:15):',dict(zip(y_vals_rus_sm_xgb_15, counts_rus_sm_xgb_15)),'\n',
      'Clases en conjunto de entrenamiento con XGB y RandomUnderSampler+SMOTE (1:50):',dict(zip(y_vals_rus_sm_xgb_50, counts_rus_sm_xgb_50)),'\n',
     )


# In[61]:


X_train_rus_sm_pca_15.shape


# In[62]:


X_train_rus_sm_xgb_15.shape


# In[63]:


X_train_rus_sm_pca_50.shape


# In[65]:


X_train_rus_sm_xgb_50.shape


# # 4. Modelado

# ## 4.1. Random Forest

# ### 4.1.1. Hiperparámetros

# <div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">La idea básica del bagging es utilizar el conjunto de entrenamiento original para generar centenares o miles de conjuntos similares usando muestreo con reemplazo. En este concepto está basado el algoritmo Random Forest, la combinación de varios árboles de decisión, cada uno entrenado con una realización diferente de los datos. La decisión final del clasificador combinado (la Random Forest) se toma por mayoría, dando el mismo peso a todas las decisiones parciales tomadas por los clasificadores base (los árboles).
# </div>

# <div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
# Para decidir cuáles son los hiperparámetros óptimos se utiliza una búsqueda de rejilla (grid search), es decir, se entrena un modelo para cada combinación de hiperparámetros posible y se evalua utilizando validación cruzada (cross validation) con 3 particiones estratificadas. 
# Posteriormente se selecciona la combinación de hiperparámetros que mejor resultados haya dado.
# </div>

# In[66]:


param_grid = {
    "n_estimators" : [50, 100, 200],
    "max_depth"    : [8, 10, 20],
    "random_state" : [2020],
}

rf_clf_gs = GridSearchCV(ensemble.RandomForestClassifier(), param_grid=param_grid, cv=3, pre_dispatch=6, n_jobs=3,scoring='f1',verbose=0)


# In[67]:


# Se toma como conjunto de entrenamiento para determinar los mejores parámetros los resultantes de PCA + combinar los muestreos 
rf_clf_gs_rus_sm_pca_15 = rf_clf_gs.fit(X_train_rus_sm_pca_15, y_train_rus_sm_pca_15)


# In[68]:


rf_clf_gs_rus_sm_pca_15.best_params_


# In[69]:


means = rf_clf_gs_rus_sm_pca_15.cv_results_["mean_test_score"]
stds = rf_clf_gs_rus_sm_pca_15.cv_results_["std_test_score"]
params = rf_clf_gs_rus_sm_pca_15.cv_results_['params']
ranks = rf_clf_gs_rus_sm_pca_15.cv_results_['rank_test_score']

for rank, mean, std, pms in zip(ranks, means, stds, params):
    print("{}) Precisión media: {:.2f} +/- {:.2f} con parámetros {}".format(rank, mean*100, std*100, pms))


# In[70]:


# Se toma como conjunto de entrenamiento para determinar los mejores parámetros los resultantes de XGB + combinar los muestreos 
rf_clf_gs_rus_sm_xgb_15 = rf_clf_gs.fit(X_train_rus_sm_xgb_15, y_train_rus_sm_xgb_15)


# In[71]:


rf_clf_gs_rus_sm_xgb_15.best_params_


# In[72]:


means = rf_clf_gs_rus_sm_xgb_15.cv_results_["mean_test_score"]
stds = rf_clf_gs_rus_sm_xgb_15.cv_results_["std_test_score"]
params = rf_clf_gs_rus_sm_xgb_15.cv_results_['params']
ranks = rf_clf_gs_rus_sm_xgb_15.cv_results_['rank_test_score']

for rank, mean, std, pms in zip(ranks, means, stds, params):
    print("{}) Precisión media: {:.2f} +/- {:.2f} con parámetros {}".format(rank, mean*100, std*100, pms))


# ### 4.1.2. Ejecución inicial

# <div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">    
# A partir de los conjuntos de datos obtenidos de la reducción dimensional y el muestreo, se entrena un modelo <i>Random Forest</i> con los mejores parámetros obtenidos del <i>grid search</i>.
# </div>

# In[73]:


rf_clf = ensemble.RandomForestClassifier(n_estimators=rf_clf_gs_rus_sm_pca_15.best_params_["n_estimators"], max_depth=rf_clf_gs_rus_sm_pca_15.best_params_["max_depth"], random_state=rf_clf_gs_rus_sm_pca_15.best_params_["random_state"], n_jobs=3, verbose=0)


# <div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">    
#     Ramdom Forest con el conjunto de datos PCA y RandomUnderSampler
# </div>

# In[74]:


rf_cfl_rus_pca = rf_clf.fit(X_train_rus_pca, y_train_rus_pca)


# In[75]:


y_pred_rf_rus_pca = rf_cfl_rus_pca.predict(X_test_pca)


# In[76]:


# Se muestran los resultados a través de la precisión de las predicciones y la matriz de confusión de cada modelo
# Se utiliza una función para mostrar de forma gráfica la matriz de confusión
# NOTA: código extraído de 20182 M2.855 PEC 3 modificado para poder mostrar los datos normalizados o sin normalizar

def plot_confusion_matrix_custom(cm, normalize):
    classes = ["0", "1"]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
   
    cmap=plt.cm.Blues

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Real')
    plt.xlabel('Predicción')
    


# In[77]:


cm_rf_rus_pca = confusion_matrix(y_test, y_pred_rf_rus_pca)  

plot_confusion_matrix_custom(cm_rf_rus_pca, False)


# In[78]:


plot_confusion_matrix_custom(cm_rf_rus_pca, True)

print('Accuracy:  ', accuracy_score(y_test,y_pred_rf_rus_pca)*100)  
print('Precision: ', precision_score(y_test,y_pred_rf_rus_pca)*100)  
print('Recall:    ', recall_score(y_test,y_pred_rf_rus_pca)*100)  
print('F1 score:  ', f1_score(y_test,y_pred_rf_rus_pca)*100)  
print('MCC:       ', matthews_corrcoef(y_test,y_pred_rf_rus_pca)*100) 


# <div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">    
#     Ramdom Forest con el conjunto de datos PCA y SMOTE
# </div>

# In[79]:


rf_cfl_sm_pca = rf_clf.fit(X_train_sm_pca, y_train_sm_pca)


# In[80]:


y_pred_rf_sm_pca = rf_cfl_sm_pca.predict(X_test_pca)


# In[81]:


cm_rf_sm_pca = confusion_matrix(y_test, y_pred_rf_sm_pca)  

plot_confusion_matrix_custom(cm_rf_sm_pca,False)


# In[82]:


plot_confusion_matrix_custom(cm_rf_sm_pca,True)

print('Accuracy:  ', accuracy_score(y_test,y_pred_rf_sm_pca)*100)  
print('Precision: ', precision_score(y_test,y_pred_rf_sm_pca)*100)  
print('Recall:    ', recall_score(y_test,y_pred_rf_sm_pca)*100)  
print('F1 score:  ', f1_score(y_test,y_pred_rf_sm_pca)*100)  
print('MCC:       ', matthews_corrcoef(y_test,y_pred_rf_sm_pca)*100) 


# <div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">    
#     Ramdom Forest con el conjunto de datos PCA y RandomUnderSampler + SMOTE (1:15)
# </div>

# In[83]:


rf_cfl_rus_sm_pca_15 = rf_clf.fit(X_train_rus_sm_pca_15, y_train_rus_sm_pca_15)


# In[84]:


y_pred_rf_rus_sm_pca_15 = rf_cfl_rus_sm_pca_15.predict(X_test_pca)


# In[85]:


cm_rf_rus_sm_pca_15 = confusion_matrix(y_test, y_pred_rf_rus_sm_pca_15)  

plot_confusion_matrix_custom(cm_rf_rus_sm_pca_15, False)


# In[86]:


plot_confusion_matrix_custom(cm_rf_rus_sm_pca_15, True)

print('Accuracy:  ', accuracy_score(y_test,y_pred_rf_rus_sm_pca_15)*100)  
print('Precision: ', precision_score(y_test,y_pred_rf_rus_sm_pca_15)*100)  
print('Recall:    ', recall_score(y_test,y_pred_rf_rus_sm_pca_15)*100)  
print('F1 score:  ', f1_score(y_test,y_pred_rf_rus_sm_pca_15)*100)  
print('MCC:       ', matthews_corrcoef(y_test,y_pred_rf_rus_sm_pca_15)*100) 


# <div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">    
#     Ramdom Forest con el conjunto de datos PCA y RandomUnderSampler + SMOTE (1:50)
# </div>

# In[87]:


rf_cfl_rus_sm_pca_50 = rf_clf.fit(X_train_rus_sm_pca_50, y_train_rus_sm_pca_50)


# In[88]:


y_pred_rf_rus_sm_pca_50 = rf_cfl_rus_sm_pca_50.predict(X_test_pca)


# In[89]:


cm_rf_rus_sm_pca_50 = confusion_matrix(y_test, y_pred_rf_rus_sm_pca_50)  

plot_confusion_matrix_custom(cm_rf_rus_sm_pca_50, False)


# In[90]:


plot_confusion_matrix_custom(cm_rf_rus_sm_pca_50, True)

print('Accuracy:  ', accuracy_score(y_test,y_pred_rf_rus_sm_pca_50)*100)  
print('Precision: ', precision_score(y_test,y_pred_rf_rus_sm_pca_50)*100)  
print('Recall:    ', recall_score(y_test,y_pred_rf_rus_sm_pca_50)*100)  
print('F1 score:  ', f1_score(y_test,y_pred_rf_rus_sm_pca_50)*100)  
print('MCC:       ', matthews_corrcoef(y_test,y_pred_rf_rus_sm_pca_50)*100) 


# <div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">    
#     Ramdom Forest con el conjunto de datos XGBoost y RandomUnderSampler
# </div>

# In[91]:


rf_cfl_rus_xgb = rf_clf.fit(X_train_rus_xgb, y_train_rus_xgb)


# In[92]:


y_pred_rf_rus_xgb = rf_cfl_rus_xgb.predict(X_test_xgb)


# In[93]:


cm_rf_rus_xgb = confusion_matrix(y_test, y_pred_rf_rus_xgb)  

plot_confusion_matrix_custom(cm_rf_rus_xgb, False)


# In[94]:


plot_confusion_matrix_custom(cm_rf_rus_xgb, True)

print('Accuracy:  ', accuracy_score(y_test,y_pred_rf_rus_xgb)*100)  
print('Precision: ', precision_score(y_test,y_pred_rf_rus_xgb)*100)  
print('Recall:    ', recall_score(y_test,y_pred_rf_rus_xgb)*100)  
print('F1 score:  ', f1_score(y_test,y_pred_rf_rus_xgb)*100)  
print('MCC:       ', matthews_corrcoef(y_test,y_pred_rf_rus_xgb)*100) 


# <div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">    
#     Ramdom Forest con el conjunto de datos XGBoost y SMOTE
# </div>

# In[95]:


rf_cfl_sm_xgb = rf_clf.fit(X_train_sm_xgb, y_train_sm_xgb)


# In[96]:


y_pred_rf_sm_xgb = rf_cfl_sm_xgb.predict(X_test_xgb)


# In[97]:


cm_rf_sm_xgb = confusion_matrix(y_test, y_pred_rf_sm_xgb)  

plot_confusion_matrix_custom(cm_rf_sm_xgb, False)


# In[98]:


plot_confusion_matrix_custom(cm_rf_sm_xgb, True)

print('Accuracy:  ', accuracy_score(y_test,y_pred_rf_sm_xgb)*100)  
print('Precision: ', precision_score(y_test,y_pred_rf_sm_xgb)*100)  
print('Recall:    ', recall_score(y_test,y_pred_rf_sm_xgb)*100)  
print('F1 score:  ', f1_score(y_test,y_pred_rf_sm_xgb)*100)  
print('MCC:       ', matthews_corrcoef(y_test,y_pred_rf_sm_xgb)*100)


# <div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">    
#     Ramdom Forest con el conjunto de datos XGBoost y RandomUnderSampler + SMOTE (1:15)
# </div>

# In[99]:


rf_cfl_rus_sm_xgb_15 = rf_clf.fit(X_train_rus_sm_xgb_15, y_train_rus_sm_xgb_15)


# In[100]:


y_pred_rf_rus_sm_xgb_15 = rf_cfl_rus_sm_xgb_15.predict(X_test_xgb)


# In[101]:


cm_rf_rus_sm_xgb_15 = confusion_matrix(y_test, y_pred_rf_rus_sm_xgb_15)  

plot_confusion_matrix_custom(cm_rf_rus_sm_xgb_15, False)


# In[102]:


plot_confusion_matrix_custom(cm_rf_rus_sm_xgb_15, True)

print('Accuracy:  ', accuracy_score(y_test,y_pred_rf_rus_sm_xgb_15)*100)  
print('Precision: ', precision_score(y_test,y_pred_rf_rus_sm_xgb_15)*100)  
print('Recall:    ', recall_score(y_test,y_pred_rf_rus_sm_xgb_15)*100)  
print('F1 score:  ', f1_score(y_test,y_pred_rf_rus_sm_xgb_15)*100)  
print('MCC:       ', matthews_corrcoef(y_test,y_pred_rf_rus_sm_xgb_15)*100)


# <div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">    
#     Ramdom Forest con el conjunto de datos XGBoost y RandomUnderSampler + SMOTE (1:50)
# </div>

# In[103]:


rf_cfl_rus_sm_xgb_50 = rf_clf.fit(X_train_rus_sm_xgb_50, y_train_rus_sm_xgb_50)


# In[104]:


y_pred_rf_rus_sm_xgb_50 = rf_cfl_rus_sm_xgb_50.predict(X_test_xgb)


# In[105]:


cm_rf_rus_sm_xgb_50 = confusion_matrix(y_test, y_pred_rf_rus_sm_xgb_50)  

plot_confusion_matrix_custom(cm_rf_rus_sm_xgb_50, False)


# In[106]:


plot_confusion_matrix_custom(cm_rf_rus_sm_xgb_50, True)

print('Accuracy:  ', accuracy_score(y_test,y_pred_rf_rus_sm_xgb_50)*100)  
print('Precision: ', precision_score(y_test,y_pred_rf_rus_sm_xgb_50)*100)  
print('Recall:    ', recall_score(y_test,y_pred_rf_rus_sm_xgb_50)*100)  
print('F1 score:  ', f1_score(y_test,y_pred_rf_rus_sm_xgb_50)*100)  
print('MCC:       ', matthews_corrcoef(y_test,y_pred_rf_rus_sm_xgb_50)*100)


# ## 4.2. eXtreme Gradient Boosting (XGBoost)

# ### 4.2.1. Hiperparámetros

# <div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
# En el sistema de Boosting se combinan varios clasificadores débiles secuencialmente, y en cada uno de ellos se da más peso a los datos que han sido erróneamente clasificados en las combinaciones anteriores, para que se concentre así en los casos más difíciles de resolver.
# </div>

# <div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
# Para decidir cuáles son los hiperparámetros óptimos se utiliza una búsqueda de rejilla (grid search), es decir, se entrena un modelo para cada combinación de hiperparámetros posible y se evalua utilizando validación cruzada (cross validation) con 3 particiones estratificadas. 
# Posteriormente se selecciona la combinación de hiperparámetros que mejor resultados haya dado.
# </div>

# In[107]:


param_grid = {
    "n_estimators"     : [50, 100, 200],
    "max_depth"        : [8, 10, 20],
    "min_child_weight" : [6, 8, 10],
    "random_state"     : [2020],
}

xgb_clf_gs = GridSearchCV(XGBClassifier(), param_grid=param_grid, scoring='f1', cv=3, pre_dispatch=6, n_jobs=3, verbose=0)


# In[108]:


# Se toma como conjunto de entrenamiento para determinar los mejores parámetros los resultantes de PCA + combinar los muestreos 
xgb_clf_gs_rus_sm_pca_15 = xgb_clf_gs.fit(X_train_rus_sm_pca_15, y_train_rus_sm_pca_15)


# In[109]:


xgb_clf_gs_rus_sm_pca_15.best_params_


# In[110]:


means = xgb_clf_gs_rus_sm_pca_15.cv_results_["mean_test_score"]
stds = xgb_clf_gs_rus_sm_pca_15.cv_results_["std_test_score"]
params = xgb_clf_gs_rus_sm_pca_15.cv_results_['params']
ranks = xgb_clf_gs_rus_sm_pca_15.cv_results_['rank_test_score']

for rank, mean, std, pms in zip(ranks, means, stds, params):
    print("{}) Precisión media: {:.2f} +/- {:.2f} con parámetros {}".format(rank, mean*100, std*100, pms))


# In[111]:


# Se toma como conjunto de entrenamiento para determinar los mejores parámetros los resultantes de XGB + combinar los muestreos 
xgb_clf_gs_rus_sm_xgb_15 = xgb_clf_gs.fit(X_train_rus_sm_xgb_15, y_train_rus_sm_xgb_15)


# In[112]:


xgb_clf_gs_rus_sm_xgb_15.best_params_


# In[113]:


means = xgb_clf_gs_rus_sm_xgb_15.cv_results_["mean_test_score"]
stds = xgb_clf_gs_rus_sm_xgb_15.cv_results_["std_test_score"]
params = xgb_clf_gs_rus_sm_xgb_15.cv_results_['params']
ranks = xgb_clf_gs_rus_sm_xgb_15.cv_results_['rank_test_score']

for rank, mean, std, pms in zip(ranks, means, stds, params):
    print("{}) Precisión media: {:.2f} +/- {:.2f} con parámetros {}".format(rank, mean*100, std*100, pms))


# ### 4.1.2. Ejecución inicial

# <div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">    
# A partir de los conjuntos de datos obtenidos de la reducción dimensional y el muestreo, se entrena un modelo <i>XGBoost</i> con los mejores parámetros obtenidos del <i>grid search</i>.
# </div>

# In[114]:


xgb_clf = XGBClassifier(n_estimators=xgb_clf_gs_rus_sm_pca_15.best_params_["n_estimators"], max_depth=xgb_clf_gs_rus_sm_pca_15.best_params_["max_depth"], min_child_weight=xgb_clf_gs_rus_sm_pca_15.best_params_["min_child_weight"], random_state=xgb_clf_gs_rus_sm_pca_15.best_params_["random_state"], subsample=0.8, scoring='f1', cv=3, pre_dispatch=6, n_jobs=3, verbose=0)


# <div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">    
#     XGBoost con el conjunto de datos PCA y RandomUnderSampler
# </div>

# In[115]:


xgb_cfl_rus_pca = xgb_clf.fit(X_train_rus_pca, y_train_rus_pca)


# In[116]:


y_pred_xgb_rus_pca = xgb_cfl_rus_pca.predict(X_test_pca)


# In[117]:


cm_xgb_rus_pca = confusion_matrix(y_test, y_pred_xgb_rus_pca)  

plot_confusion_matrix_custom(cm_xgb_rus_pca, False)


# In[118]:


plot_confusion_matrix_custom(cm_xgb_rus_pca, True)

print('Accuracy:  ', accuracy_score(y_test,y_pred_xgb_rus_pca)*100)  
print('Precision: ', precision_score(y_test,y_pred_xgb_rus_pca)*100)  
print('Recall:    ', recall_score(y_test,y_pred_xgb_rus_pca)*100)  
print('F1 score:  ', f1_score(y_test,y_pred_xgb_rus_pca)*100)
print('MCC:       ', matthews_corrcoef(y_test,y_pred_xgb_rus_pca)*100)


# <div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">    
#     XGBoost con el conjunto de datos PCA y SMOTE
# </div>

# In[119]:


xgb_cfl_sm_pca = xgb_clf.fit(X_train_sm_pca, y_train_sm_pca)


# In[120]:


y_pred_xgb_sm_pca = xgb_cfl_sm_pca.predict(X_test_pca)


# In[121]:


cm_xgb_sm_pca = confusion_matrix(y_test, y_pred_xgb_sm_pca)  

plot_confusion_matrix_custom(cm_xgb_sm_pca, False)


# In[122]:


plot_confusion_matrix_custom(cm_xgb_sm_pca, True)

print('Accuracy:  ', accuracy_score(y_test,y_pred_xgb_sm_pca)*100)  
print('Precision: ', precision_score(y_test,y_pred_xgb_sm_pca)*100)  
print('Recall:    ', recall_score(y_test,y_pred_xgb_sm_pca)*100)  
print('F1 score:  ', f1_score(y_test,y_pred_xgb_sm_pca)*100)
print('MCC:       ', matthews_corrcoef(y_test,y_pred_xgb_sm_pca)*100)


# <div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">    
#     XGBoost con el conjunto de datos PCA y RandomUnderSampler + SMOTE (1:15)
# </div>

# In[123]:


xgb_cfl_rus_sm_pca_15 = xgb_clf.fit(X_train_rus_sm_pca_15, y_train_rus_sm_pca_15)


# In[124]:


y_pred_xgb_rus_sm_pca_15 = xgb_cfl_rus_sm_pca_15.predict(X_test_pca)


# In[125]:


cm_xgb_rus_sm_pca_15 = confusion_matrix(y_test, y_pred_xgb_rus_sm_pca_15)  

plot_confusion_matrix_custom(cm_xgb_rus_sm_pca_15, False)


# In[126]:


plot_confusion_matrix_custom(cm_xgb_rus_sm_pca_15, True)

print('Accuracy:  ', accuracy_score(y_test,y_pred_xgb_rus_sm_pca_15)*100)  
print('Precision: ', precision_score(y_test,y_pred_xgb_rus_sm_pca_15)*100)  
print('Recall:    ', recall_score(y_test,y_pred_xgb_rus_sm_pca_15)*100)  
print('F1 score:  ', f1_score(y_test,y_pred_xgb_rus_sm_pca_15)*100)
print('MCC:       ', matthews_corrcoef(y_test,y_pred_xgb_rus_sm_pca_15)*100)


# <div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">    
#     XGBoost con el conjunto de datos PCA y RandomUnderSampler + SMOTE (1:50)
# </div>

# In[127]:


xgb_cfl_rus_sm_pca_50 = xgb_clf.fit(X_train_rus_sm_pca_50, y_train_rus_sm_pca_50)


# In[128]:


y_pred_xgb_rus_sm_pca_50 = xgb_cfl_rus_sm_pca_50.predict(X_test_pca)


# In[129]:


cm_xgb_rus_sm_pca_50 = confusion_matrix(y_test, y_pred_xgb_rus_sm_pca_50)  

plot_confusion_matrix_custom(cm_xgb_rus_sm_pca_50, False)


# In[130]:


plot_confusion_matrix_custom(cm_xgb_rus_sm_pca_50, True)

print('Accuracy:  ', accuracy_score(y_test,y_pred_xgb_rus_sm_pca_50)*100)  
print('Precision: ', precision_score(y_test,y_pred_xgb_rus_sm_pca_50)*100)  
print('Recall:    ', recall_score(y_test,y_pred_xgb_rus_sm_pca_50)*100)  
print('F1 score:  ', f1_score(y_test,y_pred_xgb_rus_sm_pca_50)*100)
print('MCC:       ', matthews_corrcoef(y_test,y_pred_xgb_rus_sm_pca_50)*100)


# <div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">    
#     XGBoost con el conjunto de datos XGBoost y RandomUnderSampler
# </div>

# In[131]:


xgb_cfl_rus_xgb = xgb_clf.fit(X_train_rus_xgb, y_train_rus_xgb)


# In[132]:


y_pred_xgb_rus_xgb = xgb_cfl_rus_xgb.predict(X_test_xgb)


# In[133]:


cm_xgb_rus_xgb = confusion_matrix(y_test, y_pred_xgb_rus_xgb)  

plot_confusion_matrix_custom(cm_xgb_rus_xgb, False)


# In[134]:


plot_confusion_matrix_custom(cm_xgb_rus_xgb, True)

print('Accuracy:  ', accuracy_score(y_test,y_pred_xgb_rus_xgb)*100)  
print('Precision: ', precision_score(y_test,y_pred_xgb_rus_xgb)*100)  
print('Recall:    ', recall_score(y_test,y_pred_xgb_rus_xgb)*100)  
print('F1 score:  ', f1_score(y_test,y_pred_xgb_rus_xgb)*100)
print('MCC:       ', matthews_corrcoef(y_test,y_pred_xgb_rus_xgb)*100)


# <div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">    
#     XGBoost con el conjunto de datos XGBoost y SMOTE
# </div>

# In[135]:


xgb_cfl_sm_xgb = xgb_clf.fit(X_train_sm_xgb, y_train_sm_xgb)


# In[136]:


y_pred_xgb_sm_xgb = xgb_cfl_sm_xgb.predict(X_test_xgb)


# In[137]:


cm_xgb_sm_xgb = confusion_matrix(y_test, y_pred_xgb_sm_xgb)  

plot_confusion_matrix_custom(cm_xgb_sm_xgb, False)


# In[138]:


plot_confusion_matrix_custom(cm_xgb_sm_xgb, True)

print('Accuracy:  ', accuracy_score(y_test,y_pred_xgb_sm_xgb)*100)  
print('Precision: ', precision_score(y_test,y_pred_xgb_sm_xgb)*100)  
print('Recall:    ', recall_score(y_test,y_pred_xgb_sm_xgb)*100)  
print('F1 score:  ', f1_score(y_test,y_pred_xgb_sm_xgb)*100)
print('MCC:       ', matthews_corrcoef(y_test,y_pred_xgb_sm_xgb)*100)


# <div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">    
#     XGBoost con el conjunto de datos XGBoost y y RandomUnderSampler + SMOTE (1:15)
# </div>

# In[139]:


xgb_cfl_rus_sm_xgb_15 = xgb_clf.fit(X_train_rus_sm_xgb_15, y_train_rus_sm_xgb_15)


# In[140]:


y_pred_xgb_rus_sm_xgb_15 = xgb_cfl_rus_sm_xgb_15.predict(X_test_xgb)


# In[141]:


cm_xgb_rus_sm_xgb_15 = confusion_matrix(y_test, y_pred_xgb_rus_sm_xgb_15)  

plot_confusion_matrix_custom(cm_xgb_rus_sm_xgb_15, False)


# In[142]:


plot_confusion_matrix_custom(cm_xgb_rus_sm_xgb_15, True)

print('Accuracy:  ', accuracy_score(y_test,y_pred_xgb_rus_sm_xgb_15)*100)  
print('Precision: ', precision_score(y_test,y_pred_xgb_rus_sm_xgb_15)*100)  
print('Recall:    ', recall_score(y_test,y_pred_xgb_rus_sm_xgb_15)*100)  
print('F1 score:  ', f1_score(y_test,y_pred_xgb_rus_sm_xgb_15)*100)
print('MCC:       ', matthews_corrcoef(y_test,y_pred_xgb_rus_sm_xgb_15)*100)


# <div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">    
#     XGBoost con el conjunto de datos XGBoost y y RandomUnderSampler + SMOTE (1:50)
# </div>

# In[143]:


xgb_cfl_rus_sm_xgb_50 = xgb_clf.fit(X_train_rus_sm_xgb_50, y_train_rus_sm_xgb_50)


# In[144]:


y_pred_xgb_rus_sm_xgb_50 = xgb_cfl_rus_sm_xgb_50.predict(X_test_xgb)


# In[145]:


cm_xgb_rus_sm_xgb_50 = confusion_matrix(y_test, y_pred_xgb_rus_sm_xgb_50)  

plot_confusion_matrix_custom(cm_xgb_rus_sm_xgb_50, False)


# In[146]:


plot_confusion_matrix_custom(cm_xgb_rus_sm_xgb_50, True)

print('Accuracy:  ', accuracy_score(y_test,y_pred_xgb_rus_sm_xgb_50)*100)  
print('Precision: ', precision_score(y_test,y_pred_xgb_rus_sm_xgb_50)*100)  
print('Recall:    ', recall_score(y_test,y_pred_xgb_rus_sm_xgb_50)*100)  
print('F1 score:  ', f1_score(y_test,y_pred_xgb_rus_sm_xgb_50)*100)
print('MCC:       ', matthews_corrcoef(y_test,y_pred_xgb_rus_sm_xgb_50)*100)


# # 5. Evaluación

# <div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;"> 
# En base a estos resultados, cabe la pregunta de: dato que se están utilizando dos modelos diferentes, si se combinan, ¿mejora algo? 
# 
# En este apartado se explora dentro de la combinación secuencial de modelos la técnica del stacking para obtener un clasificador combinando de ambos modelos ya utilizados.
# 
# </div>

# ## 5.1. Combinación secuencial de modelos stacking

# <div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;"> 
# El stacking es una técnica de combinación secuencial de clasificadores bases diferentes a través de un meta-clasificador.
# 
# En el caso concreto de este trabajo y por las limitaciones de recursos, se entrena el clasificador de stacking utilizando los conjuntos de datos de la combinación de muestreo con ratio 1:15 y ratio 1:50 del número de positivos para los clasificadores empleados en la etapa del modelado.
# 
# </div>

# In[147]:


# Se combinan los clasificadores en un mismo meta-clasificador 

estimators = [
     ('rf', ensemble.RandomForestClassifier(n_estimators=rf_clf_gs_rus_sm_pca_15.best_params_["n_estimators"], max_depth=rf_clf_gs_rus_sm_pca_15.best_params_["max_depth"], random_state=rf_clf_gs_rus_sm_pca_15.best_params_["random_state"], n_jobs=3, verbose=0)),
     ('xgb', XGBClassifier(n_estimators=xgb_clf_gs_rus_sm_pca_15.best_params_["n_estimators"], max_depth=xgb_clf_gs_rus_sm_pca_15.best_params_["max_depth"], random_state=xgb_clf_gs_rus_sm_pca_15.best_params_["random_state"], min_child_weight=xgb_clf_gs_rus_sm_xgb_15.best_params_["min_child_weight"], subsample=0.8, scoring='f1', cv=3, pre_dispatch=6, n_jobs=3, verbose=0))
 ]

stack_clf = StackingClassifier(estimators=estimators, cv=3, n_jobs=3, verbose=0)


# <div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;"> 
# Ratio 1:15
# </div>

# In[148]:


stack_cfl_rus_sm_xgb_15 = stack_clf.fit(X_train_rus_sm_xgb_15, y_train_rus_sm_xgb_15)


# In[149]:


y_pred_stack_rus_sm_xgb_15 = stack_cfl_rus_sm_xgb_15.predict(X_test_xgb)


# In[150]:


cm_stack_rus_sm_xgb_15 = confusion_matrix(y_test, y_pred_stack_rus_sm_xgb_15)  

plot_confusion_matrix_custom(cm_stack_rus_sm_xgb_15, False)


# In[151]:


plot_confusion_matrix_custom(cm_stack_rus_sm_xgb_15, True)

print('Accuracy:  ', accuracy_score(y_test,y_pred_stack_rus_sm_xgb_15)*100)  
print('Precision: ', precision_score(y_test,y_pred_stack_rus_sm_xgb_15)*100)  
print('Recall:    ', recall_score(y_test,y_pred_stack_rus_sm_xgb_15)*100)  
print('F1 score:  ', f1_score(y_test,y_pred_stack_rus_sm_xgb_15)*100)
print('MCC:       ', matthews_corrcoef(y_test,y_pred_stack_rus_sm_xgb_15)*100)


# <div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;"> 
# Ratio 1:50
# </div>

# In[152]:


stack_cfl_rus_sm_xgb_50 = stack_clf.fit(X_train_rus_sm_xgb_50, y_train_rus_sm_xgb_50)


# In[153]:


y_pred_stack_rus_sm_xgb_50 = stack_cfl_rus_sm_xgb_50.predict(X_test_xgb)


# In[154]:


cm_stack_rus_sm_xgb_50 = confusion_matrix(y_test, y_pred_stack_rus_sm_xgb_50)  

plot_confusion_matrix_custom(cm_stack_rus_sm_xgb_50, False)


# In[155]:


plot_confusion_matrix_custom(cm_stack_rus_sm_xgb_50, True)

print('Accuracy:  ', accuracy_score(y_test,y_pred_stack_rus_sm_xgb_50)*100)  
print('Precision: ', precision_score(y_test,y_pred_stack_rus_sm_xgb_50)*100)  
print('Recall:    ', recall_score(y_test,y_pred_stack_rus_sm_xgb_50)*100)  
print('F1 score:  ', f1_score(y_test,y_pred_stack_rus_sm_xgb_50)*100)
print('MCC:       ', matthews_corrcoef(y_test,y_pred_stack_rus_sm_xgb_50)*100)

