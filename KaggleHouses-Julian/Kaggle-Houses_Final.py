import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt


def EsteticaPlot(titulo,xlabel,ylabel,fig=None,xaxis=None,fsize=None,legend=None):
    fsize=fsize or 16
    fig=fig or titulo
    plt.grid(True)

    plt.title(titulo,fontsize=fsize+2)
    plt.xlabel(xlabel,fontsize=fsize)
    if xaxis is not None:#Para histogramas
        plt.xticks(xaxis)
    plt.ylabel(ylabel,fontsize=fsize)
    plt.tick_params(labelsize=fsize-3) #tamaño numeros eje
    if(legend==None): plt.legend(loc='best',fontsize=fsize-2,borderpad=0.7,draggable=1)
#%%
# Read the data
folder=os.getcwd()
folder=r"C:\Users\CNEA\Desktop\Gasanego B Julian\Cursos\ML\Kaggle-Houses"
os.chdir(folder)

data_full = pd.read_csv(folder+'/train.csv', index_col='Id')
y = data_full.SalePrice              
X_full=data_full.copy(); X_full.drop(['SalePrice'], axis=1, inplace=True)

X_test = pd.read_csv(folder+'/test.csv', index_col='Id')

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X_full, y, 
                                                      train_size=0.8, test_size=0.2, random_state=0)
aux=X_train.copy()
auxVal=X_valid.copy()
auxTest=X_test.copy()

# Categorical and Numerical columns in the training data
CatCols = [col for col in aux.columns if aux[col].dtype == "object"]
NumCols = list(set(aux.columns)-set(CatCols))
# aux.drop(CatCols, axis=1)

# Cols w too many NaN values.
miss = (X_train.isnull().sum()); miss=miss[miss > 0]
# print(miss.sort_values(ascending=False))
nEvent=len(X_train)
missNum=[col for col in miss.index if col in NumCols]
missCat=[col for col in miss.index if col in CatCols]

# KeepCols=X_train.columns
# aux=X_train[KeepCols].copy()
# auxVal=X_valid[KeepCols].copy()
# auxTest=X_test[KeepCols].copy()

# Check correlation for numerical cols
# saleprice correlation matrix
corrmat = data_full[NumCols+['SalePrice']].corr()                                  # DataFrame
k = min([10+1,int(len(corrmat.columns)/2)]) # número de variables que eligen para el mapa de calor
corrList=corrmat.nlargest(k, 'SalePrice')['SalePrice']# solo los usa para sacar índices de los k elementos de mayor valor de corr()
cols = corrList.index 
# Plot the values
cm = np.corrcoef(data_full[cols].values.T) # corr() del df
import seaborn as sns
sns.set(font_scale=1.25)
plt.subplots(figsize=(5,5))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# Remove num columns w too many nan values AND/OR low correlation with saleprice
DropNumCols=[col for col in miss.index if (miss[col]>0.4*nEvent)]
KeepNumCols=[col for col in corrList.index if corrList[col]>0.6 and col!='SalePrice']
auxList=[col for col in KeepNumCols if 
         col not in DropNumCols 
            or (miss[col]<0.2*nEvent and corrList[col]>=0.75)
         ];
KeepNumCols=auxList; 
OkNum=KeepNumCols

for col in miss.index:
    if(col not in KeepNumCols): continue
    print(col, miss[col])

# # Plot the corr table for the selected values
# cols2=["SalePrice"]+KeepNumCols; cols2=corrList[cols2].index
# cm = np.corrcoef(data_full[cols2].values.T) # corr() del df
# import seaborn as sns
# sns.set(font_scale=1.25)
# plt.subplots(figsize=(5,5))
# hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols2.values, xticklabels=cols2.values)
# plt.show()


# Check # unique values on catetgorical columns
obj_nunique = aux[CatCols].nunique()
obj_nunique=obj_nunique.sort_values(ascending=False)
# BadCatCols=[colN for colN in obj_nunique.index if obj_nunique[colN]>10 or colN in DropCols]
# GoodCatCols=list(set(obj_nunique.index)-set(BadCatCols))

OkCat=[];DropCat=[]; aux1=list(miss)
RejectCond=0
for col in obj_nunique.index:
    diff=obj_nunique[col]
    # # if(diff>10): DropCat+=[col]
    # if col not in miss.index: continue
    # # print(col,'\t\t\t',obj_nunique[col],'\t',miss[col])
    # nan=miss[col]; diff=obj_nunique[col]
    dfCounts=aux[col].value_counts(ascending=False)
    if((col in miss.index and miss[col]>0.2*nEvent) 
       or diff>5
       # or col=='GarageFinish' or 'Qual' in col # quasi uniform distribution w price
       or (np.std(dfCounts)/np.mean(dfCounts)-1)< 0 # for some reason, this excludes quasi uniform distributions, refinate on this!
       ): 
        DropCat+=[col]
    else: OkCat+=[col]; 
    


OkCat=OkCat[-3:]
aux[OkCat].describe # Gets a good panorama of the cat columns
# for f in range(len(CatCols)):
plt.close('all')
for col in OkCat:
    ColPrice=data_full.groupby([col]).sum()['SalePrice']
    # ColPrice=data_full.groupby([col])['SalePrice'].mean()
    plt.close(col)
    plt.figure(col)
    ColPrice.plot(kind="bar",title=col)
    dfCounts=aux[col].value_counts(ascending=False)
    if(len(dfCounts)<3): continue
    # print(col,  np.std(dfCounts)/np.mean(dfCounts)-1) 
    # if(min(abs( np.diff(dfCounts) )) > .1*nEvent/len(dfCounts)): 
    # if((np.std(dfCounts)/np.mean(dfCounts)-1)< .02): 
    #     print(dfCounts)


# # Remove categorical columns
# aux = aux.drop(obj_nunique.index, axis=1)
# auxVal = auxVal.drop(obj_nunique.index, axis=1)

# # Based on this analysis, select the columns to keep.
KeepCols=OkNum+OkCat

aux=aux[KeepCols]; auxVal=X_valid[KeepCols].copy(); auxTest=X_test[KeepCols]

# X_train=aux; X_valid=auxVal

#%% Data treatment using pipelines

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Preprocessing for numerical data
numIpt = SimpleImputer(strategy='mean')

# Preprocessing for categorical data
catIpt = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numIpt, KeepNumCols),
        ('cat', catIpt, OkCat)
    ])

print('Pre-treatment ended.')

#%% Will use a XGBRegressor model, and test it if need it

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

CheckBestParams=0 
# Choosing best parameter takes some time, it's better to run it one time only
if(CheckBestParams):
    gridsize=5
    nstim=np.arange(50,1000+1,200)
    nstim=np.linspace(100,400,gridsize)
    # nstim=np.arange(250,350+1,50)
    nlearn=np.arange(10,50+1,20);
    nlearn=np.linspace(10,50,gridsize);
    nlearn=(nlearn)/1000
    
    # pipe = Pipeline([('preprocesado',StandardScaler()),('clasificador',SVC())])
    # Queremos buscar un clasificador entre RandomForestClassifier y SVC que de el mejor resultado en este conjunto de datos. 
    # Combinando GridSearchCV y Pipeline
    # Para asignar un estimador a un paso, usamos el nombre del paso despues del nombre del parametro.
    # Cuando queremos omitir un paso en el Pipeline, establecemos ese paso con None.
    param_grid = [{'model':[XGBRegressor()],'preprocessor':[preprocessor],
                   'model__n_estimators':[int(n) for n in nstim],
                   'model__learning_rate':list(nlearn)}]
                  # ,{'clasificador':[RandomForestClassifier(n_estimators=100)],
                  #  'preprocesado':[None],
                  #  'clasificador__max_features':[1,2,3]}]
    
    # Busqueda en cuadricula con validacion cruzada para ajustar parametros.
    grid = GridSearchCV(pipeXGB, param_grid,cv=5)
    grid.fit(aux,y_train)
    
    # Convertir a DataFrame
    results = pd.DataFrame(grid.cv_results_)
    results.head()
    scores = np.array(results.mean_test_score).reshape(gridsize, gridsize)
    scores = pd.DataFrame(scores)
    plt.close()
    ax = sns.heatmap(scores, annot = True,cmap="jet",cbar=0
                     ,xticklabels=param_grid[0]['model__n_estimators'],yticklabels=param_grid[0]['model__learning_rate'])
    ax.set(title='Cross-validation for XGB'
           , ylabel='learning',xlabel='n estim')
    
    # print("n_estimators | learning_rate")
    aux2=results[['param_model__learning_rate', 'param_model__n_estimators','rank_test_score','mean_test_score']].sort_values('rank_test_score')
    nstim=aux2.param_model__n_estimators[0] #100
    nlearn=aux2.param_model__learning_rate[0] #0.01
    
    print('Optimal configuration is nestimators=',nstim,' and learning rate = ',nlearn)
else: 
    nstim=100
    nlearn=0.01
    # Taken from running the above procedure once
    print('Chosen configuration is nestimators=',nstim,' and learning rate = ',nlearn)

#%% Take the best model and estimate a
DefModel=XGBRegressor(n_estimators=nstim,
                        learning_rate=nlearn,
                        # early_stopping_rounds = 10,
                        eval_metric="mae", random_state=0)
pipeXGB = Pipeline(steps=[
    ('preprocessor', preprocessor),('model', DefModel)
    ])
pipeXGB.fit(aux,y_train)

predTest=pipeXGB.predict(auxTest)
solution=pd.DataFrame({'Id':auxTest.index,'SalePrice':predTest}).set_index('Id')

solution.to_csv('output.csv')


