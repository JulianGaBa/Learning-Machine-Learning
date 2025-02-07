import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error,accuracy_score,f1_score
from xgboost import XGBClassifier


def EsteticaPlot(titulo,xlabel,ylabel,fig=None,xaxis=None,fsize=None,legend=None):
    fsize=fsize or 16
    fig=fig or titulo
    # plt.grid(True)

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
folder=r"C:\Users\CNEA\Desktop\Gasanego B Julian\Cursos\ML\Kaggle-Titanic"
os.chdir(folder)

data_full = pd.read_csv(folder+'/train.csv', index_col='PassengerId')
y = data_full.Survived              
X_full=data_full.copy(); X_full.drop(['Survived'], axis=1, inplace=True)

X_test = pd.read_csv(folder+'/test.csv', index_col='PassengerId')
# y_test = test.Survived              
# X_test=test.copy(); X_test.drop(['Survived'], axis=1, inplace=True)


# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X_full, y, 
                                                      train_size=0.8, test_size=0.2, random_state=0)
X_train.head();

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

# Remove num columns w too many nan values
DropCols=[col for col in miss.index if (col in aux.columns and miss[col]>0.4*nEvent)]
KeepCols=list(set(aux.columns)-set(DropCols))

PlotCols=[col for col in KeepCols if col in NumCols or aux[col].nunique()<.2*nEvent]
aux[PlotCols].describe # Gets a good panorama of the cat columns
plt.close('all')
for col in PlotCols:
    ColPrice=data_full.groupby([col]).sum()['Survived']
    # if col=='Age':
    if col in NumCols and aux[col].nunique()>20:
        data_cut=data_full.copy(); 
        N=10
        aaa=list(aux.Age.dropna().unique());aaa.sort()
        label=[int(np.mean(np.array(rg))) for rg in np.array_split(aaa,N)]
        data_cut[col]=pd.cut(data_full[col], bins=N, labels=label, include_lowest=True)
        ColPrice=data_cut.groupby([col]).sum()['Survived']
    
    # ColPrice=data_full.groupby([col])['Survived'].mean()
    plt.close(col)
    plt.figure(col)
    ColPrice.plot(kind="bar",title=col)
    dfCounts=aux[col].value_counts(ascending=False)
    if(len(dfCounts)<3): continue

# All plotted columns seem relevant
KeepCols=PlotCols

KeepNumCols=[col for col in KeepCols if col in NumCols]
KeepCatCols=[col for col in KeepCols if col in CatCols]

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
        ('cat', catIpt, KeepCatCols)
    ])

print('Pre-treatment ended.')

#%% Declare and test models using pipelines

nstim=np.arange(50,1000+1,100)
nstim=np.arange(250,350+1,50)

MAEs=[]; 
MAEf=[]
# for i in range(len(nstim)):

#  Without cross validation

# Declare pipeline models
pipeForest = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(n_estimators=150, random_state=0))
    ])
pipeForest.fit(aux,y_train)
predF = pipeForest.predict(auxVal)
MAEf=f1_score(y_valid, predF)



pipeXGB = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', XGBClassifier(n_estimators=150,
                            # learning_rate=0.1,
                            # early_stopping_rounds = 10,
                            eval_metric="error", random_state=0
                            )) 
    ])
pipeXGB.fit(aux,y_train
            # , eval_set=[(aux, y_train)], verbose=0
            )
predX = pipeXGB.predict(auxVal)
MAEx=f1_score(y_valid, predX)

print('Forest: MAE=',MAEf); print('XGB: MAE=',MAEx); 


#%%

from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import cross_val_score

CheckBestParams=0
if(CheckBestParams):
    gridsize=5
    # nstim=np.linspace(100,400,gridsize)
    nstim=np.linspace(140,170,gridsize-1)+np.array(150)
    nstim=[140,145,150,155,160]
    # nlearn=np.linspace(10,50,gridsize);
    nlearn=np.linspace(17,22,gridsize);
    nlearn=(nlearn)/1000
    
    # pipe = Pipeline([('preprocesado',StandardScaler()),('clasificador',SVC())])
    # Queremos buscar un clasificador entre RandomForestClassifier y SVC que de el mejor resultado en este conjunto de datos. 
    # Combinando GridSearchCV y Pipeline
    # Para asignar un estimador a un paso, usamos el nombre del paso despues del nombre del parametro.
    # Cuando queremos omitir un paso en el Pipeline, establecemos ese paso con None.
    param_grid = [{'model':[XGBClassifier(eval_metric="error", random_state=0)],'preprocessor':[preprocessor],
                   'model__n_estimators':[int(n) for n in nstim],
                   'model__learning_rate':list(nlearn)}]
                  # ,{'clasificador':[RandomForestClassifier(n_estimators=100)],
                  #  'preprocesado':[None],
                  #  'clasificador__max_features':[1,2,3]}]
    
    # Busqueda en cuadricula con validacion cruzada para ajustar parametros.
    grid = GridSearchCV(pipeXGB, param_grid,cv=5)
    grid.fit(aux,y_train)
#%   
# for i in range(1):    
    # Convertir a DataFrame
    results = pd.DataFrame(grid.cv_results_)
    results.head()
    scores = np.array(results.mean_test_score).reshape(gridsize, gridsize)
    scores = pd.DataFrame(scores)

    plt.close('all')
    ax = sns.heatmap(scores, annot = True, fmt='.4g',cmap="jet",cbar=0
                     ,xticklabels=param_grid[0]['model__n_estimators'],yticklabels=param_grid[0]['model__learning_rate'])
    ax.set(title='Cross-validation for XGB'
           , ylabel='learning',xlabel='n estim')
    
    # print("n_estimators | learning_rate")
    aux2=results[['param_model__learning_rate', 'param_model__n_estimators','rank_test_score','mean_test_score']].sort_values('rank_test_score')
    nstim=aux2.param_model__n_estimators.iloc[0] #100
    nlearn=aux2.param_model__learning_rate.iloc[0] #0.01
    
    print('Optimal configuration is nestimators=',nstim,' and learning rate = ',nlearn,
          ' with score ', aux2.mean_test_score.iloc[0])
else: 
    nstim=150
    nlearn=0.02
    print('Chosen configuration is nestimators=',nstim,' and learning rate = ',nlearn)

#%% Define model
DefModel=XGBClassifier(n_estimators=nstim,
                        learning_rate=nlearn,
                        # early_stopping_rounds = 10,
                        eval_metric="error", random_state=0)
pipeXGB = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', XGBClassifier(n_estimators=nstim,
                            learning_rate=nlearn,
                            # early_stopping_rounds = 10,
                            eval_metric="error", random_state=0
                            )) 
    ])

miss1 = (aux.isnull().sum()); miss1=miss1[miss1 > 0]

pipeXGB.fit(aux,y_train)
predVal = pipeXGB.predict(auxVal)
MAEx=mean_absolute_error(y_valid, predX)

predTest=pipeXGB.predict(auxTest)
solution=pd.DataFrame({'PassengerId':auxTest.index,'Survived':predTest}).set_index('PassengerId')

solution.to_csv('Survived-Prediction.csv')
