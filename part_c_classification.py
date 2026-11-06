"""Part C: leakage-safe classification, feature signal, tuning, and operational threshold."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from src.shipment_prep import FEATURES, NOISE_COLUMNS

frame=pd.read_csv('data/modelling_table.csv'); X=frame[FEATURES];y=frame.is_delayed
X_train,X_tmp,y_train,y_tmp=train_test_split(X,y,test_size=.3,random_state=42,stratify=y)
X_val,X_test,y_val,y_test=train_test_split(X_tmp,y_tmp,test_size=.5,random_state=42,stratify=y_tmp)
assert (len(X_train),len(X_val),len(X_test)) == (840,180,180)
assert (y_train.sum(),y_val.sum(),y_test.sum()) == (195,42,41)

def score(name,labels,pred):
    tn,fp,fn,tp=confusion_matrix(labels,pred,labels=[0,1]).ravel()
    return {'model':name,'TN':tn,'FP':fp,'FN':fn,'TP':tp,'accuracy':accuracy_score(labels,pred),
            'precision':precision_score(labels,pred,zero_division=0),
            'recall':recall_score(labels,pred,zero_division=0),'f1':f1_score(labels,pred,zero_division=0)}

baseline=score('blind_all_on_time',y_val,np.zeros(len(y_val),dtype=int))
logistic=Pipeline([('scaler',StandardScaler()),('model',LogisticRegression(max_iter=1000))]).fit(X_train,y_train)
logistic_row=score('logistic_regression',y_val,logistic.predict(X_val))
forest=RandomForestClassifier(n_estimators=100,random_state=42).fit(X_train,y_train)
forest_row=score('random_forest_default',y_val,forest.predict(X_val))
tree_candidates=[]
for depth in [2,3,4,5,6,8,10]:
    tree=DecisionTreeClassifier(criterion='gini',max_depth=depth,random_state=42).fit(X_train,y_train)
    tree_candidates.append((depth,score(f'decision_tree_depth_{depth}',y_val,tree.predict(X_val))))
best_tree=min(tree_candidates,key=lambda pair:(-pair[1]['f1'],pair[0]))[1]
pd.DataFrame([baseline,logistic_row,best_tree,forest_row]).round(4).to_csv('outputs/part_c_models.csv',index=False)
selector_all=SelectKBest(f_classif,k='all').fit(X_train,y_train)
f_scores=selector_all.scores_;rank_order=np.argsort(-f_scores,kind='stable');ranks=np.empty(len(FEATURES),dtype=int)
ranks[rank_order]=np.arange(1,len(FEATURES)+1)
pd.DataFrame({'feature':FEATURES,'f_score':f_scores,'f_rank':ranks,'rf_importance':forest.feature_importances_,
              'is_noise':[name in NOISE_COLUMNS for name in FEATURES]}).round(4).to_csv('outputs/part_c_feature_scores.csv',index=False)
select_rows=[]
for k in [5,7,10,12,17]:
    m=Pipeline([('selector',SelectKBest(f_classif,k=k)),('model',RandomForestClassifier(n_estimators=100,random_state=42))]).fit(X_train,y_train)
    select_rows.append({'k':k,'val_f1':f1_score(y_val,m.predict(X_val))})
pd.DataFrame(select_rows).round(4).to_csv('outputs/part_c_selectk.csv',index=False)
select7=SelectKBest(f_classif,k=7).fit(X_train,y_train)
train7=select7.transform(X_train);val7=select7.transform(X_val)
grid={'n_estimators':[50,100,150],'max_depth':[4,6,8],'min_samples_leaf':[1,5,10]}
base=RandomForestClassifier(random_state=42)
gs=GridSearchCV(base,grid,cv=5).fit(train7,y_train)
rs=RandomizedSearchCV(base,grid,n_iter=9,cv=5,random_state=42).fit(train7,y_train)
default=RandomForestClassifier(n_estimators=100,random_state=42).fit(train7,y_train)
configs=[('grid_search',27,5,135,str(gs.best_params_),gs.best_score_,gs.best_estimator_,gs.best_params_),
         ('randomized_search',9,5,45,str(rs.best_params_),rs.best_score_,rs.best_estimator_,rs.best_params_),
         ('default_forest_same_7_features',1,0,1,'defaults',np.nan,default,{'n_estimators':100,'max_depth':None,'min_samples_leaf':1})]
tuning=[]
for name,n,folds,fits,params,accuracy,m,config in configs:
    tuning.append({'search':name,'candidates':n,'cv_folds':folds,'total_fits':fits,
                   'best_params':params,'best_cv_accuracy':accuracy,'val_f1':f1_score(y_val,m.predict(val7))})
pd.DataFrame(tuning).round(4).to_csv('outputs/part_c_tuning.csv',index=False)
priority={'default_forest_same_7_features':0,'grid_search':1,'randomized_search':2}
chosen_idx=min(range(3),key=lambda i:(-tuning[i]['val_f1'],priority[tuning[i]['search']]))
chosen=configs[chosen_idx];chosen_name=chosen[0];chosen_params=chosen[-1]
final=Pipeline([('selector',SelectKBest(f_classif,k=7)),('model',RandomForestClassifier(random_state=42,**chosen_params))]).fit(X_train,y_train)
val_prob=final.predict_proba(X_val)[:,1]
thresholds=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
table=[]
for t in thresholds:
    row=score('final',y_val,(val_prob>=t).astype(int));row.pop('model');table.append({'threshold':t,**row})
pd.DataFrame(table).round(4).to_csv('outputs/part_c_thresholds.csv',index=False)
chosen_t=max((item['threshold'] for item in table if item['recall']>=.8),default=.1)
test_prob=final.predict_proba(X_test)[:,1]
test_row=score('final',y_test,(test_prob>=chosen_t).astype(int));test_row.pop('model')
pd.DataFrame([{'threshold':chosen_t,**test_row}]).round(4).to_csv('outputs/part_c_final_test.csv',index=False)
pd.DataFrame([{'selector_k':7,'n_estimators':chosen_params['n_estimators'],
               'max_depth':chosen_params['max_depth'] if chosen_params['max_depth'] is not None else 'unlimited',
               'min_samples_leaf':chosen_params['min_samples_leaf'],'threshold':chosen_t,'chosen_from':chosen_name}]).to_csv('outputs/part_c_final_config.csv',index=False)
print('Part C:',chosen_name,'threshold',chosen_t,'test F1',round(test_row['f1'],4))
