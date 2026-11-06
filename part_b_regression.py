"""Part B: regression diagnostics, model selection and untouched test evaluation."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.shipment_prep import FEATURES

frame=pd.read_csv('data/modelling_table.csv')
X=frame[FEATURES];y=frame.delivery_hours
X_train,X_tmp,y_train,y_tmp=train_test_split(X,y,test_size=.3,random_state=42)
X_val,X_test,y_val,y_test=train_test_split(X_tmp,y_tmp,test_size=.5,random_state=42)
scaler=StandardScaler().fit(X_train)
train_scaled=scaler.transform(X_train);test_scaled=scaler.transform(X_test)
checks={'train_rows':len(X_train),'val_rows':len(X_val),'test_rows':len(X_test),
        'train_scaled_max_abs_mean':np.abs(train_scaled.mean(axis=0)).max(),
        'test_scaled_max_abs_mean':np.abs(test_scaled.mean(axis=0)).max()}
pd.DataFrame([{'metric':k,'value':round(float(v),4)} for k,v in checks.items()]).to_csv('outputs/part_b_checks.csv',index=False)
def scores(model,split,actual,pred):
    mse=mean_squared_error(actual,pred)
    return {'model':model,'split':split,'MAE':mean_absolute_error(actual,pred),'MSE':mse,'RMSE':np.sqrt(mse),'R2':r2_score(actual,pred)}
rows=[scores('mean_baseline','val',y_val,np.full(len(y_val),y_train.mean()))]
depths=[1,2,3,4,6,8,12,None];tree_rows=[]
for d in depths:
    model=DecisionTreeRegressor(max_depth=d,random_state=42).fit(X_train,y_train)
    tree_rows.append({'max_depth':d if d is not None else 'unlimited',
                      'train_rmse':np.sqrt(mean_squared_error(y_train,model.predict(X_train))),
                      'val_rmse':np.sqrt(mean_squared_error(y_val,model.predict(X_val)))})
pd.DataFrame(tree_rows).round(4).to_csv('outputs/part_b_tree_depth.csv',index=False)
best_depth=min(range(len(depths)),key=lambda i:(tree_rows[i]['val_rmse'],float('inf') if depths[i] is None else depths[i]))
models={
 'linear_regression':Pipeline([('scaler',StandardScaler()),('model',LinearRegression())]),
 'decision_tree':DecisionTreeRegressor(max_depth=depths[best_depth],random_state=42),
 'random_forest':RandomForestRegressor(n_estimators=100,random_state=42),
 'gradient_boosting':GradientBoostingRegressor(n_estimators=100,learning_rate=.1,random_state=42),
}
for name,m in models.items():
    m.fit(X_train,y_train)
    rows.append(scores(name,'val',y_val,m.predict(X_val)))
learning=[]
for lr in [.01,.1,1.0]:
    m=GradientBoostingRegressor(n_estimators=100,learning_rate=lr,random_state=42).fit(X_train,y_train)
    pred=m.predict(X_val)
    learning.append({'learning_rate':lr,'val_rmse':np.sqrt(mean_squared_error(y_val,pred)),'val_r2':r2_score(y_val,pred)})
pd.DataFrame(learning).round(4).to_csv('outputs/part_b_gb_learning_rate.csv',index=False)
ranked=sorted(rows[1:],key=lambda r:r['RMSE'])
X_cv=pd.concat([X_train,X_val]);y_cv=pd.concat([y_train,y_val]);kf=KFold(n_splits=5,shuffle=True,random_state=42)
cv=[]
for item in ranked[:2]:
    name=item['model'];rs=cross_val_score(models[name],X_cv,y_cv,cv=kf)
    cv.append({'model':name,'mean_r2':rs.mean(),'std_r2':rs.std()})
pd.DataFrame(cv).round(4).to_csv('outputs/part_b_cv.csv',index=False)
winner=ranked[0]['model'];rows.append(scores(winner,'test',y_test,models[winner].predict(X_test)))
pd.DataFrame(rows).round(4).to_csv('outputs/part_b_metrics.csv',index=False)
print('Part B winner:',winner,'test R2',round(rows[-1]['R2'],4))
