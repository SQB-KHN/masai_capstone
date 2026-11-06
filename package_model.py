"""Rebuild and persist the selected classifier using the training split only."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.ensemble import RandomForestClassifier
from src.shipment_prep import FEATURES, IMPUTE_COLUMNS,clean_shipments

Path('models').mkdir(exist_ok=True)
config=pd.read_csv('outputs/part_c_final_config.csv').iloc[0]
frame=pd.read_csv('data/modelling_table.csv');X=frame[FEATURES];y=frame.is_delayed
X_train,X_tmp,y_train,y_tmp=train_test_split(X,y,test_size=.3,random_state=42,stratify=y)
X_val,X_test,y_val,y_test=train_test_split(X_tmp,y_tmp,test_size=.5,random_state=42,stratify=y_tmp)
depth=None if str(config.max_depth)=='unlimited' else int(config.max_depth)
model=Pipeline([('selector',SelectKBest(f_classif,k=int(config.selector_k))),
                ('model',RandomForestClassifier(random_state=42,n_estimators=int(config.n_estimators),max_depth=depth,min_samples_leaf=int(config.min_samples_leaf)))]).fit(X_train,y_train)
joblib.dump(model,'models/delay_model.pkl')
pd.DataFrame({'threshold':[float(config.threshold)]}).to_csv('models/delay_threshold.csv',index=False)
pd.DataFrame({'feature':FEATURES}).to_csv('models/feature_order.csv',index=False)
_,medians=clean_shipments(pd.read_csv('data/raw_shipments.csv'))
pd.DataFrame([{'column':k,'median':medians[k]} for k in IMPUTE_COLUMNS]).to_csv('models/impute_values.csv',index=False)
restored=joblib.load('models/delay_model.pkl')
diff=np.max(np.abs(model.predict_proba(X_test)[:,1]-restored.predict_proba(X_test)[:,1]))
pd.DataFrame([{'metric':'test_rows_scored','value':len(X_test)},
              {'metric':'reload_max_abs_probability_diff','value':round(float(diff),4)},
              {'metric':'threshold','value':float(config.threshold)}]).to_csv('outputs/part_d_reload_check.csv',index=False)
assert diff <= 1e-9
print('Model packaged, reload max difference:',diff)
