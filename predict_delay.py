"""Produce deterministic action recommendations for parcels not yet delivered."""
import joblib
import pandas as pd
from src.shipment_prep import clean_shipments,build_features

raw=pd.read_csv('data/new_shipments.csv');hubs=pd.read_csv('data/hubs.csv')
medians=dict(pd.read_csv('models/impute_values.csv').values)
order=pd.read_csv('models/feature_order.csv').feature.tolist()
threshold=float(pd.read_csv('models/delay_threshold.csv').threshold.iloc[0])
model=joblib.load('models/delay_model.pkl')
clean,_=clean_shipments(raw,medians)
featured=build_features(clean,hubs)
assert featured[order].notna().all().all()
probabilities=model.predict_proba(featured[order])[:,1].round(4)
result=featured[['shipment_id','hub_id']].copy()
result=result.merge(pd.read_csv('outputs/hub_segments.csv')[['hub_id','segment']],how='left',on='hub_id',sort=False,validate='many_to_one')
result['delay_probability']=probabilities
result['flagged']=(result.delay_probability>=threshold).astype(int)
result['action']='No action'
result.loc[result.flagged.eq(1),'action']='Proactive customer call'
result.loc[result.flagged.eq(1)&result.segment.eq('Escalate'),'action']='Escalate to hub manager'
assert result.notna().all().all() and len(result)==48 and result.shipment_id.is_unique
result[['shipment_id','hub_id','segment','delay_probability','flagged','action']].to_csv('outputs/new_shipment_actions.csv',index=False)
print('Actions:',int(result.flagged.sum()),'flagged;',int(result.action.eq('Escalate to hub manager').sum()),'escalations')
