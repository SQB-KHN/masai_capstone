"""Part D: hub-level operational segmentation and separation diagnostics."""
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

frame=pd.read_csv('data/modelling_table.csv');hubs=pd.read_csv('data/hubs.csv')
profile=(frame.groupby('hub_id',sort=True).agg(shipments=('shipment_id','count'),mean_delivery_hours=('delivery_hours','mean'),delay_rate=('is_delayed','mean'),mean_distance_km=('distance_km','mean'),mean_n_stops=('n_stops','mean')).reset_index().sort_values('hub_id'))
profile.round(4).to_csv('outputs/hub_profile.csv',index=False)
cols=['mean_delivery_hours','delay_rate','mean_distance_km','mean_n_stops']
scaled=StandardScaler().fit_transform(profile[cols])
separation=[];final_labels=None
for k in [2,3,4,5]:
    labels=KMeans(n_clusters=k,random_state=42,n_init=10).fit_predict(scaled)
    if k==3:final_labels=labels
    within=[];across=[]
    for i,j in combinations(range(len(scaled)),2):
        d=np.linalg.norm(scaled[i]-scaled[j]);(within if labels[i]==labels[j] else across).append(d)
    sizes=sorted(np.bincount(labels),reverse=True)
    separation.append({'K':k,'across_over_within':np.mean(across)/np.mean(within),'cluster_sizes':'-'.join(map(str,sizes))})
pd.DataFrame(separation).round(4).to_csv('outputs/part_d_separation.csv',index=False)
profile['cluster']=final_labels
ordering=profile.groupby('cluster').delay_rate.mean().sort_values().index
names={idx:name for idx,name in zip(ordering,['Standard','Watch','Escalate'])}
profile['segment']=profile.cluster.map(names)
segments=profile[['hub_id','segment']].merge(hubs[['hub_id','hub_city','hub_tier']],on='hub_id')
segments.to_csv('outputs/hub_segments.csv',index=False)
summary=profile.groupby('segment').agg(hubs=('hub_id','count'),mean_delivery_hours=('mean_delivery_hours','mean'),delay_rate=('delay_rate','mean'),mean_distance_km=('mean_distance_km','mean'),mean_n_stops=('mean_n_stops','mean')).reindex(['Standard','Watch','Escalate']).reset_index()
summary.round(4).to_csv('outputs/segment_profile.csv',index=False)
expected=segments.hub_tier.map({1:'Standard',2:'Watch',3:'Escalate'})
segments.loc[segments.segment.ne(expected)].to_csv('outputs/tier_vs_segment_mismatch.csv',index=False)
print('Part D segment sizes:',summary[['segment','hubs']].values.tolist(),'; mismatches:',segments.loc[segments.segment.ne(expected),'hub_id'].tolist())
