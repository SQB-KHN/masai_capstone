"""Part A: reproducible clean table, checks, descriptive evidence and correlations."""
from pathlib import Path
import numpy as np
import pandas as pd
from src.shipment_prep import FEATURES, clean_shipments, build_features

Path('outputs').mkdir(exist_ok=True)
raw = pd.read_csv('data/raw_shipments.csv')
hubs = pd.read_csv('data/hubs.csv')
dedup = raw.drop_duplicates()
clean, medians = clean_shipments(raw)
frame = build_features(clean, hubs)
frame['is_delayed'] = frame['delivery_hours'].gt(frame['promised_hours']).astype(int)
assert len(frame) == len(dedup) == 1200
assert frame[FEATURES + ['delivery_hours', 'is_delayed']].notna().all().all()
frame.to_csv('data/modelling_table.csv', index=False)
q1, q3 = frame.delivery_hours.quantile([.25, .75]); iqr = q3-q1
lower, upper = q1-1.5*iqr, q3+1.5*iqr
checks = {
 'raw_rows': len(raw), 'duplicate_rows_removed': len(raw)-len(dedup), 'rows_after_dedup':len(dedup),
 'missing_distance_km_before':dedup.distance_km.isna().sum(),
 'missing_parcel_weight_before':dedup.parcel_weight_kg.isna().sum(),
 'sentinel_weight_rows':dedup.parcel_weight_kg.gt(100).sum(),
 'missing_declared_value_before':dedup.declared_value.isna().sum(),
 'missing_payment_mode_before':dedup.payment_mode.isna().sum(),
 'service_type_spellings_before':dedup.service_type.nunique(),
 'service_type_spellings_after':clean.service_type.nunique(),
 'median_distance_km':medians['distance_km'],
 'median_parcel_weight_kg':medians['parcel_weight_kg'],
 'median_declared_value_inr':medians['declared_value_inr'],
 'rows_after_hub_merge':len(frame),
 'nulls_in_features_and_target':frame[FEATURES + ['delivery_hours','is_delayed']].isna().sum().sum(),
 'delayed_count':frame.is_delayed.sum(), 'delay_rate':frame.is_delayed.mean(),
 'hours_q1':q1, 'hours_q3':q3, 'hours_lower_bound':lower, 'hours_upper_bound':upper,
 'hours_outlier_count':((frame.delivery_hours < lower)|(frame.delivery_hours > upper)).sum(),
 'declared_value_mean_inr':frame.declared_value_inr.mean(),
 'declared_value_median_inr':frame.declared_value_inr.median(),
 'log_declared_value_mean':frame.log_declared_value.mean(),
 'log_declared_value_median':frame.log_declared_value.median(),
}
pd.DataFrame([{'metric':k,'value':round(float(v),4)} for k,v in checks.items()]).to_csv('outputs/part_a_checks.csv',index=False)
frame['distance_band'] = pd.cut(frame.distance_km,[0,200,400,600,2000],labels=['0-200','201-400','401-600','601+'])
parts=[]
for dim in ['service_type','hub_tier','distance_band']:
    group=frame.groupby(dim, observed=True).agg(shipments=('shipment_id','count'),delayed=('is_delayed','sum'),mean_hours=('delivery_hours','mean')).reset_index()
    group['delay_rate'] = group.delayed/group.shipments
    group.insert(0,'dimension',dim)
    group=group.rename(columns={dim:'group'})
    parts.append(group[['dimension','group','shipments','delayed','delay_rate','mean_hours']])
pd.concat(parts,ignore_index=True).round(4).to_csv('outputs/part_a_delay_by_group.csv',index=False)
frame.select_dtypes(include='number').corr().round(4).to_csv('outputs/part_a_correlations.csv', index=True)
print('Part A:',len(frame),'records;',int(frame.is_delayed.sum()),'delayed')
