# Delhivery Delay Intelligence — Masai Capstone

An offline, reproducible machine-learning pipeline that estimates delivery hours, predicts missed promises, segments 24 logistics hubs and produces intervention actions for newly dispatched parcels. **Every shipment in this project is synthetic**; Delhivery is the scenario, not the data source.

## Decision brief

**Operational question.** The synthetic network has three unanswered dispatch-time decisions: *When will the parcel arrive? Will it miss its promise? Which hub playbook should apply?* The repository answers them as a connected, auditable decision system rather than four unrelated modelling exercises.

| Decision | Analytical method | Evidence produced | Operational use |
| --- | --- | --- | --- |
| Estimate delivery duration | Train/validate/test regression; compare a mean baseline, linear regression and three tree-based approaches | Selected model: **linear regression**, validation RMSE **4.8912 hours**; test RMSE **4.5235 hours** | Dispatch-time expected delivery duration |
| Detect a possible missed promise | Stratified classification; training-only feature selection; compare tuning strategies; choose recall-constrained threshold | **0.5** alert threshold; test precision **0.6889**, recall **0.7561** | Prioritised calls before a shipment is late |
| Select hub operating playbook | Aggregate each hub's historical outcomes; standardise four profile variables; compare K=2–5 | **7 Standard / 8 Watch / 9 Escalate** hubs; **4** tier/behaviour mismatches | Escalate risk based on observed behaviour, not tier alone |
| Turn predictions into work | Reload saved classifier, threshold, medians, feature order and segments; score 48 unseen parcels | **8 flagged**, of which **6** require manager escalation | A deterministic, parcel-level action register |

**Design principle:** the same preparation functions serve historical data and new parcels; the target is never used to engineer dispatch-time features. Each decision is traceable to an output CSV and the final action is determined by an explicit rule, not an unexplained recommendation.

## How to run

Clone the public `main` branch and run these commands **in the specified order**:

```bash
git clone https://github.com/SQB-KHN/masai_capstone.git
cd masai_capstone
python -m pip install -r requirements.txt
python generate_data.py
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
```

The unmodified provided generator creates `data/hubs.csv` (24 rows), `data/raw_shipments.csv` (1,224 rows) and `data/new_shipments.csv` (48 rows). Part A creates `data/modelling_table.csv` (1,200 unique historical shipments). Parts B and C build regression and classification evidence; Part D assigns hub segments. `predict_delay.py` produces `outputs/new_shipment_actions.csv`.

The functions `clean_shipments()` and `build_features()` in `src/shipment_prep.py` are reused for training and inference; 17 pinned feature columns are passed in the original prescribed order. All scripts run offline with the four open-source packages pinned in `requirements.txt`.

**Pickle compatibility:** Running `python package_model.py` regenerates `models/delay_model.pkl`, `models/delay_threshold.csv`, `models/feature_order.csv`, and `models/impute_values.csv`. The committed pickle is reliably loadable only with the exact pinned scikit-learn version. If versions differ, the grader should run `package_model.py` rather than loading the earlier pickle. The reload check is in `outputs/part_d_reload_check.csv`; successive scoring runs should produce byte-identical results.

### End-to-end build sequence and evidence lineage

```text
Provided generator
  ├─ data/hubs.csv (24 hub records)
  ├─ data/raw_shipments.csv (1,224 export rows)
  └─ data/new_shipments.csv (48 parcels without outcomes)
             │
             ▼
Part A: deduplicate → standardise → impute → join hubs → engineer features
  └─ data/modelling_table.csv (1,200 rows, 17 ordered model features)
             ├──────────────────────────┐
             ▼                          ▼
Part B: delivery-hours regression    Part C: delay classification
  └─ regression experiment CSVs         ├─ feature, tuning and threshold CSVs
                                        └─ final classifier configuration
             │                          │
             └───────────┬──────────────┘
                         ▼
Part D: 24 hub profiles → K-Means → Standard / Watch / Escalate
                         │
                         ▼
Package training-only classifier + medians + feature order + threshold
                         │
                         ▼
Score 48 new parcels → join hub segments → assign action
  └─ outputs/new_shipment_actions.csv
```

**Output ownership:** `part_a_clean_eda.py` creates the modelling table and three Part A CSVs; `part_b_regression.py` creates five regression CSVs; `part_c_classification.py` creates seven classifier CSVs; `part_d_segments.py` creates five hub CSVs; `package_model.py` writes four model artefacts and the reload-check CSV; `predict_delay.py` writes the final action CSV. Each subsequent script reads data or configurations created earlier, rather than copying benchmark values from the assignment.

**Run-time integrity checks:** the generator must print `24 1224 48`; Part A must retain 1,200 unique shipments; the 17 features plus two targets must have zero nulls; the split sizes must be 840/180/180; and the final action CSV must contain 48 unique shipments without missing values. The outputs are deterministic under the pinned dependencies and the specified seeds.

## Data findings

The evidence is in `outputs/part_a_checks.csv`, `part_a_delay_by_group.csv` and `part_a_correlations.csv`.

- **Cleaning:** 24 exact duplicates removed from 1,224 raw rows; 1,200 retained. Five scanner-fault weight readings were marked missing. The historical imputation medians were 314.95 km, 1.83 kg and **INR 1,121.50**. No missing model inputs or targets remain.
- **Delayed overall:** **278/1,200 = 0.2317** (23.17%). **Express** is most frequently delayed: **119/243 = 0.4897**, versus **Standard 132/642 = 0.2056** and **Economy 27/315 = 0.0857**. Promised delivery windows are 36/60/96 hours respectively.
- **Tier gradient:** Tier 1 **16/400 = 0.0400**, Tier 2 **86/400 = 0.2150**, and Tier 3 **176/400 = 0.4400**.
- **Distance gradient:** 0–200 km **0.0219**; 201–400 km **0.1368**; 401–600 km **0.4364**; 601+ km **0.8523**.
- **Top signed correlations with delivery hours**, ordered by absolute Pearson correlation and excluding the target: `promised_hours` **+0.8372**, `distance_km` **+0.4013**, and `n_stops` **+0.3247**. Association is not causation.
- **Log value:** Median-filled declared value is right-skewed (mean **INR 1,533.0542**, median **INR 1,121.50**). The log values (mean **7.0277**, median **7.0224**) compress extreme rupee values, reducing the effect of scale and skew.

### Steps taken to turn an export into trustworthy evidence

1. **Diagnosed the raw extract:** checked duplicates, spelling inconsistencies, currency-formatted amounts, incomplete values and invalid 999 kg scanner readings. Counted missingness *before* filling so the cleaning step remains auditable.
2. **Standardised the records in the prescribed order:** removed duplicates, normalised service labels, retained unknown payment mode explicitly, parsed INR amounts and timestamps, then replaced scanner sentinels with missing values.
3. **Calculated the three historical medians:** filled distance, parcel weight and declared value using fixed historical values, which are later saved for scoring unseen parcels. No unobserved delivery outcome is needed for this transformation.
4. **Reused one feature contract:** left-joined the 24-hub reference table and computed promised hours, COD, weekend, dispatch hour and log value. The original bookkeeping fields remain available to *test whether they carry signal*, rather than being silently discarded.
5. **Validated before modelling:** confirmed all 1,200 rows survive the hub merge, target and features are complete, delay definition uses `delivery_hours > promised_hours`, and every group total reconciles with the complete population.

## Model results (regression)

The pinned two-step split is **840 train / 180 validation / 180 test**, `random_state=42`; only training rows fit the scaler. See `outputs/part_b_checks.csv`, `part_b_metrics.csv`, `part_b_tree_depth.csv`, `part_b_gb_learning_rate.csv` and `part_b_cv.csv`.

| Candidate | Validation RMSE (hours) | Validation R² |
| --- | ---: | ---: |
| Mean baseline | 18.8552 | -0.0212 |
| Linear regression | **4.8912** | **0.9313** |
| Decision tree (depth 6) | 6.2796 | 0.8867 |
| Random forest | 5.5095 | 0.9128 |
| Gradient boosting | 4.9165 | 0.9306 |

Baseline R² **-0.0212** means the training mean predicts validation slightly worse than its own mean. **Linear regression** won by validation RMSE and obtained held-out test **RMSE 4.5235 hours / R² 0.9493**; only this selected model was evaluated on the test set.

The depth-1 tree has high bias (train/validation RMSE **11.6334/12.1324**). Depth **6** minimises validation error (**6.2796**). Unlimited depth perfectly fits training (**0.0000**) but deteriorates to validation RMSE **7.8368**, illustrating high variance. Boosting learning rate **0.1** worked best (**R² 0.9306**) versus **0.01** (**0.7127**, too little learning in 100 trees) and **1.0** (**0.8440**, overshooting). Five-fold CV slightly favoured boosting (**mean R² 0.9353**, SD 0.0045) over linear regression (**0.9261**, SD 0.0048). Despite this reversal, the pinned selection criterion is lowest **validation** RMSE, so linear regression is the model to ship under the brief. The small gap merits monitoring.

### Regression experiment and selection steps

1. **Reserved an untouched final test:** used two deterministic random splits to isolate training, validation and test observations. The scaler's training means were checked against the non-zero transformed test means as a leakage diagnostic.
2. **Established a genuine baseline:** predicted the historical training mean, making model improvement interpretable against a simple operational alternative.
3. **Explored complexity before choosing a tree:** tested eight maximum depths, identifying underfitting at depth 1 and overfitting without a depth limit.
4. **Held the comparison constant:** evaluated all four trained candidates with MAE, MSE, RMSE and R² on the *same* validation set; separately tested three boosting learning rates.
5. **Challenged the winner:** used the same shuffled five-fold CV partitions for the two strongest candidates on training-plus-validation rows, recorded the ranking disagreement, and then assessed only the validation-selected winner on the untouched test set.

## Model results (classification)

The stratified split gives **840/180/180** samples with **195/42/41** delayed, respectively. Sources: `outputs/part_c_models.csv`, `part_c_feature_scores.csv`, `part_c_selectk.csv`, `part_c_tuning.csv`, `part_c_thresholds.csv`, `part_c_final_config.csv`, `part_c_final_test.csv`.

The all-on-time baseline achieves **0.7667 validation accuracy** yet **0.0000 recall**, missing all 42 delayed shipments. A model that never raises an alert is operationally useless. Validation F1: logistic regression **0.8046**, best decision tree (depth 6) **0.7470**, default 17-feature forest **0.7848**.

Five bookkeeping noise columns account for **0.2245** (22.45%) of the forest's combined importance, despite all five ranking **outside the top seven** by training-only ANOVA F-score. The selected seven features are distance, hub tier, daily hub capacity, stops, promised hours, rain and sale-event flag. SelectKBest validation F1 was **0.6988** (k=5), **0.8333** (k=7), **0.7857** (k=10), **0.8049** (k=12), and **0.7848** (k=17).

Tuning did not beat the same seven-feature default forest: grid search **27 candidates/135 fits**, best CV accuracy **0.8952**, validation F1 **0.7750**; randomized search **9 candidates/45 fits**, CV accuracy **0.8905**, F1 **0.7848**; untuned seven-feature forest **F1 0.8333**. This is evidence that more tuning does not necessarily improve generalisation, especially with 180 validation rows.

**Threshold choice:** Select the highest threshold with validation recall at least 0.80. The winner is **0.5**, delivering validation **precision 0.7955, recall 0.8333, F1 0.8140**. Threshold 0.6 increases precision to **0.8889** but lowers recall to **0.7619**, breaching the minimum. At threshold 0.5 the single untouched test evaluation gives **precision 0.6889, recall 0.7561, F1 0.7209** (TN 125, FP 14, FN 10, TP 31). Test recall is **0.0772** below validation recall; with just **41** delayed test parcels, it should not be overstated.

### Classification experiment and threshold steps

1. **Protected minority-class representation:** stratified both splits so delayed parcels were distributed as **195 training / 42 validation / 41 test**, rather than trusting overall accuracy on an imbalanced problem.
2. **Compared operationally meaningful baselines:** measured false negatives and false positives alongside accuracy, precision, recall and F1. The all-on-time classifier illustrates why 76.67% accuracy is not useful for prevention.
3. **Separated signal from noise:** fitted ANOVA F-scores and forest importance on training rows only; investigated five deliberately irrelevant bookkeeping fields, then compared k=5, 7, 10, 12 and 17.
4. **Tested whether tuning added value:** applied the 27-combination grid and nine-trial randomized searches to the *same seven selected inputs*, keeping validation comparisons fair. The untuned option outperformed the searches and was selected under the stated tie-break rule.
5. **Optimised the action rule, not accuracy alone:** evaluated literal thresholds **0.1–0.9**; chose the **highest** one with at least **0.80 validation recall**. Tested the locked model and threshold once, and reported the lower test recall transparently.

## Operations recommendation

Hub-level evidence and K-Means are in `outputs/hub_profile.csv`, `hub_segments.csv`, `segment_profile.csv`, `tier_vs_segment_mismatch.csv` and `part_d_separation.csv`. Clusters are named by their mean **hub-level** delay rate.

| Segment | Hubs | Average hub delay rate | Average delivery hours | Playbook |
| --- | ---: | ---: | ---: | --- |
| Standard | 7 | 0.0257 | 47.2396 | Routine exception handling |
| Watch | 8 | 0.1550 | 56.8137 | Prioritise customer contact and lane review |
| Escalate | 9 | 0.4600 | 64.6504 | Hub-manager attention for flagged parcels |

**Escalate hubs and individual delayed shares:** HUB13 Nagpur **0.4600**; HUB15 Kochi **0.3600**; HUB17 Patna **0.4600**; HUB18 Ranchi **0.3800**; HUB19 Guwahati **0.6200**; HUB20 Raipur **0.4800**; HUB21 Bhubaneswar **0.4000**; HUB23 Siliguri **0.4400**; HUB24 Agartala **0.5400**. The **0.4600** segment figure is an average, not each hub's individual rate.

Four labels mismatch observed behaviour: **HUB07 Kolkata (Tier 1 → Watch)**, **HUB13 Nagpur (Tier 2 → Escalate)**, **HUB15 Kochi (Tier 2 → Escalate)** and **HUB22 Dehradun (Tier 3 → Watch)**. Operational interventions should reflect the observed segment rather than blindly following the nominal tier.

Keep **K=3** because the organisation supports three playbooks, with separation ratio **5.3928** versus **2.8940** for K=2. K=4 increases the ratio to **6.3047** and K=5 to **6.8866**, but these create clusters containing only **4** and **2** hubs respectively, too small for a separate standardised network playbook.

At threshold **0.5**, test precision **0.6889** means approximately **6.9 of every 10 alerted parcels** were actually late, while recall **0.7561** catches **7.6 of every 10 actually late parcels**. On the **48** new parcels the model flags **8**, with **6** hub-manager escalations in the Escalate segment, **2** proactive customer calls elsewhere and **40** no-action outcomes. See `outputs/new_shipment_actions.csv`.

### Steps from hub behaviour to parcel-level interventions

1. **Summarised actual hub performance:** for each hub, computed its shipment count, average delivery hours, late share, mean distance and mean rider stops; retained the prescribed ascending HUB01–HUB24 ordering.
2. **Standardised before clustering:** scaled the four behavioural profile measures so hours and kilometres could not dominate solely through their numerical units. Fit K-Means with `random_state=42` and `n_init=10`.
3. **Checked separation and playbook feasibility:** calculated the ratio of mean across-cluster pairwise distance to mean within-cluster distance for K=2, 3, 4 and 5. Higher ratios alone were not sufficient to justify operationally tiny groups.
4. **Made labels meaningful:** ordered clusters by actual average delay share and assigned **Standard**, **Watch** and **Escalate**; compared them to the inherited Tier 1/2/3 designation.
5. **Built the final action register:** reloaded the packaged classifier and historical imputation values; applied the shared feature code to new parcels; scored probabilities; flagged at the stored threshold; then assigned `Escalate to hub manager`, `Proactive customer call` or `No action` using both probability and segment.

**Decision discipline:** a parcel from an Escalate hub is *not automatically escalated*; escalation requires both `flagged = 1` **and** `segment = Escalate`. This prevents costly blanket escalation based only on hub identity.

## Production plan

| Quality | Repository evidence or proposed deployment control |
| --- | --- |
| Accuracy | Validation/test RMSE, F1, precision/recall, recall-constrained threshold |
| Reliability | Input cleaning, sentinel handling, non-null checks and model reload comparison |
| Reproducibility | Fixed generator, exact seeds, pinned dependency versions, ordered offline commands |
| Performance | Seven selected features and batch scoring; add latency measurement |
| Maintainability | Separate Part A–D scripts and shared `src/shipment_prep.py` |
| Consistency | Stored medians, feature order, threshold; same code for training and scoring |
| Monitorability | Model, hub, threshold, feature-score and action CSV audit outputs |

Proposed monitoring triggers (to be calibrated with real operational data): investigate **weekly delivered delay share above 30%** for two comparable weeks, **weekly flag rate above 35% or below 10%** for two comparable weeks, or **rolling four-week recall below 0.70** once outcomes are available. Also investigate feature missingness above **5%** and p95 scoring latency exceeding **2 seconds per 100 parcels**.

**Retrain quarterly**, review metrics monthly, and retrain earlier if monitoring breaches persist. The **15–28 September sale event** should be explicitly identified with `sale_event_flag` and benchmarked against similar sale periods, not ordinary weeks. Drift could appear as sustained shifts in lane distance, stops, rain prevalence, service mix, hub capacity, delay rates or score calibration. Validate new models and thresholds on later, genuinely unseen dispatch periods.

### Reproducibility and grading evidence

The published repository contains the prescribed script, data, model and CSV paths on the public default branch `main`. Its clean-checkout GitHub Actions run installed the pinned dependencies, executed the seven Python commands in sequence, checked core counts, regenerated the model, compared two scoring-output hashes and committed the generated files. The temporary workflow was removed afterward so the final submission contains only the permitted deliverables.

Validation gates include: **24/1,224/48** raw generator counts; **1,200** historical modelling rows; **278** delayed parcels; **840/180/180** model splits; prescribed feature order; **24** clustered hubs with four tier mismatches; **180** pickle-reload test scores with zero probability difference; and **48** deterministic scored actions. These are reproducibility checks, not claims of a real-world deployment.

## Limitations

1. Entirely **synthetic** data; actual network constraints and cost trade-offs are unverified.
2. Only **41** delayed parcels in the classification test set; uncertainty around recall and precision is material.
3. The model, feature count and alert threshold were all selected using one **180-row validation set**; temporal validation is preferable for deployment.
4. No probability calibration or explicit INR cost model for missed alerts versus customer calls and escalations.
5. Hub clusters use only four aggregate measures and **50 historical shipments per hub**, with K fixed for operational reasons.
6. The batch script is not a live service; concept drift, data quality and workload need monitoring before any real rollout.

The final submission is the public [GitHub repository on main](https://github.com/SQB-KHN/masai_capstone/tree/main). All performance numbers above are drawn from named CSV outputs rather than typed as fixed model targets.
