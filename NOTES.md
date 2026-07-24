# Telecom Churn Prediction System

**Cramer's V**: a statistical measure of the strength of association between two categorical variables.

* 0.0: No relationships or independence between variables.
* 0.1 to 0.3: Weak association
* 0.3 to 0.5: Moderate to strong association.
* .> 0.5: Very strong association. 

**Chi-Square p_value**: measures the probability that an observed difference between data and expected values happened by chance.

* **Null Hypothesis (H_0)**: Assumes no real relationship or difference between the categorical variables.
    
    * **p < 0.05**: This indicates a statistically significant result, meaning we can reject the Null Hypothesis

    * **p >= 0.05**: This indicates that the data and the expected values are happen to be by a random chance, so we fail to reject the null hypothesis. 

- **Pearson correlation** only captures linear relationships

## Decisions 
**TotalCharges**
- Found `TotalCharges = AccountAge × MonthlyCharges` exactly (residual ~1e-12, floating point only, not real variance) — a deterministic identity, not just a strong correlation. Confirmed via two derived ratio columns: `avg_charge_per_month(AccountAge)` (= TotalCharges/AccountAge) correlated 1.000000 with `MonthlyCharges`, and `avg_charge_per_month(MonthlyCharges)` (= TotalCharges/MonthlyCharges) correlated 1.000000 with `AccountAge` — both were exact duplicates of existing columns, adding zero new information.
- Dropped all three: `TotalCharges`, `avg_charge_per_month(AccountAge)`, `avg_charge_per_month(MonthlyCharges)`. Kept `AccountAge` and `MonthlyCharges` (correlation ≈ 0.0017, independent) as the two underlying signals — this applies to every model type (linear and tree-based alike), since the redundancy is deterministic, not just a linear-model concern.

- I dropped the features `DeviceRegistered, MultiDeviceAccess, PaperlessBilling` because their p_value is greater than 0.05 meaning we fail to reject the null hypothesis, there is no statistically significant relationship with the `Churn` feature.

**UserRating and WatchlistSize**
- Both features show very weak linear correlation with Churn (UserRating: 0.022, WatchlistSize: 0.022). Rather than dropping them from the dataset, they will be exluded only from the linear model's **ColumnTransformer**, since pearson correlation only captures linear relationships, but these features may still hold nonlinear relationships. they will be kept in for **tree-based model training (RandomForest, BalancedRandomForest, CatBoost)** so their actual feature importance can be checked post-training — this also catches any interaction effects that univariate correlation/MI can't detect. Will drop only if feature importance confirms they're uninformative in the trained models too. Already excluded from the linear model's ColumnTransformer regardless, since linear models can't exploit interaction effects the same way.

## Conclusions
**Engagement features (negative correlation → more engagement = less churn):**
- AccountAge (-0.198): churners avg 45.7 months tenure vs. 63.3 for retained — newer customers churn more
- AverageViewingDuration (-0.147): churners watch 76.5 min/session vs. 95.8 — less engaged sessions
- ContentDownloadsPerMonth (-0.130): churners download 20.5/month vs. 25.4
- ViewingHoursPerWeek (-0.129): churners watch 17.4 hrs/week vs. 21.2

**Friction/cost features (positive correlation → more cost/friction = more churn):**
- MonthlyCharges (+0.100): churners pay $13.41 vs. $12.29 — price sensitivity signal
- SupportTicketsPerMonth (+0.084): churners raise 5.02 tickets/month vs. 4.39 — support friction signal

**Negligible:**
- UserRating (+0.022), WatchlistSize (+0.022) — barely distinguishable from noise, similar situation to the weak categorical features

---

**Outlier detection**
- IQR method (1.5×IQR beyond Q1/Q3) applied to all 8 numeric features (AccountAge, MonthlyCharges, ViewingHoursPerWeek, AverageViewingDuration, ContentDownloadsPerMonth, UserRating, SupportTicketsPerMonth, WatchlistSize). Zero outliers found across all features. No outlier handling (clipping/winsorizing) needed in the preprocessing pipeline.

## Feature Engineering

**is_new_customer**
- Created `is_new_customer` (Yes/No) from `AccountAge`, threshold: `AccountAge < 12` month
s. Threshold chosen as the standard "first year" convention rather than an arbitrary data-
driven cutoff, and validated against the `AccountAge` 10th percentile (13 months) — confir
ms the cutoff captures a meaningful minority segment (~9.06% of customers: 22,085 Yes / 22
1,702 No).
- Validated against Churn: chi2 = 2927.97, p_value ≈ 0 (reject null — real relationship),
Cramér's V = 0.1096 — the strongest categorical-style signal found so far, ~3x stronger th
an any raw categorical feature (best was `SubscriptionType` at 0.036).
- Checked redundancy with `AccountAge`: correlation = -0.498. Moderate, well below the mul
ticollinearity concern threshold (~0.7-0.8, same threshold that flagged the old `TotalChar
ges`/`AccountAge` pair at 0.82), confirming this is a complementary non-linear signal (cap
tures a threshold/kink effect) rather than a duplicate. Keeping both `AccountAge` and `is_
new_customer` in the linear pipeline; for tree-based models the added value is smaller (tr
ees can discover this split on their own) but harmless to keep.

## Modeling

**Pipeline setup**
- Two `ColumnTransformer`s: `linear_preprocessor` (StandardScaler + OneHotEncoder, `remainder="drop"`, excludes UserRating/WatchlistSize) for LogisticRegression, and `tree_preprocessor` (OrdinalEncoder for categoricals + explicit passthrough of numeric_cols_tree, `remainder="drop"`) for RandomForest/BalancedRandomForest. CatBoost bypasses both — takes raw `X[tree_feature_cols]` directly with `cat_features=categorical_cols`.
- Imbalance handling per model: `class_weight="balanced"` for LogisticRegression and RandomForest, native balanced bootstrap sampling for BalancedRandomForest, `auto_class_weights="Balanced"` for CatBoost.
- Train/test split: `stratify=y`, `test_size=0.2`, `random_state=42`. 5-fold `StratifiedKFold` CV used for model comparison on `X_train` only — `X_test` held out untouched until a final model is chosen, to avoid biasing the final performance estimate.

**Model comparison (5-fold CV on X_train)**

| model | roc_auc | pr_auc | f1 | recall |
|---|---|---|---|---|
| logistic_regression | 0.7480 | 0.4004 | 0.4376 | 0.6913 |
| catboost | 0.7418 | 0.3927 | 0.4347 | 0.6710 |
| balanced_rf | 0.7343 | 0.3728 | 0.4245 | 0.5147 |
| random_forest | 0.7268 | 0.3666 | 0.1087 | 0.0602 |

**RandomForest diagnosis**
- Plain RandomForest's F1/recall are far worse than its ROC-AUC/PR-AUC would suggest. Root cause: `class_weight="balanced"` only reweights the split criterion (Gini/entropy) — it doesn't resample what each tree's bootstrap sees, so predicted probabilities stay compressed toward the majority class. Confirmed via `cross_val_predict`: only 2.19% of predictions crossed the default 0.5 threshold, vs. an actual 18.12% churn rate (probability distribution: median 0.15, max 0.85).
- `BalancedRandomForestClassifier` avoids this by actually undersampling the majority class per tree, giving much better-calibrated probabilities (recall 0.51 vs. 0.06 on the same algorithm family).
- Decision: drop plain `RandomForestClassifier` from further consideration — `BalancedRandomForestClassifier` strictly dominates it here. Remaining comparison is `logistic_regression` vs. `catboost` (top two on every metric), with `balanced_rf` a step behind both.

**Threshold tuning (logistic_regression, on held-out X_test)**
- Default threshold (0.5) with `class_weight="balanced"`: precision 0.32, recall 0.70 for the churn class — too many false "will churn" flags for the business's retention-campaign budget.
- Swept thresholds 0.10–0.90 in 0.05 steps, tracking precision/recall/F1/F0.5 (F0.5 weights precision higher, matching the stated priority of avoiding false positives). Best F1 at threshold 0.55 (precision 0.349, recall 0.623). Best F0.5 at threshold 0.70 (precision 0.455, recall 0.343).
- Precision has a hard ceiling around ~0.45–0.55 no matter how far the threshold is pushed — beyond ~0.75 recall collapses to near-zero (catches almost no real churners). This ceiling reflects the model's overall discriminative power (ROC-AUC ~0.75), not the threshold choice — going higher requires a better model, not more threshold tuning.
- Also tested removing `class_weight="balanced"` entirely: at its default 0.5 threshold, precision 0.57 / recall 0.12 — but ROC-AUC/PR-AUC essentially unchanged (0.754/0.408 vs. 0.748/0.400 weighted), and a full threshold sweep on the unweighted model landed on nearly identical best-F1/F0.5 operating points as the weighted model. Conclusion: `class_weight="balanced"` and threshold tuning are two knobs on the same dial for this model — redundant, not complementary. Threshold tuning on either variant is sufficient; no need to grid-search both.
- Recommended operating point: **threshold ≈ 0.65–0.70**, giving precision ~0.41–0.46 at recall ~0.34–0.45 — the best precision gain before recall drops off a cliff.

**Final comparison: logistic_regression vs. catboost (on held-out X_test)**
- Ran the same threshold sweep for CatBoost's probabilities. Results track logistic_regression's curve almost exactly at every threshold (e.g. best F1 at 0.55: CatBoost precision 0.345/recall 0.612/F1 0.4409 vs. LR precision 0.349/recall 0.623/F1 0.4476; best F0.5 at 0.70: CatBoost precision 0.448/recall 0.345/F0.5 0.4225 vs. LR precision 0.455/recall 0.343/F0.5 0.4271).
- CatBoost does not meaningfully outperform LogisticRegression on this dataset/feature set — curves are nearly identical.
- **Decision: go with `LogisticRegression`.** Equal performance, but simpler, faster to train, and gives interpretable coefficients — useful for explaining *why* a customer is flagged as at-risk to whoever runs the retention campaign.

**Further feature engineering (interactions, ratios, composite score) — no improvement**
- Added `UserRating`/`WatchlistSize` (raw) + `UserRating × SupportTicketsPerMonth` + `WatchlistSize × ContentDownloadsPerMonth` interaction terms to the linear pipeline (targeting the two features CatBoost valued most despite near-zero univariate correlation). Result: negligible change (ROC-AUC +0.001, PR-AUC +0.0015).
- Checked `UserRating`/`WatchlistSize` churn rate by decile: smooth, weak, monotonic trend (~2.5 point spread top-to-bottom) — no threshold/kink pattern like `AccountAge` had, so binning wouldn't help; CatBoost's higher importance for these features is most likely from interaction effects, not a discoverable marginal pattern.
- Tried `cost_per_viewing_hour` (MonthlyCharges/ViewingHoursPerWeek), `support_tickets_per_tenure` (SupportTicketsPerMonth/AccountAge), and `engagement_score` (composite z-score average of ViewingHoursPerWeek, AverageViewingDuration, ContentDownloadsPerMonth). Despite strong individual correlations with Churn (engagement_score: -0.234, the strongest single predictor found in the whole dataset), adding them to the full feature set produced no real improvement for LogisticRegression (+0.001 ROC-AUC) and a very slight regression for CatBoost (-0.0008 ROC-AUC). Reason: these are largely redundant recombinations of features already in the model (`engagement_score` is literally a linear combination of 3 existing features — mathematically near-redundant for a linear model, which can already form that same combination via its own coefficients).
- **Conclusion: ~0.75 ROC-AUC / ~0.40 PR-AUC is the practical ceiling for this dataset's information content.** Confirmed independently by both a simple linear model and a much more flexible boosted tree model converging on the same performance, and by three different feature engineering angles all failing to move it. Further gains would require genuinely new data (e.g. usage trend over time, support-call sentiment, payment history) rather than reshaping existing columns.

## Final Model Decision

**We choose `LogisticRegression` as our baseline model** because:
1. **Performance parity with CatBoost**: across every CV metric and the full precision-recall curve on held-out `X_test`, `LogisticRegression` matched or slightly exceeded CatBoost (ROC-AUC 0.748 vs 0.742, PR-AUC 0.400 vs 0.393). A much more flexible model failed to beat it, indicating the dataset's available signal — not model capacity — is the limiting factor.
2. **Simplicity and interpretability**: `LogisticRegression` gives directly interpretable coefficients, which matters for explaining *why* a customer is flagged as at-risk to the team running retention campaigns (ties into the planned `explain.py` work), and is far simpler to deploy, monitor, and retrain than a boosted tree model.
3. **`BalancedRandomForest` and plain `RandomForest` both underperformed**: plain `RandomForest` was ruled out entirely due to a probability-calibration failure (`class_weight="balanced"` didn't rebalance bootstrap sampling, so recall collapsed to 0.06 at the default threshold); `BalancedRandomForest` was consistently a step behind both `LogisticRegression` and CatBoost on every metric.
4. **Business value simulation confirmed the choice is actionable, not just statistically adequate**: at the chosen operating threshold (~0.65-0.70), the model produces a positive net benefit (~$114K-$122K under placeholder cost assumptions: customer_value=$500, retention_cost=$50, save_rate=30%) versus taking no action at all — and the profit-maximizing threshold (0.70) aligns closely with the F0.5-optimal threshold found earlier from the precision-recall tradeoff alone, cross-validating both approaches.
5. Operating threshold: **~0.70**, chosen because it is simultaneously near-optimal for F0.5 (precision-weighted, matching the business's stated preference to minimize false "will churn" flags) and for the simulated net profit curve. Below threshold ~0.50, the model actually becomes net-negative vs. no model at all, due to retention campaign costs on false positives outweighing value saved — a reminder that classification metrics alone (recall/F1) do not guarantee business value.

## Hyperparameter Tuning (04_model_finetuning.ipynb)

- Ran `GridSearchCV` over `LogisticRegression`'s `penalty`/`C`/`solver` (reduced grid: `l2`/`lbfgs`, `C` in [0.01, 0.1, 1, 10] — the original full grid across l1/l2/elasticnet penalties and saga/liblinear solvers crashed the machine; full sweep deferred to a more powerful machine).
- Business value at the profit-maximizing threshold (0.70) barely moved: `net_benefit_with_model` went from $121,600 (default params) to $121,550 (grid-searched params) — a ~$50 swing on a $4.4M baseline (~0.04%), i.e. noise (most likely a single customer's predicted probability crossing the 0.70 cutoff differently), not a real regression.
- **Conclusion: hyperparameter tuning does not move the needle.** Consistent with the practical performance ceiling (~0.75 ROC-AUC / ~0.40 PR-AUC) already established above — the limiting factor is the dataset's information content, not the model's parameters. **Keeping the default-parameter `LogisticRegression(class_weight="balanced")` as the final model** rather than the grid-searched one; no reason to add complexity for a statistically meaningless difference.