# Technical Details — Explainable Machine Learning for Hotel Booking Cancellations: The Role of Public Holidays

**Author:** Nurbek Suvonov  
**Affiliation:** Graduate School of Business IT, Kookmin University  
**Advisor:** Prof. Hyunchul Ahn  
**Year:** 2026  
**Source Notebook:** `analysis.ipynb`

---

## 1. Project Overview

### 1.1 Objective

This study develops and evaluates explainable machine learning models for predicting hotel booking cancellations, with a specific focus on the differential impact of a novel **public holiday feature** on resort versus city hotel cancellation predictions.

### 1.2 Problem Definition

Hotel booking cancellations impose significant revenue uncertainty on the hospitality industry. While prior work has explored cancellation prediction using booking-level features, the role of public holidays as a contextual driver of cancellation behaviour has not been systematically investigated. This study addresses this gap by engineering a public holiday indicator derived from Portuguese national holidays and evaluating its marginal contribution to model performance across two hotel types (resort and city).

### 1.3 Research Questions

- **RQ1:** How accurately can Logistic Regression, Random Forest, and XGBoost models predict hotel booking cancellations for resort and city hotels, and what factors account for differences in predictive performance across hotel types?
- **RQ2:** Which model achieves the best predictive performance for each hotel type, and why?
- **RQ3:** What are the key factors driving booking cancellations for resort and city hotels, as identified through SHAP analysis?
- **RQ4:** How do public holidays influence cancellation patterns differently between resort and city hotels?

---

## 2. Dataset Description

### 2.1 Data Source

The dataset originates from the publicly available hotel booking demand dataset published by Antonio, Almeida, and Nunes (2019):

> Antonio, N., de Almeida, A., & Nunes, L. (2019). Hotel booking demand datasets. *Data in Brief*, 22, 41–49. https://www.sciencedirect.com/science/article/pii/S2352340918315191

A pre-cleaned version of this dataset (`cleaned_dataset.csv`) is used as input. The dataset covers bookings from **July 2015 to August 2017** at two Portuguese hotels (one resort hotel and one city hotel).

### 2.2 Dataset Dimensions

- **Total bookings:** Loaded via `pd.read_csv('cleaned_dataset.csv')` with shape reported at runtime.
- **Features:** The dataset contains multiple features including booking-level, temporal, guest, and hotel characteristics. The exact feature count is reported at runtime from `df.shape[1]`.
- **Missing values:** Zero (`df.isnull().sum().sum()` confirms no missing values in the cleaned dataset).

### 2.3 Target Variable

- **`is_canceled`**: Binary variable (0 = not canceled, 1 = canceled).
- Overall cancellation rate is reported at runtime.
- Resort Hotel cancellation rate is lower than City Hotel cancellation rate (Resort ~27.8%, City ~41.3% based on notebook outputs).

### 2.4 Hotel Type Distribution

The dataset is pre-encoded with a binary `hotel` column:
- **Resort Hotel (hotel = 1)**
- **City Hotel (hotel = 0)**

The city hotel constitutes the larger share of total bookings.

### 2.5 Class Imbalance

Class imbalance is present and differs between hotel types:
- **Resort Hotel:** Approximately 72.2% not canceled / 27.8% canceled (majority class dominant).
- **City Hotel:** Approximately 58.7% not canceled / 41.3% canceled (less imbalanced).

This imbalance is explicitly addressed via random undersampling of the training set (see Section 3.3).

### 2.6 Data Types and Structure

The cleaned dataset is entirely numeric (pre-encoded). Categorical features have already been one-hot encoded prior to loading. Key feature groups include:

| Feature Group | Examples | Encoding |
|---|---|---|
| Temporal | `arrival_date_year`, `arrival_date_month`, `arrival_date_week_number`, `arrival_date_day_of_month` | Numeric |
| Stay duration | `stays_in_weekend_nights`, `stays_in_week_nights` | Numeric |
| Guest attributes | `adults`, `children`, `babies`, `is_repeated_guest`, `previous_cancellations`, `previous_bookings_not_canceled` | Numeric |
| Booking details | `lead_time`, `adr`, `booking_changes`, `days_in_waiting_list`, `total_of_special_requests`, `required_car_parking_spaces` | Numeric |
| Country | `country_PRT`, `country_GBR`, `country_FRA`, `country_ESP`, `country_DEU`, `country_ITA`, `country_BEL`, `country_NLD`, `country_BRA`, `country_IRL`, `country_USA`, `country_OTHER` | One-hot encoded |
| Market segment | `MS_Online TA`, `MS_Offline TA/TO`, `MS_Direct`, `MS_Corporate`, `MS_Groups`, `MS_Complementary`, `MS_Aviation` | One-hot encoded |
| Customer type | `CT_Transient`, `CT_Transient-Party`, `CT_Contract`, `CT_Group` | One-hot encoded |
| Distribution channel | `DCH_TA/TO`, `DCH_Direct`, `DCH_Corporate`, `DCH_GDS` | One-hot encoded |
| Reserved room type | `RRT_A`, `RRT_B`, `RRT_D`, `RRT_E`, `RRT_F`, `RRT_G`, `RRT_H`, `RRT_L` | One-hot encoded |
| Assigned room type | `ART_A`, `ART_B`, `ART_C`, `ART_D`, `ART_E`, `ART_F`, `ART_G`, `ART_H`, `ART_I`, `ART_J`, `ART_K` | One-hot encoded |
| Agent/company | `agent_binary`, `company_binary` | Binary |
| Meal | `meal` | Numeric |

---

## 3. Data Preprocessing

### 3.1 Input Data State

The dataset is loaded from `cleaned_dataset.csv`, which has already been cleaned and numerically encoded. No additional missing value imputation or basic cleaning is performed in the notebook, as these steps were handled in a prior pipeline.

### 3.2 Multicollinearity Handling

One column per one-hot encoded group is dropped to eliminate **perfect multicollinearity** (each group of binary dummies sums to exactly 1.0). This is critical for Logistic Regression coefficient stability.

**Dropped reference columns:**

| One-Hot Group | Dropped Column |
|---|---|
| Country | `country_OTHER` |
| Market Segment | `MS_Aviation` |
| Customer Type | `CT_Group` |
| Distribution Channel | `DCH_GDS` |
| Reserved Room Type | `RRT_L` |
| Assigned Room Type | `ART_K` |


### 3.3 Hotel-Type Separation

The dataset is split into two independent subsets before any modeling:
- **Resort Hotel subset:** All rows where `hotel == 1`.
- **City Hotel subset:** All rows where `hotel == 0`.

### 3.4 Train/Test Split and Class Balancing

**Pipeline flow:**
```
Full data → Split by hotel → 80/20 Train/Test split (stratified) → Undersample ONLY training set → Train → Evaluate on ORIGINAL imbalanced test set
```

Implementation details:
- **Split method:** `train_test_split` with `test_size=0.2`, `stratify=y`, `random_state=42`.
- **Undersampling:** Random undersampling of the majority class in the training set to match the minority class count. If the canceled class is the minority, the not-canceled class is downsampled, and vice versa.
- **Test set:** Retains the original imbalanced class distribution for realistic evaluation.
- **Shuffling:** The balanced training set is shuffled (`sample(frac=1, random_state=42)`) after concatenation.

### 3.5 Feature Scaling

- Random Forest and XGBoost do not require feature scaling and receive unscaled features.

### 3.6 Feature Sets

Two feature sets are prepared for each hotel type:
1. **Without Holiday:** All features excluding `is_public_holiday` (and `holiday_x_domestic`).
2. **With Holiday:** All features including `is_public_holiday` and `holiday_x_domestic`.

This enables direct comparison of model performance with and without the holiday feature.

---

## 4. Feature Engineering

### 4.1 Public Holiday Indicator (`is_public_holiday`)

This is the primary novel contribution of the study. The feature is a binary indicator created by cross-referencing each booking's arrival date against official Portuguese national public holidays for the dataset period (July 2015 – August 2017).

**Construction procedure:**

1. **Reconstruct arrival date:** A temporary `arrival_date` column is derived from `arrival_date_year`, `arrival_date_month`, and `arrival_date_day_of_month`.
2. **Define holiday centers:** 27 Portuguese national public holiday dates across 2015–2017 are manually specified.
3. **Expand into windows:** Each holiday date is expanded into a ±3-day window (the window size itself is tested in the sensitivity analysis, §4.2).
4. **Create binary feature:** `is_public_holiday = 1` if the booking's arrival date falls within any holiday window; 0 otherwise.
5. **Drop temporary column:** The reconstructed `arrival_date` column is dropped after feature creation.

**Portuguese National Holidays Used:**

| Year | Holidays |
|---|---|
| 2015 | Aug 15 (Assumption of Mary), Oct 5 (Republic Day)*, Nov 1 (All Saints' Day)*, Dec 1 (Restoration of Independence)*, Dec 8 (Immaculate Conception), Dec 25 (Christmas) |
| 2016 | Jan 1, Mar 25, Mar 27, Apr 25, May 1, May 26, Jun 10, Aug 15, Oct 5, Nov 1, Dec 1, Dec 8, Dec 25 |
| 2017 | Jan 1, Apr 14, Apr 16, Apr 25, May 1, Jun 10, Jun 15, Aug 15 |


**Holiday feature distribution:** Approximately 23.4% of bookings fall within a ±3-day holiday window.

### 4.2 Holiday Window Sensitivity Analysis

A systematic sensitivity analysis tests window sizes {±0, ±1, ±2, ±3, ±4, ±5, ±7} to determine the optimal trade-off between signal strength and noise dilution. For each window, the following are computed per hotel type:
- Percentage of bookings flagged as holiday
- Cancellation rate for regular vs. holiday bookings
- Difference in cancellation rate (percentage points)
- Chi-squared test statistic and p-value
- Cramér's V effect size

**Key results from sensitivity analysis:**

| Window | Hotel | % Flagged | Cancel Regular | Cancel Holiday | Diff (pp) | χ² | p-value | Cramér's V |
|---|---|---|---|---|---|---|---|---|
| ±0 | Resort | 3.6% | 27.64% | 31.57% | +3.93 | 9.68 | 0.0019 | 0.0156 |
| ±0 | City | 3.6% | 41.58% | 47.30% | +5.72 | 37.73 | <0.001 | 0.0219 |
| ±1 | Resort | 10.6% | 27.40% | 30.84% | +3.43 | 22.21 | <0.001 | 0.0235 |
| ±1 | City | 10.6% | 41.41% | 45.04% | +3.63 | 40.19 | <0.001 | 0.0226 |
| ±2 | Resort | 17.3% | 27.18% | 30.52% | +3.35 | 32.20 | <0.001 | 0.0284 |
| ±2 | City | 17.3% | 40.89% | 46.14% | +5.25 | 127.34 | <0.001 | 0.0402 |
| ±3 | Resort | 23.4% | 27.04% | 30.11% | +3.07 | 33.71 | <0.001 | 0.0290 |
| ±3 | City | 23.4% | 40.74% | 45.27% | +4.53 | 119.25 | <0.001 | 0.0389 |
| ±4 | Resort | 28.2% | 26.98% | 29.80% | +2.82 | 31.89 | <0.001 | 0.0282 |
| ±4 | City | 28.2% | 40.61% | 44.79% | +4.18 | 114.68 | <0.001 | 0.0381 |
| ±5 | Resort | 32.6% | 26.90% | 29.63% | +2.73 | 32.14 | <0.001 | 0.0283 |
| ±5 | City | 32.6% | 40.41% | 44.61% | +4.20 | 126.37 | <0.001 | 0.0400 |
| ±7 | Resort | 40.8% | 27.36% | 28.38% | +1.03 | 4.99 | 0.0255 | 0.0112 |
| ±7 | City | 40.8% | 40.06% | 44.27% | +4.22 | 139.65 | <0.001 | 0.0421 |

**Key observation:** All Cramér's V values are below 0.05, indicating **negligible effect sizes** across all window sizes. While statistically significant (due to large sample sizes), the practical magnitude of the holiday effect is minimal.

### 4.3 Chi-Squared Test for Holiday vs. Cancellation

A chi-squared test of independence is conducted between `is_public_holiday` and `is_canceled` for each hotel type, with **Cramér's V** reported as an effect size measure.

**Effect size interpretation:**
- V < 0.1: Negligible
- 0.1–0.3: Small
- 0.3–0.5: Medium
- \> 0.5: Large

The notebook explicitly notes that with large sample sizes (n > 40,000), chi-squared tests have enormous statistical power to detect trivially small differences. The Cramér's V values are reported as negligible.

### 4.4 Holiday × Domestic Guest Interaction Feature (`holiday_x_domestic`)

An interaction term is created: `holiday_x_domestic = is_public_holiday × country_PRT`.


The cancellation rate differences between flagged (1) and unflagged (0) groups are reported for both `is_public_holiday` and `holiday_x_domestic` per hotel type.

---

## 5. Exploratory Data Analysis

### 5.1 Visualizations Produced

A 2×3 panel EDA figure (`figures/eda_overview.png`) includes:

1. **Distribution of Lead Time:** Right-skewed histogram with KDE overlay showing most bookings have short lead times, with a long tail extending to 700+ days.
2. **ADR by Hotel Type:** Box plots comparing average daily rate across city and resort hotels. Resort hotels exhibit greater variance and a wider interquartile range.
3. **Overall Cancellation Distribution:** Bar chart showing counts and percentages of canceled vs. not-canceled bookings.
4. **Cancellation Rate by Hotel Type:** City hotels have a substantially higher cancellation rate (~41.3%) than resort hotels (~27.8%).
5. **Lead Time by Cancellation Status:** Canceled bookings tend to have longer lead times than non-canceled bookings.
6. **Special Requests by Cancellation Status:** Bookings with more special requests are less likely to be canceled.

### 5.2 Class Balance Visualization

A 2×3 panel figure (`figures/class_balance.png`) displays class distributions across:
- Full original dataset (per hotel type)
- Balanced training set (after undersampling)
- Imbalanced test set (original distribution, realistic evaluation)

### 5.3 Holiday Analysis

A 2×2 panel figure (`figures/holiday_analysis.png`) examines:
- Booking volume during regular days vs. holiday windows (per hotel type)
- Cancellation rates during regular days vs. holiday windows (per hotel type)

### 5.4 Key Insights

- Canceled bookings exhibit longer lead times than non-canceled bookings.
- Guests with more special requests are less likely to cancel.
- City hotels have a higher baseline cancellation rate than resort hotels.
- Cancellation rates are slightly higher during holiday windows, but the effect size is negligible (Cramér's V < 0.05).

---

## 6. Model Development

### 6.1 Models Used

Three classification algorithms are employed:

1. **Logistic Regression (LR)** — Linear baseline with elastic net regularization.
2. **Random Forest (RF)** — Ensemble of decision trees with bagging.
3. **XGBoost (XGB)** — Gradient-boosted decision trees.

### 6.2 Experimental Design

A total of **12 configurations** are trained and evaluated:
- 3 models × 2 feature sets (with/without holiday) × 2 hotel types.

Additionally, **12 temporal validation configurations** are trained as a robustness check.

### 6.3 Hyperparameter Tuning

All models are tuned via **GridSearchCV** with:
- **Cross-validation:** 5-fold Stratified K-Fold (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`)
- **Scoring metric:** ROC-AUC (`scoring='roc_auc'`)
- **Parallelization:** `n_jobs=-1`
- **Refit:** Best model is refit on the entire training set (`refit=True`)

#### 6.3.1 Logistic Regression Hyperparameter Grid

| Parameter | Values |
|---|---|
| `C` | [0.001, 0.01, 0.1, 1, 10, 100] |
| `l1_ratio` | [0, 0.5, 1] |


#### 6.3.2 Random Forest Hyperparameter Grid

| Parameter | Values |
|---|---|
| `n_estimators` | [100, 200] |
| `max_depth` | [10, 20, None] |
| `min_samples_split` | [2, 5] |
| `min_samples_leaf` | [1, 2] |

Total combinations: 2 × 3 × 2 × 2 = 24

#### 6.3.3 XGBoost Hyperparameter Grid

| Parameter | Values |
|---|---|
| `n_estimators` | [100, 200] |
| `max_depth` | [3, 5, 7] |
| `learning_rate` | [0.01, 0.1, 0.2] |
| `subsample` | [0.8, 1.0] |

Total combinations: 2 × 3 × 3 × 2 = 36

### 6.4 Best Hyperparameters (from GridSearchCV)

#### Resort Hotel

| Model | Feature Set | Best Parameters | CV AUC | CV AUC Std |
|---|---|---|---|---|
| Logistic Regression | Without Holiday | C=100, l1_ratio=1.0 | 0.9093 | 0.0033 |
| Random Forest | Without Holiday | max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=200 | 0.9553 | 0.0029 |
| XGBoost | Without Holiday | max_depth=7, n_estimators=200, learning_rate=0.1, subsample=0.8 | 0.9550 | 0.0034 |
| Logistic Regression | With Holiday | C=100, l1_ratio=1.0 | 0.9093 | 0.0034 |
| Random Forest | With Holiday | max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=200 | 0.9555 | 0.0031 |
| XGBoost | With Holiday | max_depth=7, n_estimators=200, learning_rate=0.1, subsample=0.8 | 0.9548 | 0.0029 |

#### City Hotel

| Model | Feature Set | Best Parameters | CV AUC | CV AUC Std |
|---|---|---|---|---|
| Logistic Regression | Without Holiday | C=100, l1_ratio=0.0 | 0.8789 | 0.0049 |
| Random Forest | Without Holiday | max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=200 | 0.9495 | 0.0016 |
| XGBoost | Without Holiday | max_depth=7, n_estimators=200, learning_rate=0.2, subsample=1.0 | 0.9461 | 0.0018 |
| Logistic Regression | With Holiday | C=100, l1_ratio=1.0 | 0.8789 | 0.0049 |
| Random Forest | With Holiday | max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=200 | 0.9496 | 0.0013 |
| XGBoost | With Holiday | max_depth=7, n_estimators=200, learning_rate=0.2, subsample=1.0 | 0.9464 | 0.0020 |

**Observations:**
- LR selects `l1_ratio=1.0` (Lasso) for resort (both feature sets) and city with holiday. City without holiday selects `l1_ratio=0.0` (Ridge).
- RF consistently selects `n_estimators=200`, `max_depth=None` (unlimited), `min_samples_leaf=1`, `min_samples_split=2`.
- XGBoost selects `max_depth=7`, `n_estimators=200` across all configurations; learning rate and subsample differ by hotel type.

### 6.5 Training Procedure

1. Features and target are separated; `hotel` column is dropped.
2. Train/test split (80/20, stratified).
3. Training set is balanced via random undersampling.
4. For LR: `StandardScaler` is fit on training data only.
5. GridSearchCV selects best hyperparameters using 5-fold stratified CV on the balanced training set.
6. Best model is refit on the full balanced training set.
7. Evaluation is performed on the original imbalanced test set.

### 6.6 Temporal Validation Split

- **Training set:** Bookings with `arrival_date_year < 2017` (Jul 2015 – Dec 2016).
- **Test set:** Bookings with `arrival_date_year >= 2017` (Jan – Aug 2017).



---

## 7. Model Evaluation

### 7.1 Metrics Used

| Metric | Formula / Description | Threshold-Dependent |
|---|---|---|
| **Accuracy** | (TP + TN) / (TP + TN + FP + FN) | Yes |
| **Recall (Sensitivity)** | TP / (TP + FN) | Yes |
| **Precision** | TP / (TP + FP) | Yes |
| **F1 Score** | 2 × (Precision × Recall) / (Precision + Recall) | Yes |
| **Specificity** | TN / (TN + FP) | Yes |
| **AUC (ROC)** | Area under the Receiver Operating Characteristic curve | **No** |
| **PR-AUC** | Area under the Precision-Recall curve (`average_precision_score`) | **No** |

**Threshold note:** All models use the default decision threshold of 0.5. Since models are trained on balanced data but evaluated on the original imbalanced test set, threshold-dependent metrics are influenced by this mismatch. **AUC and PR-AUC are threshold-independent** and serve as the primary basis for model comparison.

### 7.2 Resort Hotel Results (Random Split)

| Model | Feature Set | Accuracy | Recall | Precision | F1 Score | Specificity | AUC | PR-AUC |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | Without Holiday | 0.8263 | 0.8611 | 0.6390 | 0.7336 | 0.8130 | 0.9098 | 0.7940 |
| Random Forest | Without Holiday | 0.8841 | 0.9033 | 0.7381 | 0.8124 | 0.8768 | 0.9572 | 0.9044 |
| XGBoost | Without Holiday | 0.8799 | 0.9087 | 0.7270 | 0.8078 | 0.8688 | **0.9582** | 0.9022 |
| Logistic Regression | With Holiday | 0.8267 | 0.8611 | 0.6396 | 0.7340 | 0.8135 | 0.9098 | 0.7941 |
| Random Forest | With Holiday | 0.8818 | 0.9020 | 0.7335 | 0.8090 | 0.8740 | 0.9576 | 0.9051 |
| XGBoost | With Holiday | 0.8811 | 0.9096 | 0.7293 | 0.8095 | 0.8702 | **0.9581** | 0.9020 |

**Best model (Resort):** XGBoost with AUC = 0.9582 (without holiday) / 0.9581 (with holiday). The difference is 0.0001 — negligible.

### 7.3 City Hotel Results (Random Split)

| Model | Feature Set | Accuracy | Recall | Precision | F1 Score | Specificity | AUC | PR-AUC |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | Without Holiday | 0.7931 | 0.7775 | 0.7405 | 0.7586 | 0.8043 | 0.8779 | 0.8544 |
| Random Forest | Without Holiday | 0.8783 | 0.8513 | 0.8567 | 0.8540 | 0.8977 | 0.9526 | 0.9444 |
| XGBoost | Without Holiday | 0.8702 | 0.8610 | 0.8339 | 0.8472 | 0.8768 | 0.9478 | 0.9381 |
| Logistic Regression | With Holiday | 0.7930 | 0.7775 | 0.7403 | 0.7585 | 0.8041 | 0.8780 | 0.8545 |
| Random Forest | With Holiday | 0.8814 | 0.8535 | 0.8616 | 0.8575 | 0.9015 | **0.9531** | 0.9448 |
| XGBoost | With Holiday | 0.8705 | 0.8598 | 0.8353 | 0.8474 | 0.8782 | 0.9478 | 0.9380 |

**Best model (City):** Random Forest with AUC = 0.9531 (with holiday) / 0.9526 (without holiday). The difference is 0.0005 — negligible.

### 7.4 Temporal Validation Results

| Model | Hotel | Feature Set | Accuracy | Recall | Precision | F1 Score | Specificity | AUC | PR-AUC |
|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | Resort | Without Holiday | 0.7476 | 0.8705 | 0.5576 | 0.6798 | 0.6929 | 0.8615 | 0.7088 |
| Random Forest | Resort | Without Holiday | 0.8028 | 0.5414 | 0.7481 | 0.6282 | 0.9190 | 0.8376 | 0.7106 |
| XGBoost | Resort | Without Holiday | 0.8038 | 0.5710 | 0.7326 | 0.6418 | 0.9073 | 0.8554 | 0.7365 |
| Logistic Regression | Resort | With Holiday | 0.7475 | 0.8705 | 0.5575 | 0.6797 | 0.6928 | 0.8612 | 0.7078 |
| Random Forest | Resort | With Holiday | 0.8018 | 0.5496 | 0.7395 | 0.6305 | 0.9139 | 0.8509 | 0.7292 |
| XGBoost | Resort | With Holiday | 0.8003 | 0.5614 | 0.7276 | 0.6338 | 0.9066 | 0.8498 | 0.7303 |
| Logistic Regression | City | Without Holiday | 0.7158 | 0.8650 | 0.6187 | 0.7214 | 0.6054 | 0.8411 | 0.8146 |
| Random Forest | City | Without Holiday | 0.7627 | 0.6296 | 0.7706 | 0.6930 | 0.8613 | 0.8455 | 0.8013 |
| XGBoost | City | Without Holiday | 0.7633 | 0.6855 | 0.7392 | 0.7113 | 0.8210 | 0.8455 | 0.8202 |
| Logistic Regression | City | With Holiday | 0.7160 | 0.8656 | 0.6189 | 0.7217 | 0.6053 | 0.8410 | 0.8149 |
| Random Forest | City | With Holiday | 0.7634 | 0.6330 | 0.7700 | 0.6948 | 0.8600 | 0.8437 | 0.7984 |
| XGBoost | City | With Holiday | 0.7675 | 0.6866 | 0.7465 | 0.7153 | 0.8274 | 0.8490 | 0.8239 |

**Key observations from temporal validation:**
- AUC scores are **substantially lower** than random-split results (e.g., Resort XGBoost: 0.8554 temporal vs. 0.9582 random; City RF: 0.8455 temporal vs. 0.9531 random).
- This ~10-point AUC drop indicates **temporal distribution shift** — booking patterns in 2017 differ from 2015–2016.
- The holiday feature continues to show negligible impact under temporal validation.
- Model ranking is partially preserved: tree-based models still outperform LR in most cases, though the gap narrows.

### 7.5 Bootstrap Confidence Intervals for AUC

1,000 bootstrap iterations are used to estimate 95% confidence intervals for test-set AUC on the best models (with and without holiday).

| Configuration | AUC (mean) | 95% CI Lower | 95% CI Upper | CI Width |
|---|---|---|---|---|
| Resort XGB No Holiday | 0.9583 | 0.9544 | 0.9623 | 0.0079 |
| Resort XGB With Holiday | 0.9581 | 0.9543 | 0.9621 | 0.0078 |
| City RF No Holiday | 0.9526 | 0.9496 | 0.9554 | 0.0058 |
| City RF With Holiday | 0.9531 | 0.9502 | 0.9559 | 0.0057 |

**Interpretation:**
- The confidence intervals for with-holiday and without-holiday models **fully overlap** for both hotel types.
- Resort: [0.9543, 0.9621] vs. [0.9544, 0.9623] — nearly identical.
- City: [0.9502, 0.9559] vs. [0.9496, 0.9554] — nearly identical.
- **Conclusion:** The holiday feature has **no statistically significant effect** on model AUC.

### 7.6 Visualizations

- **Confusion matrices** for best models per hotel type (XGBoost for Resort, RF for City) — saved as `figures/confusion_matrix_best_models.png`.
- **ROC curves** for all three models per hotel type with holiday features — saved as `figures/roc_curves_all_models.png`.
- **AUC comparison bar chart** for all 12 configurations — saved as `figures/auc_comparison.png`.

---

## 8. Explainability (SHAP Analysis)

### 8.1 Method

**SHAP (SHapley Additive exPlanations)** via `TreeExplainer` is used to interpret model predictions. SHAP values quantify each feature's contribution to individual predictions, grounded in cooperative game theory.

### 8.2 Models Analyzed

- **Resort Hotel:** XGBoost (best AUC)
- **City Hotel:** Random Forest (best AUC)

### 8.3 SHAP Computation Details

- **Resort (XGBoost):** SHAP values computed on a random sample of 2,000 test observations.
- **City (Random Forest):** SHAP values computed on a random sample of 500 test observations (smaller due to computational cost of RF SHAP; `approximate=True` used).
- Random seed: 42 for reproducible sampling.
- For Random Forest, `shap_values` returns a 3D array (samples × features × classes); the positive-class slice `[:, :, 1]` is selected.

### 8.4 SHAP Outputs

For each hotel type, the following are generated:
1. **Beeswarm plot** — Feature-level SHAP value distribution across all samples.
2. **Bar plot** — Top 15 features by mean |SHAP|.
3. **Dependence plots** — SHAP value vs. feature value for: `lead_time`, `arrival_date_month`, `adr`, `is_public_holiday`.
4. **All features ranked CSV** — Complete ranking of all features by mean |SHAP| and normalized percentage.
5. **Top features CSV** — Top 15 features.

### 8.5 Top 15 Features — Resort Hotel (XGBoost)

| Rank | Feature | Mean |SHAP| | % of Total |
|---|---|---|---|
| 1 | `country_PRT` | 1.3108 | 16.73% |
| 2 | `required_car_parking_spaces` | 1.0393 | 13.26% |
| 3 | `lead_time` | 0.6577 | 8.39% |
| 4 | `total_of_special_requests` | 0.4238 | 5.41% |
| 5 | `MS_Online TA` | 0.3962 | 5.06% |
| 6 | `arrival_date_year` | 0.3576 | 4.56% |
| 7 | `adr` | 0.3570 | 4.56% |
| 8 | `CT_Transient` | 0.3422 | 4.37% |
| 9 | `ART_A` | 0.3221 | 4.11% |
| 10 | `MS_Offline TA/TO` | 0.2688 | 3.43% |
| 11 | `RRT_A` | 0.2309 | 2.95% |
| 12 | `arrival_date_week_number` | 0.2223 | 2.84% |
| 13 | `previous_cancellations` | 0.1912 | 2.44% |
| 14 | `booking_changes` | 0.1728 | 2.20% |
| 15 | `arrival_date_day_of_month` | 0.1351 | 1.72% |

### 8.6 Top 15 Features — City Hotel (Random Forest)

| Rank | Feature | Mean |SHAP| | % of Total |
|---|---|---|---|
| 1 | `total_of_special_requests` | 0.1079 | 13.28% |
| 2 | `country_PRT` | 0.1045 | 12.86% |
| 3 | `lead_time` | 0.0755 | 9.29% |
| 4 | `CT_Transient` | 0.0357 | 4.39% |
| 5 | `adr` | 0.0327 | 4.03% |
| 6 | `CT_Transient-Party` | 0.0309 | 3.80% |
| 7 | `MS_Online TA` | 0.0295 | 3.63% |
| 8 | `booking_changes` | 0.0283 | 3.48% |
| 9 | `previous_cancellations` | 0.0282 | 3.47% |
| 10 | `arrival_date_year` | 0.0259 | 3.19% |
| 11 | `MS_Groups` | 0.0219 | 2.69% |
| 12 | `ART_A` | 0.0218 | 2.69% |
| 13 | `arrival_date_week_number` | 0.0195 | 2.40% |
| 14 | `MS_Offline TA/TO` | 0.0164 | 2.01% |
| 15 | `arrival_date_day_of_month` | 0.0163 | 2.01% |

### 8.7 SHAP Rank of `is_public_holiday`

| Hotel | Model | Rank | Total Features | Mean |SHAP| | % of Total |
|---|---|---|---|---|---|
| Resort | XGBoost | **43** | 64 | 0.22 | 0.22% |
| City | Random Forest | **31** | 64 | 0.56 | 0.56% |

**Interpretation:** The `is_public_holiday` feature ranks in the bottom half of all features for both hotel types. Its contribution to model predictions is minimal.

### 8.8 SHAP Rank of `holiday_x_domestic`

| Hotel | Model | Rank | % of Total |
|---|---|---|---|
| Resort | XGBoost | 38 | 0.34% |
| City | Random Forest | 21 | 1.51% |

The interaction feature `holiday_x_domestic` shows slightly higher importance than the raw `is_public_holiday` in the city hotel context, but remains a minor predictor overall.

### 8.9 Normalized Cross-Hotel SHAP Comparison

Since Resort uses XGBoost and City uses Random Forest, absolute SHAP magnitudes are not directly comparable. Normalized values (% of total mean |SHAP|) enable valid ordinal comparison:

**Top 10 features by average normalized importance:**

| Feature | Resort % | Resort Rank | City % | City Rank | Avg % |
|---|---|---|---|---|---|
| `country_PRT` | 16.73 | 1 | 12.86 | 2 | 14.79 |
| `total_of_special_requests` | 5.41 | 4 | 13.28 | 1 | 9.34 |
| `lead_time` | 8.39 | 3 | 9.29 | 3 | 8.84 |
| `required_car_parking_spaces` | 13.26 | 2 | 1.39 | 22 | 7.33 |
| `CT_Transient` | 4.37 | 8 | 4.39 | 4 | 4.38 |
| `MS_Online TA` | 5.06 | 5 | 3.63 | 7 | 4.34 |
| `adr` | 4.56 | 7 | 4.03 | 5 | 4.29 |
| `arrival_date_year` | 4.56 | 6 | 3.19 | 10 | 3.87 |
| `ART_A` | 4.11 | 9 | 2.69 | 12 | 3.40 |
| `previous_cancellations` | 2.44 | 13 | 3.47 | 9 | 2.96 |

**Key cross-hotel differences:**
- `country_PRT` is the #1 driver for resort and #2 for city.
- `required_car_parking_spaces` is #2 for resort but only #22 for city — a major structural difference reflecting the resort's reliance on car access.
- `total_of_special_requests` is more important for city (#1) than resort (#4).
- `CT_Transient-Party` is #6 for city but #37 for resort.

---

## 9. Results Summary

### 9.1 Main Findings

1. **Model Performance (RQ1/RQ2):**
   - Tree-based models (Random Forest and XGBoost) substantially outperform Logistic Regression across both hotel types.
   - **Resort Hotel:** XGBoost achieves the best AUC (0.9582 without holiday, 0.9581 with holiday).
   - **City Hotel:** Random Forest achieves the best AUC (0.9531 with holiday, 0.9526 without holiday).
   - All ensemble models achieve AUC > 0.94 on the imbalanced test set.

2. **Holiday Feature Impact (RQ4):**
   - The `is_public_holiday` feature has **negligible impact** on model performance.
   - AUC differences between with-holiday and without-holiday models are ≤ 0.0005 for both hotel types.
   - Bootstrap 95% confidence intervals for with/without holiday models **fully overlap**.
   - SHAP analysis ranks `is_public_holiday` at position 43/64 (resort) and 31/64 (city).
   - Chi-squared tests show statistical significance (p < 0.05) but Cramér's V values are below 0.05 (negligible effect size).
   - Holiday window sensitivity analysis confirms negligible effect across all window sizes (±0 to ±7).

3. **Key Cancellation Drivers (RQ3):**
   - **Shared top features:** `country_PRT`, `lead_time`, `total_of_special_requests`, `adr`, `CT_Transient`.
   - **Resort-specific:** `required_car_parking_spaces` is the #2 driver (13.26% of total SHAP), reflecting resort guests' dependence on car access.
   - **City-specific:** `total_of_special_requests` is the #1 driver (13.28%), and `CT_Transient-Party` is notably more important (#6 vs. #37).

4. **Temporal Robustness:**
   - Temporal validation shows a ~10-point AUC drop compared to random-split results, indicating temporal distribution shift between 2015–2016 and 2017 booking patterns.
   - Model rankings are partially preserved under temporal validation.
   - The holiday feature remains negligible under temporal validation.

### 9.2 Model Comparison Summary

| Hotel | Best Model | AUC (No Holiday) | AUC (With Holiday) | Δ AUC |
|---|---|---|---|---|
| Resort | XGBoost | 0.9582 | 0.9581 | −0.0001 |
| City | Random Forest | 0.9526 | 0.9531 | +0.0005 |

### 9.3 Conclusions

- Public holidays, as operationalized in this study (±3-day window around Portuguese national holidays), do not meaningfully improve hotel booking cancellation prediction.
- The holiday effect on cancellation rates is statistically significant but practically negligible (Cramér's V < 0.05).
- Guest origin (`country_PRT`), booking lead time, special requests, and room/parking preferences are far more powerful predictors of cancellation behaviour.
- Different cancellation dynamics exist between resort and city hotels, justifying separate model pipelines.

### 9.4 Practical Implications for Hotel Managers

| Driver | Action |
|---|---|
| **Lead time** (rank 3, both hotels) | Target high-risk bookings made far in advance with early confirmation prompts or flexible rebooking incentives |
| **Special requests** (rank 1 city, rank 4 resort) | Encourage guests to add special requests at booking; guests with more requests are less likely to cancel |
| **Domestic guests** (rank 1 resort, rank 2 city) | Develop tailored retention strategies for Portuguese guests, who show distinct cancellation patterns |
| **Car parking** (rank 2 resort) | For resort hotels, parking reservation confirmations may reinforce guest commitment |
| **Online TA bookings** (rank 5 resort, rank 7 city) | Online travel agency bookings carry specific cancellation risk; consider differentiated deposit policies |
| **Previous cancellations** (rank 9–13) | Flag repeat cancellers for proactive outreach or adjusted overbooking strategies |

---

## 10. Reproducibility Notes

### 10.1 Libraries and Imports

| Library | Purpose |
|---|---|
| `numpy` | Numerical operations |
| `pandas` | Data manipulation |
| `matplotlib` | Visualization |
| `seaborn` | Statistical visualization |
| `scikit-learn` | ML models, preprocessing, evaluation (`train_test_split`, `GridSearchCV`, `StratifiedKFold`, `StandardScaler`, `LogisticRegression`, `RandomForestClassifier`, metrics) |
| `xgboost` | Gradient boosting (`XGBClassifier`) |
| `scipy.stats` | Chi-squared test (`chi2_contingency`) |
| `shap` | SHAP explainability (`TreeExplainer`) |

Specific version numbers are not explicitly recorded in the notebook. Targeted warning filters are used:
```python
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='xgboost')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
```

### 10.2 Random Seeds

A single global random state is used throughout the pipeline:

```python
RANDOM_STATE = 42
```

- `train_test_split` (both random and temporal splits)
- Random undersampling
- Shuffling of balanced training data
- `StratifiedKFold` cross-validation
- All model constructors (`LogisticRegression`, `RandomForestClassifier`, `XGBClassifier`)
- SHAP sample selection (`X_test.sample(random_state=42)`)
- Bootstrap AUC CI (`np.random.default_rng(42)`)

### 10.3 Output Directories

| Directory | Contents |
|---|---|
| `results/` | CSV files: hyperparameters, bootstrap CIs, hotel comparisons, temporal validation, holiday sensitivity, cross-hotel SHAP |
| `figures/` | PNG figures: EDA overview, class balance, confusion matrices, ROC curves, AUC comparison, holiday analysis |
| `shap_outputs/` | SHAP CSV rankings and PNG plots (beeswarm, bar, dependence) per hotel type |

### 10.4 Output Files Manifest

**results/**
- `best_hyperparameters.csv` — Best GridSearchCV parameters for all 12 configurations
- `bootstrap_auc_ci.csv` — Bootstrap 95% CIs for AUC
- `City_comparison.csv` — City hotel metrics (with/without holiday)
- `Resort_comparison.csv` — Resort hotel metrics (with/without holiday)
- `temporal_validation_results.csv` — Temporal split results for all configurations
- `holiday_window_sensitivity.csv` — Sensitivity analysis across window sizes
- `cross_hotel_shap_normalized.csv` — Normalized SHAP comparison across hotel types

**figures/**
- `eda_overview.png` — 2×3 EDA panel
- `class_balance.png` — Class distribution visualization
- `confusion_matrix_best_models.png` — Best model confusion matrices
- `roc_curves_all_models.png` — ROC curves for all models
- `auc_comparison.png` — AUC bar chart for all configurations
- `holiday_analysis.png` — Holiday distribution and impact

**shap_outputs/**
- `Resort_XGBoost_all_shap_ranked.csv` — Full SHAP ranking (resort)
- `Resort_XGBoost_top_shap.csv` — Top 15 SHAP features (resort)
- `Resort_XGBoost_SHAP_beeswarm.png` — Beeswarm plot (resort)
- `Resort_XGBoost_SHAP_bar.png` — Bar plot (resort)
- `Resort_XGBoost_SHAP_dep_*.png` — Dependence plots (resort)
- `City_RandomForest_all_shap_ranked.csv` — Full SHAP ranking (city)
- `City_RandomForest_top_shap.csv` — Top 15 SHAP features (city)
- `City_RandomForest_SHAP_beeswarm.png` — Beeswarm plot (city)
- `City_RandomForest_SHAP_bar.png` — Bar plot (city)
- `City_RandomForest_SHAP_dep_*.png` — Dependence plots (city)

### 10.5 Revisions from Critical Review

Changes made to the analysis in response to review feedback:

| ID | Priority | Description |
|---|---|---|
| CR2 P1 | Added Cramér's V effect size to chi-squared test |
| CR2 P1 | Added PR-AUC (Precision-Recall AUC) metric for imbalanced evaluation |
| CR2 P1 | Reported CV AUC ± standard deviation from GridSearchCV for all models |
| CR2 P1 | Added SHAP normalization (% of total) for valid cross-hotel comparison |
| CR2 P1 | Added holiday window sensitivity analysis (±0, ±1, ±2, ±3, ±4, ±5, ±7) |
| CR2 P1 | Added `holiday_x_domestic` interaction term (`is_public_holiday` × `country_PRT`) |
| CR2 P2 | Acknowledged `assigned_room_type` (ART_*) data leakage limitation |
| CR2 P2 | Acknowledged 0.5 decision threshold limitation; AUC as primary metric |
| CR2 P3 | Removed empty cells at end of notebook |

### 10.6 Acknowledged Limitations

1. **ART_* features (assigned_room_type):** Determined at or near check-in, not at booking time. For canceled bookings that never reach check-in, the assigned room may be a default placeholder. These features may constitute data leakage for booking-time prediction. The study predicts cancellation at any point before arrival, retaining these features. `ART_A` ranks #9 for resort (4.11%) and #12 for city (2.69%). Removal is noted as future work.

2. **Default 0.5 decision threshold:** Models trained on balanced data but evaluated on imbalanced test data. Threshold-dependent metrics (Accuracy, Precision, Recall, F1, Specificity) are affected by this mismatch. AUC and PR-AUC (threshold-free) serve as primary comparison metrics.

3. **`arrival_date_year` as a feature:** Encodes temporal trends (e.g., year-over-year growth), not causal booking-level effects. It ranks #6 for resort (4.56%) and #10 for city (3.19%). It should be excluded from discussions of "actionable" cancellation drivers.

4. **Random undersampling:** Discards substantial majority-class training data. Alternative approaches (class_weight balancing, SMOTE, XGBoost's `scale_pos_weight`) should be explored in future work.
