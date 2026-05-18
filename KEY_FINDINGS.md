# Key Findings — SHAP & ML Model Analysis

**Study:** Explainable Machine Learning for Hotel Booking Cancellations: The Role of Public Holidays  
**Author:** Nurbek Suvonov | Kookmin University | 2026

---

## 1. ML Model Performance Overview

### 1.1 Best Models by Hotel Type

| Hotel Type | Best Model | AUC (Without Holiday) | AUC (With Holiday) | ΔAUC |
|---|---|---|---|---|
| **Resort Hotel** | **XGBoost** | **0.9582** | **0.9581** | −0.0001 |
| **City Hotel** | **Random Forest** | **0.9526** | **0.9531** | +0.0005 |

> **Key Finding:** Tree-based ensemble models (Random Forest and XGBoost) significantly outperform Logistic Regression for both hotel types, achieving AUC > 0.94 across all configurations. The public holiday feature has a **negligible effect** on predictive performance (ΔAUC ≤ 0.0005).

### 1.2 Full Model Comparison — Resort Hotel

| Model | Feature Set | Accuracy | Recall | Precision | F1 Score | AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| Logistic Regression | Without Holiday | 0.8263 | 0.8611 | 0.6390 | 0.7336 | 0.9098 | 0.7940 |
| Logistic Regression | With Holiday | 0.8267 | 0.8611 | 0.6396 | 0.7340 | 0.9098 | 0.7941 |
| Random Forest | Without Holiday | 0.8841 | 0.9033 | 0.7381 | 0.8124 | 0.9572 | 0.9044 |
| Random Forest | With Holiday | 0.8818 | 0.9020 | 0.7335 | 0.8090 | 0.9576 | 0.9051 |
| **XGBoost** | **Without Holiday** | **0.8799** | **0.9087** | **0.7270** | **0.8078** | **0.9582** | **0.9022** |
| **XGBoost** | **With Holiday** | **0.8811** | **0.9096** | **0.7293** | **0.8095** | **0.9581** | **0.9020** |

**Resort takeaways:**
- XGBoost is the top performer with AUC = 0.9582 — the highest score across all configurations.
- Ensemble models achieve ~5 percentage points higher AUC than Logistic Regression.
- XGBoost captures 90.9% of actual cancellations (Recall), making it highly sensitive.
- The holiday feature changes AUC by only 0.0001 — statistically and practically insignificant.

### 1.3 Full Model Comparison — City Hotel

| Model | Feature Set | Accuracy | Recall | Precision | F1 Score | AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| Logistic Regression | Without Holiday | 0.7931 | 0.7775 | 0.7405 | 0.7586 | 0.8779 | 0.8544 |
| Logistic Regression | With Holiday | 0.7930 | 0.7775 | 0.7403 | 0.7585 | 0.8780 | 0.8545 |
| **Random Forest** | **Without Holiday** | **0.8783** | **0.8513** | **0.8567** | **0.8540** | **0.9526** | **0.9444** |
| **Random Forest** | **With Holiday** | **0.8814** | **0.8535** | **0.8616** | **0.8575** | **0.9531** | **0.9448** |
| XGBoost | Without Holiday | 0.8702 | 0.8610 | 0.8339 | 0.8472 | 0.9478 | 0.9381 |
| XGBoost | With Holiday | 0.8705 | 0.8598 | 0.8353 | 0.8474 | 0.9478 | 0.9380 |

**City takeaways:**
- Random Forest is the top performer with AUC = 0.9531, slightly edging out XGBoost (0.9478).
- Random Forest achieves the best balance of Precision (0.8616) and Recall (0.8535), yielding the highest F1 = 0.8575.
- Ensemble models outperform Logistic Regression by ~7.5 percentage points in AUC.
- Adding the holiday feature improves AUC by only 0.0005 — negligible.

### 1.4 Statistical Significance — Bootstrap 95% Confidence Intervals

| Configuration | AUC (Mean) | 95% CI Lower | 95% CI Upper | CI Width |
|---|---|---|---|---|
| Resort XGBoost — No Holiday | 0.9583 | 0.9544 | 0.9623 | 0.0079 |
| Resort XGBoost — With Holiday | 0.9581 | 0.9543 | 0.9621 | 0.0078 |
| City Random Forest — No Holiday | 0.9526 | 0.9496 | 0.9554 | 0.0058 |
| City Random Forest — With Holiday | 0.9531 | 0.9502 | 0.9559 | 0.0057 |


---

## 2. Temporal Robustness Validation

Models were also evaluated using a temporal split (train: 2015–2016, test: 2017) to assess real-world generalization.

| Model | Hotel | AUC (No Holiday) | AUC (With Holiday) | Random Split AUC | AUC Drop |
|---|---|---|---|---|---|
| XGBoost | Resort | 0.8554 | 0.8498 | 0.9582 | **−10.3 pp (percentage points)** |
| Random Forest | Resort | 0.8376 | 0.8509 | 0.9572 | **−11.9 pp** |
| Random Forest | City | 0.8455 | 0.8437 | 0.9526 | **−10.7 pp** |
| XGBoost | City | 0.8455 | 0.8490 | 0.9478 | **−10.2 pp** |
| Logistic Regression | Resort | 0.8615 | 0.8612 | 0.9098 | **−4.8 pp** |
| Logistic Regression | City | 0.8411 | 0.8410 | 0.8779 | **−3.7 pp** |

> **Key Finding:** A ~10-point AUC drop under temporal validation indicates **temporal distribution shift** — booking patterns in 2017 differ from 2015–2016. However, all models still achieve AUC > 0.83, demonstrating reasonable generalizability. The holiday feature remains negligible under temporal validation as well.

---

## 3. SHAP Explainability Findings

### 3.1 Top 15 Features — Resort Hotel (XGBoost)

| Rank | Feature | Mean SHAP | % of Total | Interpretation |
|---:|---|---:|---:|---|
| 1 | `country_PRT` | 1.3108 | 16.73% | Portuguese guests have distinct cancellation behaviour |
| 2 | `required_car_parking_spaces` | 1.0393 | 13.26% | Car parking demand signals commitment to stay |
| 3 | `lead_time` | 0.6577 | 8.39% | Longer lead times → higher cancellation risk |
| 4 | `total_of_special_requests` | 0.4238 | 5.41% | More special requests → lower cancellation risk |
| 5 | `MS_Online TA` | 0.3962 | 5.06% | Online travel agency bookings show specific patterns |
| 6 | `arrival_date_year` | 0.3576 | 4.56% | Captures year-over-year temporal trends |
| 7 | `adr` | 0.3570 | 4.56% | Average daily rate influences cancellation decisions |
| 8 | `CT_Transient` | 0.3422 | 4.37% | Transient customer type has unique cancellation dynamics |
| 9 | `ART_A` | 0.3221 | 4.11% | Assigned room type A is a significant factor |
| 10 | `MS_Offline TA/TO` | 0.2688 | 3.43% | Offline travel agency/tour operator bookings |
| 11 | `RRT_A` | 0.2309 | 2.95% | Reserved room type A |
| 12 | `arrival_date_week_number` | 0.2223 | 2.84% | Weekly seasonality effects |
| 13 | `previous_cancellations` | 0.1912 | 2.44% | Prior cancellation history predicts future behaviour |
| 14 | `booking_changes` | 0.1728 | 2.20% | Number of booking modifications |
| 15 | `arrival_date_day_of_month` | 0.1351 | 1.72% | Day-of-month arrival effects |

### 3.2 Top 15 Features — City Hotel (Random Forest)

| Rank | Feature | Mean SHAP | % of Total | Interpretation |
|---:|---|---:|---:|---|
| 1 | `total_of_special_requests` | 0.1079 | 13.28% | Strongest single predictor — signals guest commitment |
| 2 | `country_PRT` | 0.1045 | 12.86% | Domestic guests show distinct cancellation patterns |
| 3 | `lead_time` | 0.0755 | 9.29% | Longer lead time → higher cancellation probability |
| 4 | `CT_Transient` | 0.0357 | 4.39% | Transient bookings are more cancellation-prone |
| 5 | `adr` | 0.0327 | 4.03% | Price sensitivity impacts cancellation decisions |
| 6 | `CT_Transient-Party` | 0.0309 | 3.80% | Group transient bookings show unique patterns |
| 7 | `MS_Online TA` | 0.0295 | 3.63% | Online travel agency channel effects |
| 8 | `booking_changes` | 0.0283 | 3.48% | Modifications correlate with cancellation intent |
| 9 | `previous_cancellations` | 0.0282 | 3.47% | Guest cancellation history is predictive |
| 10 | `arrival_date_year` | 0.0259 | 3.19% | Year-over-year trends in cancellation rates |
| 11 | `MS_Groups` | 0.0219 | 2.69% | Group market segment bookings |
| 12 | `ART_A` | 0.0218 | 2.69% | Room assignment effects |
| 13 | `arrival_date_week_number` | 0.0195 | 2.40% | Seasonal variation in cancellations |
| 14 | `MS_Offline TA/TO` | 0.0164 | 2.01% | Offline travel agency/tour operator |
| 15 | `arrival_date_day_of_month` | 0.0163 | 2.01% | Day-of-month patterns |

### 3.3 Cross-Hotel SHAP Comparison (Top 10 Normalized)

| Feature | Resort % | Resort Rank | City % | City Rank | Average % |
|---|---|---|---|---|---|
| `country_PRT` | 16.73 | 1 | 12.86 | 2 | **14.79** |
| `total_of_special_requests` | 5.41 | 4 | 13.28 | 1 | **9.34** |
| `lead_time` | 8.39 | 3 | 9.29 | 3 | **8.84** |
| `required_car_parking_spaces` | 13.26 | 2 | 1.39 | 22 | **7.33** |
| `CT_Transient` | 4.37 | 8 | 4.39 | 4 | **4.38** |
| `MS_Online TA` | 5.06 | 5 | 3.63 | 7 | **4.34** |
| `adr` | 4.56 | 7 | 4.03 | 5 | **4.29** |
| `arrival_date_year` | 4.56 | 6 | 3.19 | 10 | **3.87** |
| `ART_A` | 4.11 | 9 | 2.69 | 12 | **3.40** |
| `previous_cancellations` | 2.44 | 13 | 3.47 | 9 | **2.96** |

### 3.4 Structural Differences Between Hotel Types

| Aspect | Resort Hotel | City Hotel |
|---|---|---|
| **#1 Driver** | `country_PRT` (16.73%) | `total_of_special_requests` (13.28%) |
| **#2 Driver** | `required_car_parking_spaces` (13.26%) | `country_PRT` (12.86%) |
| **Unique to Top 5** | `required_car_parking_spaces` | `CT_Transient` |
| **Biggest rank gap** | `required_car_parking_spaces`: Rank 2 (Resort) → Rank 22 (City) | `CT_Transient-Party`: Rank 6 (City) → Rank 37 (Resort) |
| **Best model** | XGBoost | Random Forest |
| **Cancellation rate** | ~27.8% | ~41.3% |

> **Key Finding:** Resort and city hotels have **fundamentally different cancellation dynamics**. Resort cancellations are heavily driven by guest origin and car parking needs (reflecting remote locations requiring car access), while city cancellations are primarily driven by special requests and guest origin. This justifies building **separate model pipelines** for each hotel type.

---

## 4. Public Holiday Feature — Detailed Assessment

### 4.1 SHAP Ranking of Holiday Features

| Feature | Resort Rank | Resort % | City Rank | City % |
|---|---|---|---|---|
| `is_public_holiday` | **43 out of 64** | 0.22% | **31 out of 64** | 0.56% |
| `holiday_x_domestic` | 38 out of 64 | 0.34% | 21 out of 64 | 1.51% |

> **Key Finding:** The `is_public_holiday` feature ranks in the **bottom half** of all features for both hotel types. It contributes less than 1% of total SHAP importance. Even the interaction term `holiday_x_domestic` (holiday × Portuguese guest) remains a minor predictor.

### 4.2 Holiday Effect on Cancellation Rates

| Window | Hotel | % Bookings Flagged | Cancel Rate (Regular) | Cancel Rate (Holiday) | Difference | Cramér's V |
|---|---|---|---|---|---|---|
| ±3 days | Resort | 23.4% | 27.04% | 30.11% | +3.07 pp | 0.029 |
| ±3 days | City | 23.4% | 40.74% | 45.27% | +4.53 pp | 0.039 |

> **Key Finding:** While cancellation rates are slightly higher during holiday windows (3–5 percentage points), the effect size is **negligible** (Cramér's V < 0.05). The statistical significance arises purely from large sample sizes (n > 40,000), not from practical importance.

### 4.3 Holiday Window Sensitivity

| Window Size | Resort Cramér's V | City Cramér's V |
|---|---|---|
| ±0 (exact day) | 0.016 | 0.022 |
| ±1 day | 0.024 | 0.023 |
| ±2 days | 0.028 | 0.040 |
| **±3 days** | **0.029** | **0.039** |
| ±4 days | 0.028 | 0.038 |
| ±5 days | 0.028 | 0.040 |
| ±7 days | 0.011 | 0.042 |

> All Cramér's V values remain below 0.05 across all window sizes — confirming that the negligible effect is **robust** and not an artifact of window choice.

---

## 5. Core Insights & Practical Implications

### 5.1 Five Most Important Findings

1. **Tree-based models excel at cancellation prediction.** Both XGBoost (Resort AUC = 0.958) and Random Forest (City AUC = 0.953) deliver high discriminative performance, significantly outperforming Logistic Regression by 5–8 AUC points.

2. **Public holidays do not meaningfully predict cancellations.** Despite statistical significance (due to large n), the holiday feature contributes <1% of SHAP importance and changes model AUC by <0.001. This is the central null finding of the study.

3. **Guest origin is the single most powerful predictor.** `country_PRT` (Portuguese domestic guests) is the #1 feature for resort hotels and #2 for city hotels, accounting for 13–17% of total SHAP importance.

4. **Hotel types require separate models.** The dramatic difference in feature importance (e.g., `required_car_parking_spaces` at Rank 2 for resort vs. Rank 22 for city) demonstrates that a one-size-fits-all model is suboptimal.

5. **Temporal drift is a real concern.** The ~10-point AUC drop under temporal validation highlights that cancellation patterns evolve over time, and models need periodic retraining.

### 5.2 Actionable Recommendations for Hotel Managers

| Driver | Action |
|---|---|
| **Lead time** (Rank 3 both hotels) | Target high-risk bookings made far in advance with early confirmation prompts or flexible rebooking incentives |
| **Special requests** (Rank 1 city, Rank 4 resort) | Encourage guests to add special requests at booking; guests with more requests are less likely to cancel |
| **Domestic guests** (Rank 1 resort, Rank 2 city) | Develop tailored retention strategies for Portuguese guests, who show distinct cancellation patterns |
| **Car parking** (Rank 2 resort) | For resort hotels, parking reservation confirmations may reinforce guest commitment |
| **Online TA bookings** (Rank 5 resort, Rank 7 city) | Online travel agency bookings carry specific cancellation risk; consider differentiated deposit policies |
| **Previous cancellations** (Rank 9–13) | Flag repeat cancellers for proactive outreach or adjusted overbooking strategies |

### 5.3 Methodological Contributions

- **Dual validation strategy:** Both random-split and temporal-split validation provide complementary confidence in results.
- **Bootstrap confidence intervals:** Formal statistical test confirming the null holiday effect.
- **Normalized SHAP comparison:** Enables valid cross-model, cross-hotel feature importance comparison despite different model types and SHAP scales.

---

## 6. Summary Table — Answers to Research Questions

| RQ | Question | Answer |
|---|---|---|
| **RQ1** | How accurately can LR, RF, and XGBoost predict cancellations? | All ensemble models achieve AUC > 0.94. LR achieves 0.88–0.91. Tree-based models outperform LR by 5–8 AUC points. |
| **RQ2** | Which model is best per hotel type? | **XGBoost** for Resort (AUC = 0.958); **Random Forest** for City (AUC = 0.953). |
| **RQ3** | What are the key cancellation drivers? | `country_PRT`, `lead_time`, `total_of_special_requests`, `required_car_parking_spaces` (resort), `adr`, and `CT_Transient` are the dominant drivers. |

---

## Notes & abbreviations

- SHAP values: absolute per-model SHAP values were normalized to percent contribution (so % of Total sums to 100% per model) to enable fair cross-model comparisons.
- Abbreviations used in tables: `country_PRT` = booking from Portugal; `MS_Online TA` = market segment: Online Travel Agency; `MS_Offline TA/TO` = Offline Travel Agency / Tour Operator; `CT_Transient` = customer type: transient; `ART_A`, `RRT_A` = assigned/reserved room type A; `adr` = average daily rate.
- Metrics: AUC refers to ROC-AUC (higher is better); PR-AUC denotes average precision.
- "pp" denotes percentage points (e.g., a drop from 0.95 to 0.85 is −10.0 pp).
- Reproducibility: Results were generated from `analysis.ipynb`; see `requirements.txt` for package versions.
