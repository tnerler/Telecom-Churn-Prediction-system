# Telecom Churn Prediction System



# What does the system do?



- To predict whether the customer will continue to be subscribed with the company or not.



# What is the business objective?



- The business objective is to correctly predict if the customer will leave. If it is, the company can give them a discount for 2 months to not lose the customer.



### Business Value



Assumptions:



- Average monthly subscription revenue: **$20**

- Average customer lifetime: **18 months**

- Average Customer Lifetime Value (CLV): **$360**

- Retention offer: **2 months free service**

- Cost of retention offer: **$40 per targeted customer**



If the model correctly identifies a customer who would otherwise churn, the company can spend **$40** to potentially retain a customer worth **$360**, resulting in a significant positive return on investment.



- **Recall** is the most important metric because we want to identify as many customers who will actually churn as possible. Missing a customer who is about to churn (a **False Negative**) means losing their Customer Lifetime Value (CLV) without having the opportunity to intervene.

- **Precision** is also important because retention offers have a cost. If the model incorrectly predicts that a loyal customer will churn (a **False Positive**), the company may spend money on unnecessary discounts or incentives. Higher precision helps ensure that retention campaigns are targeted at the right customers.



# What results can be acceptable?



## Without a Machine Learning Model



Suppose we have **100 customers**.



- 20 customers will churn.

- 80 customers will remain subscribed.



Business impact:



- Customer Lifetime Value (CLV): **$360**

- Lost revenue from churn:

    - **20 × $360 = $7,200**

- No retention campaign cost because the company cannot identify customers at risk.



**Total loss: $7,200**



---



## With a Machine Learning Model



Suppose the trained model achieves:



- **Recall:** 60%

- **Precision:** 80%



### Step 1: Confusion Matrix



Actual churners = **20**



Recall = 60%



- True Positives (TP) = **12**

- False Negatives (FN) = **8**



Precision = 80%



- False Positives (FP) = **3**

- True Negatives (TN) = **77**



|  | Predicted Churn | Predicted Stay |

| --- | --- | --- |

| **Actual Churn** | 12 | 8 |

| **Actual Stay** | 3 | 77 |



---



### Step 2: Retention Campaign



The company offers a **2-month discount** to every customer predicted to churn.


Campaign size:


- TP + FP = **15 customers**


Campaign cost (because we gave free service to the customers who are predicted to churn for 2 months):


- 15 × $40 = **$600**


---


### Step 3: Business Impact



Customers successfully identified before they churn:



- **12 customers**



Potential revenue preserved:



- 12 × $360 = **$4,320**



Customers missed by the model:



- 8 customers



Revenue still lost:



- 8 × $360 = **$2,880**



Campaign cost:



- **$600**



Total business loss:



- $2,880 + $600 = **$3,480**



---



## Comparison



| Scenario | Total Loss |

| --- | --- |

| Without ML | $7,200 |

| With ML | $3,480 |



**Business improvement**



$7,200 − $3,480 = **$3,720**



The machine learning model reduces the company's expected loss by approximately **52%**, while enabling the business to target retention offers only to customers who are most likely to churn



# Notebook structure
notebooks/<br>
│<br>
├── 01_data_understanding.ipynb<br>
├── 02_exploratory_data_analysis.ipynb<br>
├── 03_data_preprocessing.ipynb<br>
├── 04_feature_engineering.ipynb<br>
├── 05_baseline_model.ipynb<br>
├── 06_model_comparison.ipynb<br>
├── 07_hyperparameter_tuning.ipynb<br>
├── 08_model_evaluation.ipynb<br>
├── 09_business_evaluation.ipynb<br>
└── 10_model_explainability.ipynb<br>

## 01_data_understanding.ipynb
1. Load data
2. Inspect shape
3. Inspect columns
4. Data types
5. Missing values
6. Duplicates
7. Unique values
8. Summary statistics
9. Target distribution
10. Invalid values

--------------------

## 02_exploratory_data_analysis.ipynb
- Univariate analysis
- Target analysis
- Numerical features
- Categorical features
- Correlation analysis
- Outlier analysis
- Relationship with target