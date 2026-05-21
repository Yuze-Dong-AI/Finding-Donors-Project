
# Supervised Learning Project: Finding Donors for CharityML

## 📌 Project Overview
CharityML is a U.S.-based non-profit organization that provides educational training in next-generation technologies. To fuel their expansion programs, they rely heavily on targeted donation campaigns. Rather than executing broad, resource-heavy marketing sweeps across entire populations, CharityML aims to construct an automated predictive pipeline to target only high-probability prospects.

Socio-economic research indicates that individuals earning **more than $50,000 annually** represent the most viable donor candidate group. This project implements, benchmarks, and optimizes multiple supervised learning algorithms utilizing U.S. Census Bureau records to deliver a highly precise, cost-effective donor targeting engine.

---

## 📂 Repository Architecture
- `finding_donors.ipynb`: Primary interactive workspace covering exploratory analysis, skewed log-transformations, pipeline benchmarking, grid tuning, and dimensionality reduction profiling.
- `census.csv`: Main demographic training dataset containing financial metrics, education factors, and baseline target flags.
- `test_census.csv`: Holdout testing vector utilized to measure validation accuracy and out-of-sample generalization.
- `example_submission.csv`: Template mapping structural output syntax requirements.
- `visuals.py`: Embedded Python graphics script mapping out featureset skew distributions, algorithm comparison metrics, and feature relevance thresholds.

---

## 🛠 Tech Stack & Environment Setup
This project runs on **Python 3.x** and relies on the core PyData ecosystem:
- **NumPy** & **Pandas** (Data matrix ingestion, serialization, and series alignment)
- **Matplotlib** (Data distribution plots)
- **scikit-learn** (Model evaluation loops, preprocessors, and `GridSearchCV`)

To initialize the project container locally, execute:
```bash
jupyter notebook finding_donors.ipynb

```

---

## ⚙️ Data Preprocessing & Feature Engineering

Before feeding raw census vectors to the classifiers, the data was processed through a 3-stage feature pipeline:

* **Logarithmic Scaling:** Continuous fields like `capital-gain` and `capital-loss` displayed severe right-hand skews. A natural log transform ($x \rightarrow \ln(x + 1)$) was applied to stabilize variance and minimize outlier influence.
* **MinMax Normalization:** Continuous dimensions were bound cleanly between `0` and `1` using a `MinMaxScaler` to guarantee that scale variances (such as `hours-per-week` vs `age`) did not improperly weight gradient and distance calculations.
* **One-Hot Encoding:** Categorical attributes (e.g., `occupation`, `marital-status`) were converted to binary flags, expanding the original feature footprint to 103 input parameters.

---

## 📊 Optimization Strategy & Success Metrics

Because only 24.78% of the data encompasses individuals earning over $50K, baseline classification accuracy is a deceptive metric. CharityML’s primary financial constraint is **minimizing False Positives** (wasting precious postage/outreach capital on non-viable candidates) while maintaining a healthy acquisition funnel.

To optimize explicitly for this constraint, the **$F_{\beta}$ score** with a beta factor of **$\beta = 0.5$** was selected:

$$F_{0.5} = (1 + 0.5^2) \times \frac{\text{Precision} \times \text{Recall}}{(0.5^2 \times \text{Precision}) + \text{Recall}}$$

By giving **Precision double the mathematical weight of Recall**, the model guarantees outbound budget safety.

---

## 📈 Final Project Conclusion & Experimental Analysis


### 1. Classifier Selection Synthesis

Three core supervised architectures—a Decision Tree, a Random Forest, and an **AdaBoost Ensemble Classifier**—were rigorously evaluated against a baseline Naive Predictor. Following cross-validated hyperparameter tuning via `GridSearchCV`, **AdaBoost** was deployed as the production model.

| Metric | Naive Predictor Benchmark | Unoptimized AdaBoost | Optimized Production Model |
| --- | --- | --- | --- |
| **Accuracy Score** | 0.2478 (24.78%) | 0.8483 (84.83%) | **0.8568 (85.68%)** |
| **$F_{0.5}$ Score** | 0.2917 | 0.7029 | **0.7223** |

* **The Breakthrough:** While a Naive Baseline model guessing that *everyone* earns over $50K scores a low $F_{0.5}$ of `0.2917`, the fine-tuned AdaBoost architecture achieved a definitive **`0.7223`**.
* **Generalization Strength:** AdaBoost exhibited the lowest train-to-test overfitting gap among all candidates. By sequentially training weak learners (decision stubs) that adaptively focus on the classification errors of their predecessors, it forms a robust consolidated voting team highly resilient to unseen data patterns.



### 2. The Predictive Blueprint: Top 5 Earning Predictors

Invoking the `.feature_importances_` attribute of the optimized AdaBoost model mapped the top 5 demographic attributes with the highest statistical leverage:

* **`capital-loss` / `capital-gain`:** Direct continuous measures of investment activity and asset scale. High investment activity directly reveals high disposable, non-wage wealth capacity.
* **`age`:** Directly tracks professional seniority, tenure cycles, and the accumulation of compounding career wage increments over time.
* **`hours-per-week`:** Explicitly separates full-time enterprise professionals and overtime workers from part-time, seasonal, or underemployed assets.
* **`education-num`:** Measures cumulative institutional years of study, validating the direct link between academic milestones and higher corporate salary bands.

> **💡 Analytical Insight:** Standard qualitative assumptions frequently assume that explicit job titles (`occupation`) are the strongest indicators. However, the model proved that continuous numerical financial indicators (`capital-gain`/`capital-loss`) hold higher statistical weight because they track actual, hard financial behaviors rather than generic categorical proxies.




### 3. Engineering Trade-off: Dimensionality vs. Operational Latency

To evaluate production efficiency, an experiment was conducted by retraining the optimized AdaBoost model using **only the top 5 primary features** listed above, completely throwing out the other 98 dimensions created during one-hot encoding:

* **Full Featureset Model (103 features):** Accuracy: `0.8568` | $F_{0.5}$ Score: `0.7223`
* **Reduced Featureset Model (5 features):** Accuracy: `0.8456` | $F_{0.5}$ Score: `0.6928`

---



## 💡 Strategic Deployment Recommendation

* **Deploy the Full-Feature Model** if CharityML is executing static, lower-volume localized physical mailing drives. When printing and postage costs represent strict financial limits, capturing every fraction of a percent of precision is paramount to directly protect physical budget limits.
* **Deploy the Reduced 5-Feature Model** if CharityML transitions into a real-time digital cloud pipeline, user-profile logging stream, or online web dashboard handling millions of continuous entries per second. The minor 3 percentage point drop in the $F_{0.5}$ score is heavily offset by near-instantaneous data processing loops, minimal memory allocation boundaries, and massive savings in cloud infrastructure compute overhead.

