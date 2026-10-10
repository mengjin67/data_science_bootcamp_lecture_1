# Datathon 2025

## Business problem

You are a junior data scientist at **FutureBright Insurance**. The underwriting department of the Automobile business line has requested your team's assistance in building a risk segmentation model using historical auto policy exposure and claims data. The goals of this project are threefold:

1. **Quantify expected losses** — and thus the risk level — for potential customers.
2. **Support the underwriting and actuarial departments** by providing insights into the nature of the business, such as:
   - Identify which segments of the portfolio are underpriced or overpriced.
   - Offer strategies to adjust rates accordingly.
3. **Assist the marketing department** in designing targeted campaigns to engage prospective customers.

## Data

Your team has received two data sets. Each row is one policy term. Neither file has missing values.

- **Model data:** exposure and claims information for 15,000 historical policy terms (24 columns). [Download](https://drive.google.com/file/d/1Wvf4WmgOVFLu3snFkIUYI6Y5unM9b5o1/view?usp=drive_link)
- **Inference data:** information for 15,000 future potential customers (19 columns). [Download](https://drive.google.com/file/d/17j2pkUwyUdOWg3-2xKQReL2JVHtSbGSN/view?usp=drive_link)
- **Data dictionary:** definitions and values of every column. [Open](https://drive.google.com/file/d/1-6HNERbWr7C5NgmOW31AzyNYKDOmZ1wY/view?usp=sharing)

To download the data in code, see the download cell in `analysis_pipeline/1_data_etl.ipynb`. Setup steps are in `repo_setup_instructions.md`.

### Claim fields (model data only)

Claim outcomes. They are not available for future customers, so they are not in the inference data.

- `claimcst0` - total claim amount ($) in the policy term, summed over all claims; 0 if there was no claim. **This is our target.**
- `numclaims` - number of claims in the term (0, 1 or 2+).
- `clm` - claim indicator: 1 if the term had at least one claim (`numclaims >= 1`), otherwise 0.

### Exposure

Policy terms differ in how long the policy was in force. `exposure` is the share of a year the policy was in force (between 0 and 1, never more than the term length). A longer exposure means more time for claims to happen, so compare claim amounts **per unit of exposure**, not per term. The notebooks model capped claim amount per exposure.

### Other columns

- `sample`, `fold` (model data only): `sample` is `1|bld` for build (training) rows and `2|val` for validation rows; `fold` is 1-3 for build rows and 4 for validation rows. For cross-validation, use folds 1-3 and leave fold 4 untouched.
- All other columns are predictors present in both files; see the data dictionary above for their definitions.

## Modeling

After reviewing the data, your manager - a lead data scientist - decides to develop a machine learning model to predict the claim cost per policy term (`claimcst0`), adjusted for exposure. You are asked to follow standard modeling steps on this effort starting from model design, data collection and cleaning, data exploration, model selection and model implementation.

In addition, your manager encourages initiative and experimentation, and recommends testing the following techniques:

#### 1. Variable reduction

Variable reduction is often a pain point in modeling, particularly in the era of big data. In previous roles, your manager used a principal component based hierarchical variable reduction technique in SAS called `VARCLUS` ([support.sas.com/documentation/onlinedoc/stat/132/varclus.pdf](https://support.sas.com/documentation/onlinedoc/stat/132/varclus.pdf)) to conduct efficient variable reduction. Recently, he found a Python package implementing a similar approach - `VarClusHi` ([jingtt/varclushi: A Python package for variable clustering](https://github.com/jingtt/varclushi?tab=readme-ov-file)), which he would like you to try.

For an introduction of this method, refer to this notebook: [2_data_exploration.ipynb](analysis_pipeline/2_data_exploration.ipynb).

You are also welcome to explore alternative methods, such as penalized regression. Regardless of the approach, your final delivery should include a detailed description of the method used on variable reduction.

#### 2. Tree-based models & Hyperparameter Tuning

Tree-based models (Gradient Boosting Machines and Random Forests) are widely used in finance and insurance due to their high predictive accuracy. In this project, your manager decides to use the tree-based model as the primary modeling technique. However, there are many hyperparameters in tree-based models.

Your manager suggests moving away from exhaustive grid/random search due to its inefficiency. Instead, you are encouraged to consider an interactive hyperparameter tuning approach - tuning two or three hyperparameters at a time - which allows for both human insight and better computational efficiency.

Tutorials on this topic:

- [interactive hyperparameter tuning blog 1](https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/).
- [interactive hyperparameter tuning blog 2](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/). Refer to [3_model_selection.ipynb](analysis_pipeline/3_model_selection.ipynb) for more details.

Whatever method you choose, please include a detailed description of the hyperparameters and tuning strategy.

#### 3. Foundation Models for Tabular Data

Following the success of large language models, transformer architectures are now also used on tabular data. A 2025 article in _Nature_ titled "Accurate predictions on small data with a tabular foundation model" presents `TabPFN`, a transformer-based model that claims superior performance on small to medium size tabular datasets.

In this project, you may explore and compare `TabPFN`'s performance against the tree-based models built.

Resources:

- `TabPFN` *Nature* article (2025): [https://www.nature.com/articles/s41586-024-08328-6](https://www.nature.com/articles/s41586-024-08328-6).
- `TabPFN` paper (original 2022 version): [https://arxiv.org/abs/2207.01848](https://arxiv.org/abs/2207.01848).
- `TabPFN` GitHub repository: [https://github.com/PriorLabs/TabPFN](https://github.com/PriorLabs/TabPFN).

Refer to the end of [3_model_selection.ipynb](analysis_pipeline/3_model_selection.ipynb) for some more notes.

If you choose to use `TabPFN`, please include a detailed explanation of this method, along with its strengths and limitations, in the final delivery.

#### 4. Frequency-Severity Modeling & Exposure Handling

Estimating claim cost per policy term is complex due to two main challenges:

- The total claim cost is a product of frequency and severity.
- Policy terms have varying exposure durations.

To address these:

- Use a composite modeling approach: build separate models for frequency and severity, then combine their predictions.
- Account for variable exposures using weighted likelihood estimation or offset techniques.

For more details, refer to [3_model_selection.ipynb](analysis_pipeline/3_model_selection.ipynb).

If you explore any of these methodologies, be sure to include detailed documentation of your approach.

#### 5. Model Explainability

Machine learning models are often criticized as "black box" due to their lack of interpretability. One powerful method for explainability is `SHAP` (Shapley Additive Explanations), which quantifies the contribution of each feature to individual predictions as well as the overall model.

Resources:

- `SHAP` paper: [https://arxiv.org/abs/1705.07874](https://arxiv.org/abs/1705.07874).
- [Explaining XGBoost model predictions with SHAP values — SAMueL - Stroke Audit Machine Learning](https://samuel-book.github.io/samuel_shap_paper_1/xgb_10_features/03_xgb_combined_shap_key_features.html).

Please use `SHAP`, or similar tools, to explain your model results - this will be particularly helpful for addressing business questions.

## Final Delivery

Your final delivery should include two parts, with the focus on part 1.

1. Internal review (Technical Audience)
   - Target: Technical Team - technical product manager, technical lead, data scientists and developers in general.
   - Purpose: To describe the modeling procedures and techniques used at each stage, with particular emphasis on data exploration and model selection. Include both qualitative discussion and quantitative results from your analysis.
   - Note:
     - In the model data, the build and validation samples can be filtered using the `sample` column. Please use the rows where `sample=='1|bld'` (folds 1-3) to train the model, and the rows where `sample=='2|val'` (fold 4) to evaluate or test the model performance. Please avoid over-using the validation samples.
     - You do not need to include every modeling technique discussed in the Modeling section above, please decide based on your interest and availability, and the fitness of the techniques to the data.
2. External review (Business Audience)
   - Target: Stakeholders (underwriters, actuaries, sales team)
   - Purpose: To address the original business questions by providing clear and actionable recommendations on prospective customers based on the inference data.

## Note

The data used in this datathon is synthetic, derived from the dataset used in the Kaggle contest: [2023 UMN Travelers Analytics Case Competition](https://www.kaggle.com/competitions/2023-umn-travelers-analytics-case-competition/data).
