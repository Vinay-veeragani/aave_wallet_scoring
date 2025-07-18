Credit Score Distribution Analysis

Dataset Overview

The dataset contains wallet-based credit scores generated via a machine learning model, with values ranging from 0 to 1000. The goal of this analysis is to explore the distribution of scores and gain insights into the model’s behavior.

1. Distribution Summary

Minimum score: 0

Maximum score: 1000

Mean score: ~70.8

Median score: ~0.5

Interpretation

The mean is much higher than the median, indicating a strong right-skewed distribution.

The distribution is bimodal with high density near both ends: a large number of scores near 0 and a second cluster near 1000.

2. Outliers and Clusters

Lower-end Cluster:

~2700 wallets scored extremely low (close to 0).

Indicates a significant population of wallets flagged as high-risk.

Upper-end Cluster:

~700 wallets score nearly perfect (close to 1000).

These may be high-trust wallets or white-listed accounts.

Mid-range Scarcity:

Very few wallets have scores between 300–900.

This lack of mid-range distribution may imply that the ML model used is overconfident or classifies accounts as extremes.

3. Model Evaluation Implication

The distribution suggests a need to revisit the classification threshold and scoring logic.

Potential bias may exist in the training data or in feature weighting.

The model likely segments users into binary classes (good vs bad) rather than a continuous score.

4. Recommendations

Perform further feature importance analysis.

Rebalance training data to improve score spread.

Introduce score smoothing techniques to reduce polarization.

