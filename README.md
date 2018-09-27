# Source code for Imputing Missing Value via CDMF
Missing data problem often poses a significant challenge as it may introduce uncertainties into the data analysis. Recent advances in matrix completion have shown competitive imputation performance when applied to many real-world domains. However, there are two major limitations when applying matrix completion methods to spatial data. First, they make a strong assumption that the entries are missing-at-random, which may not hold for spatial data. Second, they may not effectively utilize the underlying spatial structure of the data. To address these limitations, this work presents a novel clustered adversarial matrix factorization method to explore and exploit the underlying cluster structure of the spatial data in order to facilitate effective imputation. The proposed method utilizes an adversarial network to learn the joint probability distribution of the variables and improve the imputation performance for the missing entries that are not randomly sampled.
## Usage
`example.py` is an example of imputing missing value with CAMF. In the function 'load_data', please load the data files as follows: 

- `X_full`: the full data (used to compute RMSE)

- `X_missing_data`: the part that has missing value

- `X_notmissing_data`: the part that has missing values

- `wieght`: weight vector

- `index1`: i in d_{ij}

- `index2`: j in i in d_{ij} 


## Citation

As you use this code for your exciting discoveries, please cite the paper below:

> Qi Wang, Pang-Ning Tan, and Jiayu Zhou. Imputing Structured Missing Values in Spatial Data with Clustered Adversarial Matrix Factorization." In Data Mining (ICDM), 2018 IEEE 18th International Conference on, IEEE, 2018
