# Overview
## Introduction
This program is designed to calculate how similar a set of synthetic data is to the real data is was synthesized off of. The purpose of this is to be able to find if the data generated is actually new and unique or if it copies off the real data. This can be a huge concern for privacy reasons.
## Dataset Overview
The dataset itself contains 36 different columns. The data is present in all numerical variables with some being used to represent booleans. For more information on the dataset, go to see the
[Kaggle Dataset](https://www.kaggle.com/datasets/loveall/cervical-cancer-risk-classification?resource=download).

## Data Loading
The data is loaded into python using pandas and imported in as a pandas dataframe.
``` Python
import pandas as pd
syndata = pd.read_csv('synData9.csv')
realdata = pd.read_csv('kag_risk_factors_cervical_cancer.csv')
```
The data has some missing data and the dtypes are objects for some columns where we need them to be numbers. Before we can use the data further, we'll need to do some data cleaning to get the data into a workable format.

## Data Cleaning
To get over the issues with the dataset, we first start off by changing all the missing values to -1 and then go over and convert the dataframe to all floats. This ensuresd the dataframe will be able to be used by the code later on.

We could also simply remove rows with missing data but in this project, it would end up leaving us with too few rows. 
``` Python
def clean(data1):
    """
    Replaces the missing data with a -1 value and converts the strings into floats
    """
    for i in range(len(data1)):
        for f in data1.columns:
            if data1.loc[i, f] == "?":
                data1.loc[i, f] = float(-1)
                continue
            if isinstance(data1.loc[i, f], str):
                data1.loc[i, f] = float(data1.loc[i, f])
    data1 =data1.set_axis([f for f in range(len(data1))],   axis='index')
    return data1.astype('float')
```
From there, we need to normalize the data and in order to do that, we use min-max normalization
``` Py
def norm(data1):
    """
    Min-Max normalizes the data
    """
    new_dat = data1.copy()
    for col in new_dat.columns:
        new_dat[col] = (new_dat[col] - new_dat[col].min()) / (new_dat[col].max() - new_dat[col].min())
    return new_dat
```
Using this, we end up with a finalized dataframe full of cleaned and normalized data that we can use for the rest of the project.
## Data Weighting
Since the data will have different importance based on what column it's in, we add a weight factor to each column. Currently, the way this works is it assigns each column a high, medium or low weight.
``` Py
#Init
weights =[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

high_weights =['Age', 'Number of sexual partners', 'First sexual intercourse',
       'Num of pregnancies', 'Smokes']
med_weights =['Smokes (years)', 'Smokes (packs/year)',
       'Hormonal Contraceptives', 'Hormonal Contraceptives (years)', 'IUD',
       'IUD (years)', 'STDs', 'STDs (number)', 'STDs:condylomatosis',
       'STDs:cervical condylomatosis']
low_weights =['STDs:vaginal condylomatosis',
       'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis',
       'STDs:pelvic inflammatory disease', 'STDs:genital herpes',
       'STDs:molluscum contagiosum', 'STDs:AIDS', 'STDs:HIV',
       'STDs:Hepatitis B', 'STDs:HPV', 'STDs: Number of diagnosis',
       'STDs: Time since first diagnosis', 'STDs: Time since last diagnosis',
       'Dx:Cancer', 'Dx:CIN', 'Dx:HPV', 'Dx', 'Hinselmann', 'Schiller',
       'Citology', 'Biopsy']

for i in range(len(cols)):
    if cols[i] in high_weights:
        weights[i] = 3
    if cols[i] in med_weights:
        weights[i] = 2
    if cols[i] in low_weights:
        weights[i] = 1
```
## Comparison function
Now that the data is all set up and we have the weights prepared, we can begin with the comparison function.
![images/compare.png](./images/compare.png)
The comparison functions iterates over each row of the synthetic data and compares it to one selected row of the real data. 

``` py
def compare(data1, data2, row_comp, weights, mini=False):
    """
    This is a comparsion function  
    data1: First dataset for comparsion, 1 set row
    data2: Second dataset for comparsion
    row_comp: The row we want compared
    mini: Chooses if the function returns the minimum value or not
    """
    lst = []
    for i in range(len(data2)):
        lst.append(similiar(data1, data2, i, row_comp, weights))
    if mini:
        lst = min(lst)
    return lst
```
The comparison function also runs the similarity function into it which will be responsible for providing a numerical comparison of how similar the rows truly are. 
 
## Scoring
The similiar() function works by comparing how similar the synthetic data is to the real data.
``` Py
def similiar(data1, data2, ind, row_comp, weights):
    """
    Computes how similiar the two datasets are 
    """
    score = 0
    for i in range(len(data1.columns)):
        score = score + (abs(data1.loc[row_comp,:][i] - data2.loc[ind,:][i]) * weights[i])
    return score 
```
We want this comparison function to work differently depending on which column the data falls under. Currently the data just takes the absolute difference of the datasets and then multiplies it by the correct corresponding weight. 

We want the comparison function to change based on the numbers and what is present in that column. For now, we simply use the simple comparison function.

## Ranking
The ranking system currently is simply the lowest result from the comparison system. We start off by finding the minimum, which results in showing us the comparison number of the smallest comparison number. Then, we find the index of the value we're looking for and finally use that index to figure out the row of the values we're interested in observing. 

## Matching
By repeating our comparison function and obtaining scores, we can generate a new dataframe that holds all the information. This dataframe has the same number of rows as the synthetic data and then the rows of the old real data become the columns for this new data and the score for their matching being the numbers that correspond to the row of both the synthetic and real data. From here, we can choose the minimum value and that becomes the closest match in this algorithm.

The Gale Shapley part of this algorithm comes once the scores are obtained as instead of doing the smallest scores, we can also implement the Gale Shapley test in order and rank which synthetic data is closest to the specific real columns. 

## Heatmaps
The Heatmaps are generated using matplotlib and work based off of the score values. The lightest color and darkest colors for the heatmap depend on the minimum and maximum scores respectively.


## Current Issues
How to deal with missing values
The scoring metric
- Hamming error was mentioned
The error metric
