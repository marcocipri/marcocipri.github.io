
# Hyperparameter Tuning & ML Pipelines

This is the notebook underlying the reveal.js slides that were used for Richmond Data Science community meetup meetup on May 29th, 2018.

<!--more-->
## About me:

I am Atul Saurav (@twtAtul), 
- Lead Genworth Financials' Data Engineering Team
- MS in Decison Analytics, VCU DAPT class of 2019
- Passionate about learning, data and everything around it

You can also find me on LinkedIn

## What is this talk about?

- Building Machine Learning Models
- Python specific
- scikit-learn based models

## What is this talk not about?

- Data Cleansing
- Feature Engineering
- Deep Learning 
- Other exciting stuff that is difficult to cover in 1 meetup!


># All models are wrong but some are useful
</br>
</br>
<div class="small">Box, G. E. P. (1979), "Robustness in the strategy of scientific model building", in Launer, R. L.; Wilkinson, G. N., Robustness in Statistics, Academic Press, pp. 201–236.</div>


## Reality is complex - 

<ul>
    <li><p class="fragment fade-left">too many factors influence outcome</p></li>
    <li><p class="fragment fade-left">factors difficult to measure accurately and objectively</p></li>
    <li><p class="fragment fade-left">not all factors may be known</p></li>
</ul>

> ## How do we build <span class="fragment highlight-green">useful</span> models?
> ## How do we <span class="fragment highlight-green">minimize</span> our <span class="fragment highlight-red">effort</span> in model building?

## Approach
- Use toy datasets for illustration and visualization
- Use real dataset for demonstrating application efficacy

## Scikit-learn API Overview

- All methods are implemented as estimators
- All estimators have a ```.fit()``` method
- All supervised estimators have ```.predict()``` method
- All unsupervised estimators have ```.transform()``` method

<table>
<tr style="border:None; font-size:20px; padding:10px;"><th>``model.predict``</th><th>``model.transform``</th></tr>
<tr style="border:None; font-size:20px; padding:10px;"><td>Classification</td><td>Preprocessing</td></tr>
<tr style="border:None; font-size:20px; padding:10px;"><td>Regression</td><td>Dimensionality Reduction</td></tr>
<tr style="border:None; font-size:20px; padding:10px;"><td>Clustering</td><td>Feature Extraction</td></tr>
<tr style="border:None; font-size:20px; padding:10px;"><td>&nbsp;</td><td>Feature Selection</td></tr>

</table>


### Usual High Level Flow


```python
from sklearn.family import SomeModel
myModel = SomeModel()
myModel.fit(X_train,y_train)

# supervised
myModel.predict(X_test)
myModel.score(X_test, y_test)

## unsupervised
myModel.transform(X_train)
```

## Problem Statement

- Classify data into 2 groups based on already observed groupings
- Using house price data, predict if the price of the given house will be >= 500K

> ### Binary Classification Problem

### Toy Example


```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
plt.style.use('seaborn-poster')
_ = plt.xkcd()
```


```python
from sklearn.datasets import make_blobs

scaled_X, scaled_y = make_blobs(centers=2, random_state=0)
scaled_X[:,0] = 10* scaled_X[:,0] + 3
X, y = make_blobs(centers=2, random_state=0)

print('X ~ n_samples x n_features:', X.shape)
print('y ~ n_samples:', y.shape)
```

    X ~ n_samples x n_features: (100, 2)
    y ~ n_samples: (100,)


### Visualize Data


```python
fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(10,5))

ax0.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', s=40, label='0')
ax0.scatter(X[y == 1, 0], X[y == 1, 1], c='red', s=40, label='1', marker='s')
ax0.set_xlabel('first feature')
ax0.set_ylabel('second feature')

ax1.scatter(scaled_X[scaled_y == 0, 0], scaled_X[scaled_y == 0, 1], c='blue', s=40, label='0')
ax1.scatter(scaled_X[scaled_y == 1, 0], scaled_X[scaled_y == 1, 1], c='red', s=40, label='1', marker='s')
ax1.set_xlabel('first feature')
```




    Text(0.5,0,'first feature')




![png](https://s3.amazonaws.com/ghpage/htmp/output_19_1.png)


### Create Training and Test Set

**Training Set** - Data used to train the model
 
**Test Set** - Data held out in the very beginning for testing model performance. This data should not be touched until final scoring


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.25,
                                                    random_state=1234,
                                                    stratify=y)
scaled_X_train, scaled_X_test, \
scaled_y_train, scaled_y_test = train_test_split(scaled_X, 
                                                    scaled_y,
                                                    test_size=0.25,
                                                    random_state=1234,
                                                    stratify=y)
```


```python
X_train.shape
```




    (75, 2)




```python
y_train.shape
```




    (75,)



### Train a Nearest Neighbor Classifier - 1 Neighbor

Overfit!!


```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)
```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=1, n_neighbors=1, p=2,
               weights='uniform')




```python
scaled_knn = KNeighborsClassifier(n_neighbors=1)
scaled_knn.fit(scaled_X_train, scaled_y_train)
```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=1, n_neighbors=1, p=2,
               weights='uniform')



### Comapre Model Performance


```python
knn.score(X_test, y_test)
```




    1.0




```python
scaled_knn.score(scaled_X_test, scaled_y_test)
```




    0.95999999999999996



### Visualize Model - regular


```python
from figures import plot_2d_separator
plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', s=40, label='0')
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', s=40, label='1', marker='s')

plt.xlabel("first feature")
plt.ylabel("second feature")
plot_2d_separator(knn, X)
_ = plt.legend()
```


![png](https://s3.amazonaws.com/ghpage/htmp/output_31_0.png)


### Visualize Model - scaled


```python
from figures import plot_2d_separator
plt.scatter(scaled_X[scaled_y == 0, 0], scaled_X[scaled_y == 0, 1], c='blue', s=40, label='0')
plt.scatter(scaled_X[scaled_y == 1, 0], scaled_X[scaled_y == 1, 1], c='red', s=40, label='1', marker='s')

plt.xlabel("first feature")
plt.ylabel("second feature")
plot_2d_separator(scaled_knn, scaled_X)
_ = plt.legend()
```


![png](https://s3.amazonaws.com/ghpage/htmp/output_33_0.png)


### Revelation 

> ### Scale of various features matters!

### Train Nearest Neighbor Classifier - 10 neighbors


```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)

knn.fit(X_train, y_train)
```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=1, n_neighbors=10, p=2,
               weights='uniform')




```python
scaled_knn = KNeighborsClassifier(n_neighbors=10)
scaled_knn.fit(scaled_X_train, scaled_y_train)
```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=1, n_neighbors=10, p=2,
               weights='uniform')



### Comapre Model Performance


```python
knn.score(X_test, y_test)
```




    0.83999999999999997




```python
scaled_knn.score(scaled_X_test, scaled_y_test)
```




    0.80000000000000004



### Visualize Model - regular


```python
from figures import plot_2d_separator
plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', s=40, label='0')
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', s=40, label='1', marker='s')

plt.xlabel("first feature")
plt.ylabel("second feature")
plot_2d_separator(knn, X)
_ = plt.legend()
```


![png](https://s3.amazonaws.com/ghpage/htmp/output_42_0.png)


### Visualize Model - scaled


```python
from figures import plot_2d_separator
plt.scatter(scaled_X[scaled_y == 0, 0], scaled_X[scaled_y == 0, 1], c='blue', s=40, label='0')
plt.scatter(scaled_X[scaled_y == 1, 0], scaled_X[scaled_y == 1, 1], c='red', s=40, label='1', marker='s')

plt.xlabel("first feature")
plt.ylabel("second feature")
plot_2d_separator(scaled_knn, scaled_X)
_ = plt.legend()
```


![png](https://s3.amazonaws.com/ghpage/htmp/output_44_0.png)


### Revelation 

> ### Number of neighbors matters as well!

### Tunning # of Neighbors


```python
train_scores = []
test_scores = []
n_neighbors = range(1,28)
for neighbor in n_neighbors:
    knn = KNeighborsClassifier(n_neighbors=neighbor)
    knn.fit(X_train, y_train)
    train_scores.append(knn.score(X_train, y_train))
    test_scores.append(knn.score(X_test, y_test))

plt.plot(n_neighbors, train_scores, label='train')
plt.plot(n_neighbors, test_scores, label='test')
plt.ylabel('Accuracy')
plt.xlabel('# of neighbors')
plt.legend();plt.show()
```


![png](https://s3.amazonaws.com/ghpage/htmp/output_47_0.png)


> ### But Wait!! Is that Hyperparameter Tuning?

> # No

> ### Hyperparameter tuning is part of Model building and Test Data <span style="color:red">should not</span> be used in model build

> ### Hyperparameter Tuning should be performed using <span style="color:blue">Validation Set</span> - a subset of training set


```python
import pandas as pd
# get columns with null - to return sorted in future
def null_pct(df):
    return {  k:sum(df[k].isnull())/len(df) for k in df.columns}
def null_count(df):
    return {  k:sum(df[k].isnull()) for k in df.columns}
```


```python
from sklearn.base import TransformerMixin, BaseEstimator

class CategoricalTransformer(BaseEstimator, TransformerMixin):
    "Converts a set of columns in a DataFrame to categoricals"
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        'Records the categorical information'
        self.cat_map_ = {col: X[col].astype('category').cat
                         for col in self.columns}
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for col in self.columns:
            X[col] = pd.Categorical(X[col],
            categories=self.cat_map_[col].categories,
            ordered=self.cat_map_[col].ordered)
        return X

    def inverse_transform(self, trn, y=None):
        trn = trn.copy()
        trn[self.columns] = trn[self.columns].apply(lambda x: x.astype(object))
        return trn
    
class DummyEncoder(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self.columns_ = X.columns
        self.cat_cols_ = X.select_dtypes(include=['category']).columns
        self.non_cat_cols_ = X.columns.drop(self.cat_cols_)
        self.cat_map_ = {col: X[col].cat for col in self.cat_cols_}

        self.cat_blocks_ = {}  # {cat col: slice}
        left = len(self.non_cat_cols_)
        for col in self.cat_cols_:
            right = left + len(self.cat_map_[col].categories)
            self.cat_blocks_[col] = slice(left, right)
            left = right
        return self

    def transform(self, X, y=None):
        return np.asarray(pd.get_dummies(X))

    def inverse_transform(self, trn, y=None):
        numeric = pd.DataFrame(trn[:, :len(self.non_cat_cols_)],
                               columns=self.non_cat_cols_)
        series = []
        for col, slice_ in self.cat_blocks_.items():
            codes = trn[:, slice_].argmax(1)
            cat = self.cat_map_[col]
            cat = pd.Categorical.from_codes(codes,
                                            cat.categories,
                                            cat.ordered)
            series.append(pd.Series(cat, name=col))
        return pd.concat([numeric] + series, axis='columns')[self.columns_]
```

### Real Life Example - Housing Data


```python
data = pd.read_csv('new_train.csv')
data.columns = [ x.lower().replace('.','_') for x in data.columns]
data.head().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>id</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>sale_type</th>
      <td>MLS Listing</td>
      <td>MLS Listing</td>
      <td>MLS Listing</td>
      <td>MLS Listing</td>
      <td>MLS Listing</td>
    </tr>
    <tr>
      <th>sold_date</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>property_type</th>
      <td>Condo/Co-op</td>
      <td>Single Family Residential</td>
      <td>Single Family Residential</td>
      <td>Single Family Residential</td>
      <td>Single Family Residential</td>
    </tr>
    <tr>
      <th>city</th>
      <td>Kew Gardens</td>
      <td>Anaheim</td>
      <td>Howard Beach</td>
      <td>Aliso Viejo</td>
      <td>Orlando</td>
    </tr>
    <tr>
      <th>state</th>
      <td>NY</td>
      <td>CA</td>
      <td>NY</td>
      <td>CA</td>
      <td>FL</td>
    </tr>
    <tr>
      <th>zip</th>
      <td>11415</td>
      <td>92807</td>
      <td>11414</td>
      <td>92656</td>
      <td>32837</td>
    </tr>
    <tr>
      <th>beds</th>
      <td>0</td>
      <td>7</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
    </tr>
    <tr>
      <th>baths</th>
      <td>1</td>
      <td>5.5</td>
      <td>1.5</td>
      <td>4.5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>location</th>
      <td>The Texas</td>
      <td>91 - Sycamore Canyon</td>
      <td>Howard Beach</td>
      <td>AV - Aliso Viejo</td>
      <td>Orlando</td>
    </tr>
    <tr>
      <th>square_feet</th>
      <td>NaN</td>
      <td>7400</td>
      <td>NaN</td>
      <td>3258</td>
      <td>1596</td>
    </tr>
    <tr>
      <th>lot_size</th>
      <td>NaN</td>
      <td>56628</td>
      <td>2400</td>
      <td>5893</td>
      <td>5623</td>
    </tr>
    <tr>
      <th>year_built</th>
      <td>1956</td>
      <td>2000</td>
      <td>1950</td>
      <td>2011</td>
      <td>1994</td>
    </tr>
    <tr>
      <th>days_on_market</th>
      <td>1</td>
      <td>2</td>
      <td>15</td>
      <td>6</td>
      <td>8</td>
    </tr>
    <tr>
      <th>x__square_feet</th>
      <td>NaN</td>
      <td>514</td>
      <td>NaN</td>
      <td>457</td>
      <td>166</td>
    </tr>
    <tr>
      <th>hoa_month</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>258</td>
      <td>64</td>
    </tr>
    <tr>
      <th>status</th>
      <td>Active</td>
      <td>Active</td>
      <td>Active</td>
      <td>Active</td>
      <td>Active</td>
    </tr>
    <tr>
      <th>next_open_house_start_time</th>
      <td>March-11-2018 01:00 PM</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>next_open_house_end_time</th>
      <td>March-11-2018 03:00 PM</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>source</th>
      <td>MLSLI</td>
      <td>CRMLS</td>
      <td>MLSLI</td>
      <td>CRMLS</td>
      <td>MFRMLS</td>
    </tr>
    <tr>
      <th>favorite</th>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
    </tr>
    <tr>
      <th>interested</th>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>latitude</th>
      <td>40.7</td>
      <td>33.8</td>
      <td>40.7</td>
      <td>33.6</td>
      <td>28.4</td>
    </tr>
    <tr>
      <th>longitude</th>
      <td>-73.8</td>
      <td>-117.8</td>
      <td>-73.8</td>
      <td>-117.7</td>
      <td>-81.4</td>
    </tr>
    <tr>
      <th>target</th>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



### Some Observations


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 19318 entries, 0 to 19317
    Data columns (total 25 columns):
    id                            19318 non-null int64
    sale_type                     19318 non-null object
    sold_date                     0 non-null float64
    property_type                 19318 non-null object
    city                          19306 non-null object
    state                         19318 non-null object
    zip                           19271 non-null object
    beds                          18216 non-null float64
    baths                         18053 non-null float64
    location                      18773 non-null object
    square_feet                   15693 non-null float64
    lot_size                      10267 non-null float64
    year_built                    16950 non-null float64
    days_on_market                18328 non-null float64
    x__square_feet                15693 non-null float64
    hoa_month                     7553 non-null float64
    status                        19318 non-null object
    next_open_house_start_time    933 non-null object
    next_open_house_end_time      933 non-null object
    source                        19318 non-null object
    favorite                      19318 non-null object
    interested                    19318 non-null object
    latitude                      19318 non-null float64
    longitude                     19318 non-null float64
    target                        19318 non-null bool
    dtypes: bool(1), float64(11), int64(1), object(12)
    memory usage: 3.6+ MB


### What's different?

- Not all features are numeric - **Categorical Variables **
- Lot of missing data points

### More Revalations:
> ### scikit-learn models need data to be numeric/float
> ### scikit-learn models implicitly cannot handle missing values

### Final Revelation

> ### Any transformation applied on the training set to handle first 3 revelations should later be applied on the test set as well


```python
del(data['sold_date'])
del(data['next_open_house_start_time'])
del(data['next_open_house_end_time'])
```

## Summary so far

We need to string following steps into a managable fashion to build <span style="color:red">effective</span> models

- Handle missing data
- Handle different scales across features
- Handle categorical data

### - Pipelines

#### But then also tune the model

- split training data into (cross)validation sets
- search best values for hyperparameters for optimal model performance

### - GridSearch

### More on Pipelines and GridSearchCV

- also known as meta-estimators in scikit-learn
- they inherit the properties of the last estimator used
- pipeline is used to chain multiple estimator into one pipe so output of one flows as input of next
- GridSearchCV is used to search on a hyperparameter grid the model with the best score 

### Making ML Pipelines


```python
from sklearn.pipeline import make_pipeline
make_pipeline( CategoricalTransformer(columns=cat_cols), DummyClassifier("most_frequent"))
```

This will inherit the properties of DummyClassifier

![pipeline](https://s3.amazonaws.com/ghpage/htmp/pipeline.svg)

Image Source: <a href="https://github.com/amueller/scipy-2017-sklearn/">SciPy 2016 Scikit-learn Tutorial</a>

> ## Back to real life housing classification problem

### Which features have less than 20% missing values?


```python
d = pd.DataFrame(null_pct(data), index=['null_pct']).T.sort_values('null_pct')
d[(d.null_pct< .2 )]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>null_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>id</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>latitude</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>interested</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>favorite</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>source</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>status</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>longitude</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>target</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>state</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>property_type</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>sale_type</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>city</th>
      <td>0.000621</td>
    </tr>
    <tr>
      <th>zip</th>
      <td>0.002433</td>
    </tr>
    <tr>
      <th>location</th>
      <td>0.028212</td>
    </tr>
    <tr>
      <th>days_on_market</th>
      <td>0.051248</td>
    </tr>
    <tr>
      <th>beds</th>
      <td>0.057045</td>
    </tr>
    <tr>
      <th>baths</th>
      <td>0.065483</td>
    </tr>
    <tr>
      <th>year_built</th>
      <td>0.122580</td>
    </tr>
    <tr>
      <th>square_feet</th>
      <td>0.187649</td>
    </tr>
    <tr>
      <th>x__square_feet</th>
      <td>0.187649</td>
    </tr>
  </tbody>
</table>
</div>



### Set features to work with


```python
new_features = ['id', 'favorite', 'interested', 'latitude', 'longitude', 'status', 'property_type', 'sale_type', 'source', 
                'state', 'beds', 'baths', 'year_built', 'x__square_feet', 'square_feet', 'target']

sub_data = data[new_features]
sub_data.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>favorite</th>
      <th>interested</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>status</th>
      <th>property_type</th>
      <th>sale_type</th>
      <th>source</th>
      <th>state</th>
      <th>beds</th>
      <th>baths</th>
      <th>year_built</th>
      <th>x__square_feet</th>
      <th>square_feet</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>N</td>
      <td>Y</td>
      <td>40.7</td>
      <td>-73.8</td>
      <td>Active</td>
      <td>Condo/Co-op</td>
      <td>MLS Listing</td>
      <td>MLSLI</td>
      <td>NY</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1956.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>N</td>
      <td>Y</td>
      <td>33.8</td>
      <td>-117.8</td>
      <td>Active</td>
      <td>Single Family Residential</td>
      <td>MLS Listing</td>
      <td>CRMLS</td>
      <td>CA</td>
      <td>7.0</td>
      <td>5.5</td>
      <td>2000.0</td>
      <td>514.0</td>
      <td>7400.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>N</td>
      <td>Y</td>
      <td>40.7</td>
      <td>-73.8</td>
      <td>Active</td>
      <td>Single Family Residential</td>
      <td>MLS Listing</td>
      <td>MLSLI</td>
      <td>NY</td>
      <td>3.0</td>
      <td>1.5</td>
      <td>1950.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
print (len(sub_data))
sub_data = sub_data.drop_duplicates(subset=['favorite', 'interested', 'latitude', 'longitude', 'status', 'property_type', 'sale_type', 
                                            'source', 'state', 'beds', 'baths', 'year_built', 'x__square_feet', 'square_feet', 'target'])
print (len(sub_data))
```

    19318
    17815



```python
sub_data = sub_data.copy()
sub_data.loc[sub_data['sale_type'] == 'New Construction Plan','year_built'] = 2018.0
sub_data.loc[sub_data['property_type'] == 'Vacant Land','year_built'] = 2019.0
```


```python
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import Imputer, RobustScaler, StandardScaler
```


```python
def plot_roc(model, X_test, y_test):
    df = model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, df)
    acc = model.score(X_test, y_test)
    auc0 = roc_auc_score(y_test, df)
    auc1 = roc_auc_score(y_test, model.predict(X_test))
    plt.plot(fpr, tpr, label="acc:%.2f auc0:%.2f auc1:%.2f" % (acc, auc0, auc1), linewidth=3)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (recall)")
    plt.title(repr(model).split('(')[0])
    plt.legend(loc="best");
```


```python
roc_auc_scorer = make_scorer(roc_auc_score, greater_is_better=True,
                             needs_threshold=True)
```

### Set model Inputs


```python
X = sub_data[['favorite', 'interested', 'latitude', 'longitude', 'status', 'property_type', 'sale_type', 
                    'source', 'state', 'beds', 'baths', 'year_built', 'x__square_feet', 'square_feet']]
y = sub_data.target
```


```python
cat_cols = ['favorite', 'interested', 'status', 'property_type', 'sale_type', 'source', 'state']
```

### Establish base case for prediction


```python
dummy_pipe = make_pipeline( CategoricalTransformer(columns=cat_cols), DummyEncoder(), DummyClassifier("most_frequent"))
cross_val_score(dummy_pipe, X, y)
```




    array([ 0.68984678,  0.68996295,  0.68996295])



### Create training and test set


```python
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42 )
```

### Support Vector Machine


```python
Cs = [ 1, 10, 100, 500, 750] 
gammas = [0.0005, 0.001, 0.01, .1, 1] 
param_grid = {'svc__C': Cs, 'svc__gamma' : gammas}
```


```python
svc_pipe = make_pipeline( CategoricalTransformer(columns=cat_cols), DummyEncoder(), Imputer(strategy='median'),  StandardScaler(), SVC(random_state=42) )
```


```python
svmgrid = GridSearchCV(svc_pipe, param_grid, cv=5, n_jobs=-1, verbose=3)#, scoring=roc_auc_scorer)
svmgrid.fit(X_train, y_train)
```

    Fitting 5 folds for each of 25 candidates, totalling 125 fits
    [CV] svc__C=1, svc__gamma=0.0005 .....................................
    [CV] svc__C=1, svc__gamma=0.0005 .....................................
    [CV] svc__C=1, svc__gamma=0.0005 .....................................
    [CV] svc__C=1, svc__gamma=0.0005 .....................................
    [CV] svc__C=1, svc__gamma=0.0005 .....................................
    [CV] svc__C=1, svc__gamma=0.001 ......................................
    [CV] svc__C=1, svc__gamma=0.001 ......................................
    [CV] svc__C=1, svc__gamma=0.001 ......................................
    [CV]  svc__C=1, svc__gamma=0.001, score=0.8323980546202768, total=  17.9s
    [CV] svc__C=1, svc__gamma=0.001 ......................................
    [CV]  svc__C=1, svc__gamma=0.0005, score=0.7875046763935653, total=  19.0s
    [CV] svc__C=1, svc__gamma=0.001 ......................................
    [CV]  svc__C=1, svc__gamma=0.001, score=0.7639356528245417, total=  19.4s
    [CV] svc__C=1, svc__gamma=0.01 .......................................
    [CV]  svc__C=1, svc__gamma=0.001, score=0.780314371257485, total=  19.2s
    [CV] svc__C=1, svc__gamma=0.01 .......................................
    [CV]  svc__C=1, svc__gamma=0.0005, score=0.7542087542087542, total=  19.8s
    [CV] svc__C=1, svc__gamma=0.01 .......................................
    [CV]  svc__C=1, svc__gamma=0.0005, score=0.7746162485960314, total=  20.2s
    [CV] svc__C=1, svc__gamma=0.01 .......................................
    [CV]  svc__C=1, svc__gamma=0.0005, score=0.7690868263473054, total=  20.4s
    [CV] svc__C=1, svc__gamma=0.01 .......................................
    [CV]  svc__C=1, svc__gamma=0.0005, score=0.7784431137724551, total=  20.6s
    [CV] svc__C=1, svc__gamma=0.1 ........................................
    [CV]  svc__C=1, svc__gamma=0.01, score=0.8787878787878788, total=  13.7s
    [CV] svc__C=1, svc__gamma=0.1 ........................................
    [CV]  svc__C=1, svc__gamma=0.1, score=0.8327721661054994, total=  14.7s
    [CV] svc__C=1, svc__gamma=0.1 ........................................
    [CV]  svc__C=1, svc__gamma=0.01, score=0.8166853722409277, total=  16.5s
    [CV] svc__C=1, svc__gamma=0.1 ........................................
    [CV]  svc__C=1, svc__gamma=0.01, score=0.844311377245509, total=  16.5s
    [CV] svc__C=1, svc__gamma=0.1 ........................................
    [CV]  svc__C=1, svc__gamma=0.01, score=0.8352676900037439, total=  16.2s
    [CV]  svc__C=1, svc__gamma=0.01, score=0.8334580838323353, total=  16.7s
    [CV] svc__C=1, svc__gamma=1 ..........................................
    [CV] svc__C=1, svc__gamma=1 ..........................................
    [CV]  svc__C=1, svc__gamma=0.001, score=0.7907934131736527, total=  18.7s
    [CV] svc__C=1, svc__gamma=1 ..........................................
    [CV]  svc__C=1, svc__gamma=0.001, score=0.7858479970048671, total=  18.5s
    [CV] svc__C=1, svc__gamma=1 ..........................................


    [Parallel(n_jobs=-1)]: Done  16 tasks      | elapsed:   55.7s


    [CV]  svc__C=1, svc__gamma=0.1, score=0.8843995510662177, total=  14.3s
    [CV] svc__C=1, svc__gamma=1 ..........................................
    [CV] . svc__C=1, svc__gamma=1, score=0.9042274597830153, total=  14.2s
    [CV] svc__C=10, svc__gamma=0.0005 ....................................
    [CV]  svc__C=1, svc__gamma=0.1, score=0.8510479041916168, total=  15.9s
    [CV] svc__C=10, svc__gamma=0.0005 ....................................
    [CV]  svc__C=1, svc__gamma=0.1, score=0.847997004867091, total=  15.5s
    [CV] svc__C=10, svc__gamma=0.0005 ....................................
    [CV]  svc__C=1, svc__gamma=0.1, score=0.8401946107784432, total=  16.3s
    [CV] svc__C=10, svc__gamma=0.0005 ....................................
    [CV] . svc__C=1, svc__gamma=1, score=0.8514777403666293, total=  16.0s
    [CV] svc__C=10, svc__gamma=0.0005 ....................................
    [CV] . svc__C=1, svc__gamma=1, score=0.8589071856287425, total=  16.2s
    [CV] svc__C=10, svc__gamma=0.001 .....................................
    [CV] . svc__C=1, svc__gamma=1, score=0.8581586826347305, total=  15.8s
    [CV] svc__C=10, svc__gamma=0.001 .....................................
    [CV] . svc__C=1, svc__gamma=1, score=0.8629726694122052, total=  15.0s
    [CV] svc__C=10, svc__gamma=0.001 .....................................
    [CV]  svc__C=10, svc__gamma=0.0005, score=0.8769173213617658, total=  15.3s
    [CV] svc__C=10, svc__gamma=0.001 .....................................
    [CV]  svc__C=10, svc__gamma=0.001, score=0.8903853348297793, total=  14.2s
    [CV] svc__C=10, svc__gamma=0.001 .....................................
    [CV]  svc__C=10, svc__gamma=0.0005, score=0.7942386831275721, total=  17.5s
    [CV] svc__C=10, svc__gamma=0.01 ......................................
    [CV]  svc__C=10, svc__gamma=0.0005, score=0.8244760479041916, total=  17.8s
    [CV] svc__C=10, svc__gamma=0.01 ......................................
    [CV]  svc__C=10, svc__gamma=0.001, score=0.8065843621399177, total=  17.4s
    [CV]  svc__C=10, svc__gamma=0.0005, score=0.8181137724550899, total=  18.3s
    [CV] svc__C=10, svc__gamma=0.01 ......................................
    [CV] svc__C=10, svc__gamma=0.01 ......................................
    [CV]  svc__C=10, svc__gamma=0.0005, score=0.8172968925496069, total=  18.2s
    [CV] svc__C=10, svc__gamma=0.01 ......................................
    [CV]  svc__C=10, svc__gamma=0.001, score=0.8327095808383234, total=  16.5s
    [CV] svc__C=10, svc__gamma=0.1 .......................................
    [CV]  svc__C=10, svc__gamma=0.01, score=0.9176954732510288, total=  11.6s
    [CV] svc__C=10, svc__gamma=0.1 .......................................
    [CV]  svc__C=10, svc__gamma=0.01, score=0.8555929667040778, total=  14.0s
    [CV] svc__C=10, svc__gamma=0.1 .......................................
    [CV]  svc__C=10, svc__gamma=0.001, score=0.8308383233532934, total=  17.0s
    [CV] svc__C=10, svc__gamma=0.1 .......................................
    [CV]  svc__C=10, svc__gamma=0.01, score=0.8768712574850299, total=  14.2s
    [CV] svc__C=10, svc__gamma=0.1 .......................................
    [CV]  svc__C=10, svc__gamma=0.01, score=0.8573567952077873, total=  14.2s
    [CV]  svc__C=10, svc__gamma=0.01, score=0.8566616766467066, total=  14.6s
    [CV] svc__C=10, svc__gamma=1 .........................................
    [CV] svc__C=10, svc__gamma=1 .........................................
    [CV]  svc__C=10, svc__gamma=0.001, score=0.8296518157993261, total=  16.9s
    [CV] svc__C=10, svc__gamma=1 .........................................
    [CV]  svc__C=10, svc__gamma=0.1, score=0.9259259259259259, total=  10.5s
    [CV] svc__C=10, svc__gamma=1 .........................................
    [CV]  svc__C=10, svc__gamma=0.1, score=0.8600823045267489, total=  13.9s
    [CV] svc__C=10, svc__gamma=1 .........................................
    [CV]  svc__C=10, svc__gamma=0.1, score=0.8697604790419161, total=  13.9s
    [CV] svc__C=100, svc__gamma=0.0005 ...................................
    [CV]  svc__C=10, svc__gamma=1, score=0.9266741488963711, total=  12.2s
    [CV] svc__C=100, svc__gamma=0.0005 ...................................
    [CV]  svc__C=10, svc__gamma=0.1, score=0.8645209580838323, total=  13.8s
    [CV] svc__C=100, svc__gamma=0.0005 ...................................
    [CV]  svc__C=10, svc__gamma=0.1, score=0.871209284912018, total=  14.1s
    [CV] svc__C=100, svc__gamma=0.0005 ...................................
    [CV]  svc__C=10, svc__gamma=1, score=0.8873502994011976, total=  14.0s
    [CV] svc__C=100, svc__gamma=0.0005 ...................................
    [CV]  svc__C=10, svc__gamma=1, score=0.8829031051253273, total=  14.9s
    [CV] svc__C=100, svc__gamma=0.001 ....................................
    [CV]  svc__C=10, svc__gamma=1, score=0.8828592814371258, total=  13.3s
    [CV] svc__C=100, svc__gamma=0.001 ....................................
    [CV]  svc__C=10, svc__gamma=1, score=0.8888056907525271, total=  12.8s
    [CV] svc__C=100, svc__gamma=0.001 ....................................
    [CV]  svc__C=100, svc__gamma=0.0005, score=0.9075944631500187, total=  12.9s
    [CV] svc__C=100, svc__gamma=0.001 ....................................
    [CV]  svc__C=100, svc__gamma=0.0005, score=0.8338945005611672, total=  16.1s
    [CV] svc__C=100, svc__gamma=0.001 ....................................
    [CV]  svc__C=100, svc__gamma=0.0005, score=0.843937125748503, total=  16.0s
    [CV] svc__C=100, svc__gamma=0.01 .....................................
    [CV]  svc__C=100, svc__gamma=0.001, score=0.8537224092779648, total=  15.2s
    [CV] svc__C=100, svc__gamma=0.01 .....................................
    [CV]  svc__C=100, svc__gamma=0.0005, score=0.8476796407185628, total=  16.6s
    [CV] svc__C=100, svc__gamma=0.01 .....................................
    [CV]  svc__C=100, svc__gamma=0.0005, score=0.8457506551853239, total=  16.6s
    [CV] svc__C=100, svc__gamma=0.01 .....................................
    [CV]  svc__C=100, svc__gamma=0.001, score=0.9173213617658063, total=  11.4s
    [CV] svc__C=100, svc__gamma=0.01 .....................................
    [CV]  svc__C=100, svc__gamma=0.001, score=0.8648952095808383, total=  14.5s
    [CV] svc__C=100, svc__gamma=0.1 ......................................
    [CV]  svc__C=100, svc__gamma=0.01, score=0.9334081556303778, total=   8.7s
    [CV] svc__C=100, svc__gamma=0.1 ......................................
    [CV]  svc__C=100, svc__gamma=0.001, score=0.8615269461077845, total=  14.9s
    [CV] svc__C=100, svc__gamma=0.1 ......................................
    [CV]  svc__C=100, svc__gamma=0.01, score=0.8806584362139918, total=  12.3s
    [CV] svc__C=100, svc__gamma=0.1 ......................................
    [CV]  svc__C=100, svc__gamma=0.001, score=0.8539872706851367, total=  14.5s
    [CV] svc__C=100, svc__gamma=0.1 ......................................
    [CV]  svc__C=100, svc__gamma=0.01, score=0.8948353293413174, total=  12.0s
    [CV] svc__C=100, svc__gamma=1 ........................................
    [CV]  svc__C=100, svc__gamma=0.01, score=0.8854790419161677, total=  12.3s
    [CV] svc__C=100, svc__gamma=1 ........................................
    [CV]  svc__C=100, svc__gamma=0.01, score=0.8831898165481094, total=  12.0s
    [CV] svc__C=100, svc__gamma=1 ........................................
    [CV]  svc__C=100, svc__gamma=0.1, score=0.9375233819678264, total=   7.8s
    [CV] svc__C=100, svc__gamma=1 ........................................
    [CV]  svc__C=100, svc__gamma=0.1, score=0.8881406659184437, total=  11.6s
    [CV] svc__C=100, svc__gamma=1 ........................................
    [CV]  svc__C=100, svc__gamma=0.1, score=0.9004491017964071, total=  11.9s
    [CV] svc__C=500, svc__gamma=0.0005 ...................................
    [CV]  svc__C=100, svc__gamma=1, score=0.9438832772166106, total=   9.7s
    [CV] svc__C=500, svc__gamma=0.0005 ...................................
    [CV]  svc__C=100, svc__gamma=0.1, score=0.8948353293413174, total=  12.2s
    [CV] svc__C=500, svc__gamma=0.0005 ...................................
    [CV]  svc__C=100, svc__gamma=0.1, score=0.8955447397978286, total=  11.6s
    [CV] svc__C=500, svc__gamma=0.0005 ...................................
    [CV]  svc__C=100, svc__gamma=1, score=0.9023569023569024, total=  12.9s
    [CV] svc__C=500, svc__gamma=0.0005 ...................................
    [CV]  svc__C=100, svc__gamma=1, score=0.906062874251497, total=  13.3s
    [CV] svc__C=500, svc__gamma=0.001 ....................................
    [CV]  svc__C=100, svc__gamma=1, score=0.905314371257485, total=  13.2s
    [CV] svc__C=500, svc__gamma=0.001 ....................................
    [CV]  svc__C=100, svc__gamma=1, score=0.8989142643204793, total=  12.7s
    [CV] svc__C=500, svc__gamma=0.001 ....................................
    [CV]  svc__C=500, svc__gamma=0.0005, score=0.9225589225589226, total=  10.4s
    [CV] svc__C=500, svc__gamma=0.001 ....................................
    [CV]  svc__C=500, svc__gamma=0.0005, score=0.8649457538346427, total=  14.5s
    [CV] svc__C=500, svc__gamma=0.001 ....................................
    [CV]  svc__C=500, svc__gamma=0.0005, score=0.874625748502994, total=  14.1s
    [CV] svc__C=500, svc__gamma=0.01 .....................................
    [CV]  svc__C=500, svc__gamma=0.0005, score=0.8708832335329342, total=  14.2s
    [CV] svc__C=500, svc__gamma=0.01 .....................................
    [CV]  svc__C=500, svc__gamma=0.001, score=0.9270482603815937, total=  10.5s
    [CV] svc__C=500, svc__gamma=0.01 .....................................
    [CV]  svc__C=500, svc__gamma=0.001, score=0.8750467639356528, total=  13.5s
    [CV] svc__C=500, svc__gamma=0.01 .....................................
    [CV]  svc__C=500, svc__gamma=0.0005, score=0.8655934107076001, total=  15.3s
    [CV] svc__C=500, svc__gamma=0.01 .....................................
    [CV]  svc__C=500, svc__gamma=0.001, score=0.8866017964071856, total=  13.3s
    [CV] svc__C=500, svc__gamma=0.1 ......................................
    [CV]  svc__C=500, svc__gamma=0.01, score=0.9393939393939394, total=   7.3s
    [CV] svc__C=500, svc__gamma=0.1 ......................................
    [CV]  svc__C=500, svc__gamma=0.001, score=0.8806137724550899, total=  13.2s
    [CV] svc__C=500, svc__gamma=0.1 ......................................
    [CV]  svc__C=500, svc__gamma=0.01, score=0.9019827908716798, total=  11.3s
    [CV] svc__C=500, svc__gamma=0.1 ......................................
    [CV]  svc__C=500, svc__gamma=0.001, score=0.8745788094346687, total=  13.1s
    [CV] svc__C=500, svc__gamma=0.1 ......................................
    [CV]  svc__C=500, svc__gamma=0.01, score=0.9191616766467066, total=  11.5s
    [CV] svc__C=500, svc__gamma=1 ........................................
    [CV]  svc__C=500, svc__gamma=0.01, score=0.9109281437125748, total=  11.8s
    [CV] svc__C=500, svc__gamma=1 ........................................
    [CV]  svc__C=500, svc__gamma=0.01, score=0.9041557469112692, total=  11.6s
    [CV] svc__C=500, svc__gamma=1 ........................................
    [CV]  svc__C=500, svc__gamma=0.1, score=0.9476243920688365, total=   8.5s
    [CV] svc__C=500, svc__gamma=1 ........................................
    [CV]  svc__C=500, svc__gamma=0.1, score=0.9094650205761317, total=  13.6s
    [CV] svc__C=500, svc__gamma=1 ........................................
    [CV]  svc__C=500, svc__gamma=0.1, score=0.9172904191616766, total=  14.4s
    [CV] svc__C=750, svc__gamma=0.0005 ...................................
    [CV]  svc__C=500, svc__gamma=0.1, score=0.9131736526946108, total=  13.3s
    [CV]  svc__C=500, svc__gamma=1, score=0.9386457164234941, total=  11.5s
    [CV] svc__C=750, svc__gamma=0.0005 ...................................
    [CV] svc__C=750, svc__gamma=0.0005 ...................................
    [CV]  svc__C=500, svc__gamma=0.1, score=0.9078996630475478, total=  13.6s
    [CV] svc__C=750, svc__gamma=0.0005 ...................................
    [CV]  svc__C=500, svc__gamma=1, score=0.9143284698840254, total=  17.6s
    [CV] svc__C=750, svc__gamma=0.0005 ...................................
    [CV]  svc__C=500, svc__gamma=1, score=0.9210329341317365, total=  16.6s
    [CV] svc__C=750, svc__gamma=0.001 ....................................
    [CV]  svc__C=500, svc__gamma=1, score=0.9161676646706587, total=  15.5s
    [CV] svc__C=750, svc__gamma=0.001 ....................................
    [CV]  svc__C=500, svc__gamma=1, score=0.9071508798202921, total=  16.6s
    [CV] svc__C=750, svc__gamma=0.001 ....................................
    [CV]  svc__C=750, svc__gamma=0.0005, score=0.9210624766180322, total=  10.1s
    [CV] svc__C=750, svc__gamma=0.001 ....................................
    [CV]  svc__C=750, svc__gamma=0.0005, score=0.8720538720538721, total=  14.5s
    [CV] svc__C=750, svc__gamma=0.001 ....................................
    [CV]  svc__C=750, svc__gamma=0.0005, score=0.8802395209580839, total=  14.6s
    [CV] svc__C=750, svc__gamma=0.01 .....................................
    [CV]  svc__C=750, svc__gamma=0.001, score=0.9285447063224841, total=  10.0s
    [CV] svc__C=750, svc__gamma=0.01 .....................................
    [CV] ....... svc__C=750, svc__gamma=0.0005, score=0.875, total=  14.9s
    [CV] svc__C=750, svc__gamma=0.01 .....................................
    [CV]  svc__C=750, svc__gamma=0.001, score=0.8787878787878788, total=  13.2s
    [CV] svc__C=750, svc__gamma=0.01 .....................................
    [CV]  svc__C=750, svc__gamma=0.0005, score=0.8715836765256458, total=  14.2s
    [CV] svc__C=750, svc__gamma=0.01 .....................................
    [CV]  svc__C=750, svc__gamma=0.001, score=0.8914670658682635, total=  12.8s
    [CV] svc__C=750, svc__gamma=0.1 ......................................
    [CV]  svc__C=750, svc__gamma=0.01, score=0.9438832772166106, total=   8.3s
    [CV] svc__C=750, svc__gamma=0.1 ......................................
    [CV]  svc__C=750, svc__gamma=0.001, score=0.8851047904191617, total=  13.9s
    [CV] svc__C=750, svc__gamma=0.1 ......................................
    [CV]  svc__C=750, svc__gamma=0.001, score=0.8809434668663422, total=  13.1s
    [CV] svc__C=750, svc__gamma=0.1 ......................................
    [CV]  svc__C=750, svc__gamma=0.01, score=0.9083426861204639, total=  12.9s
    [CV] svc__C=750, svc__gamma=0.1 ......................................
    [CV]  svc__C=750, svc__gamma=0.01, score=0.9202844311377245, total=  13.8s
    [CV] svc__C=750, svc__gamma=1 ........................................
    [CV]  svc__C=750, svc__gamma=0.01, score=0.9157934131736527, total=  14.0s
    [CV]  svc__C=750, svc__gamma=0.1, score=0.9479985035540591, total=   8.9s
    [CV] svc__C=750, svc__gamma=1 ........................................
    [CV] svc__C=750, svc__gamma=1 ........................................
    [CV]  svc__C=750, svc__gamma=0.01, score=0.909771621115687, total=  13.9s
    [CV] svc__C=750, svc__gamma=1 ........................................
    [CV]  svc__C=750, svc__gamma=0.1, score=0.9135802469135802, total=  14.0s
    [CV] svc__C=750, svc__gamma=1 ........................................
    [CV]  svc__C=750, svc__gamma=0.1, score=0.9217814371257484, total=  13.9s
    [CV]  svc__C=750, svc__gamma=0.1, score=0.9157934131736527, total=  15.2s
    [CV]  svc__C=750, svc__gamma=0.1, score=0.9112691875701985, total=  14.1s
    [CV]  svc__C=750, svc__gamma=1, score=0.9349046015712682, total=  11.3s
    [CV]  svc__C=750, svc__gamma=1, score=0.9165731387953611, total=  16.8s
    [CV]  svc__C=750, svc__gamma=1, score=0.9127994011976048, total=  14.1s
    [CV]  svc__C=750, svc__gamma=1, score=0.9225299401197605, total=  15.6s
    [CV]  svc__C=750, svc__gamma=1, score=0.9127667540247099, total=  12.5s


    [Parallel(n_jobs=-1)]: Done 125 out of 125 | elapsed:  5.4min finished





    GridSearchCV(cv=5, error_score='raise',
           estimator=Pipeline(memory=None,
         steps=[('categoricaltransformer', CategoricalTransformer(columns=['favorite', 'interested', 'status', 'property_type', 'sale_type', 'source', 'state'])), ('dummyencoder', DummyEncoder()), ('imputer', Imputer(axis=0, copy=True, missing_values='NaN', strategy='median', verbose=0)), ('standardscaler', ...f',
      max_iter=-1, probability=False, random_state=42, shrinking=True,
      tol=0.001, verbose=False))]),
           fit_params=None, iid=True, n_jobs=-1,
           param_grid={'svc__C': [1, 10, 100, 500, 750], 'svc__gamma': [0.0005, 0.001, 0.01, 0.1, 1]},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=3)




```python
roc_auc_score(y_test, svmgrid.predict(X_test))
```




    0.90180396517871109




```python
scores = svmgrid.cv_results_['mean_test_score'].reshape(5,5)
sns.heatmap(scores, vmax=1, xticklabels=param_grid['svc__gamma'], yticklabels=param_grid['svc__C'], cmap='hot', annot=True)
plt.title('Hyper parameters vs score')
plt.show()
```


![png](https://s3.amazonaws.com/ghpage/htmp/output_90_0.png)



```python
plot_roc(svmgrid, X_test, y_test)
plt.title('SVM')
```




    Text(0.5,1,'SVM')




![png](https://s3.amazonaws.com/ghpage/htmp/output_91_1.png)


### Adaboost


```python
from sklearn.ensemble import AdaBoostClassifier
```


```python
n_estimators = [ 1500, 1570, 2000, 2250, 3000, 3250, 3500]
param_grid = {'adaboostclassifier__n_estimators': n_estimators}
```


```python
abpipe = make_pipeline(CategoricalTransformer(columns=cat_cols), DummyEncoder(), Imputer(strategy='median'), StandardScaler(), 
                   AdaBoostClassifier())
```


```python
abgrid = GridSearchCV(abpipe, param_grid, cv=5, n_jobs=-1, verbose=3, scoring=roc_auc_scorer)
abgrid.fit(X_train, y_train)
```

    Fitting 5 folds for each of 7 candidates, totalling 35 fits
    [CV] adaboostclassifier__n_estimators=1500 ...........................
    [CV] adaboostclassifier__n_estimators=1500 ...........................
    [CV] adaboostclassifier__n_estimators=1500 ...........................
    [CV] adaboostclassifier__n_estimators=1500 ...........................
    [CV] adaboostclassifier__n_estimators=1500 ...........................
    [CV] adaboostclassifier__n_estimators=1570 ...........................
    [CV] adaboostclassifier__n_estimators=1570 ...........................
    [CV] adaboostclassifier__n_estimators=1570 ...........................
    [CV]  adaboostclassifier__n_estimators=1500, score=0.997032827451858, total=  20.4s
    [CV] adaboostclassifier__n_estimators=1570 ...........................
    [CV]  adaboostclassifier__n_estimators=1500, score=0.996398639757171, total=  20.4s
    [CV]  adaboostclassifier__n_estimators=1500, score=0.9964454477248496, total=  20.5s
    [CV] adaboostclassifier__n_estimators=1570 ...........................
    [CV] adaboostclassifier__n_estimators=2000 ...........................
    [CV]  adaboostclassifier__n_estimators=1500, score=0.9957317073170731, total=  20.5s
    [CV] adaboostclassifier__n_estimators=2000 ...........................
    [CV]  adaboostclassifier__n_estimators=1500, score=0.9974806405731442, total=  20.7s
    [CV] adaboostclassifier__n_estimators=2000 ...........................
    [CV]  adaboostclassifier__n_estimators=1570, score=0.9970124254669422, total=  21.3s
    [CV] adaboostclassifier__n_estimators=2000 ...........................
    [CV]  adaboostclassifier__n_estimators=1570, score=0.99648032853777, total=  21.4s
    [CV] adaboostclassifier__n_estimators=2000 ...........................
    [CV]  adaboostclassifier__n_estimators=1570, score=0.9974924932834642, total=  21.7s
    [CV] adaboostclassifier__n_estimators=2250 ...........................
    [CV]  adaboostclassifier__n_estimators=1570, score=0.9963485338755317, total=  22.4s
    [CV] adaboostclassifier__n_estimators=2250 ...........................
    [CV]  adaboostclassifier__n_estimators=1570, score=0.9957521203181793, total=  22.6s
    [CV] adaboostclassifier__n_estimators=2250 ...........................
    [CV]  adaboostclassifier__n_estimators=2000, score=0.9971358245692549, total=  27.9s
    [CV] adaboostclassifier__n_estimators=2250 ...........................
    [CV]  adaboostclassifier__n_estimators=2000, score=0.9965517354849749, total=  28.1s
    [CV] adaboostclassifier__n_estimators=2250 ...........................
    [CV]  adaboostclassifier__n_estimators=2000, score=0.9976037770636885, total=  28.7s
    [CV] adaboostclassifier__n_estimators=3000 ...........................
    [CV]  adaboostclassifier__n_estimators=2000, score=0.996344578148034, total=  28.6s
    [CV] adaboostclassifier__n_estimators=3000 ...........................
    [CV]  adaboostclassifier__n_estimators=2000, score=0.9958703181794236, total=  29.0s
    [CV] adaboostclassifier__n_estimators=3000 ...........................
    [CV]  adaboostclassifier__n_estimators=2250, score=0.9965122477722349, total=  32.4s
    [CV] adaboostclassifier__n_estimators=3000 ...........................


    [Parallel(n_jobs=-1)]: Done  16 tasks      | elapsed:  1.1min


    [CV]  adaboostclassifier__n_estimators=2250, score=0.9971371408263462, total=  31.8s
    [CV] adaboostclassifier__n_estimators=3000 ...........................
    [CV]  adaboostclassifier__n_estimators=2250, score=0.9975270636885634, total=  32.3s
    [CV] adaboostclassifier__n_estimators=3250 ...........................
    [CV]  adaboostclassifier__n_estimators=2250, score=0.9962773307805706, total=  31.8s
    [CV] adaboostclassifier__n_estimators=3250 ...........................
    [CV]  adaboostclassifier__n_estimators=2250, score=0.9959233261339093, total=  32.4s
    [CV] adaboostclassifier__n_estimators=3250 ...........................
    [CV]  adaboostclassifier__n_estimators=3000, score=0.9971509615258052, total=  42.0s
    [CV] adaboostclassifier__n_estimators=3250 ...........................
    [CV]  adaboostclassifier__n_estimators=3000, score=0.9966264330749082, total=  42.8s
    [CV] adaboostclassifier__n_estimators=3250 ...........................
    [CV]  adaboostclassifier__n_estimators=3000, score=0.9975313438339567, total=  42.7s
    [CV] adaboostclassifier__n_estimators=3500 ...........................
    [CV]  adaboostclassifier__n_estimators=3000, score=0.9959667860717484, total=  43.3s
    [CV] adaboostclassifier__n_estimators=3500 ...........................
    [CV]  adaboostclassifier__n_estimators=3000, score=0.9962802975761939, total=  43.1s
    [CV] adaboostclassifier__n_estimators=3500 ...........................
    [CV]  adaboostclassifier__n_estimators=3250, score=0.996706066628934, total=  46.5s
    [CV] adaboostclassifier__n_estimators=3500 ...........................
    [CV]  adaboostclassifier__n_estimators=3250, score=0.9972095349663697, total=  46.3s
    [CV] adaboostclassifier__n_estimators=3500 ...........................
    [CV]  adaboostclassifier__n_estimators=3250, score=0.9975178449138703, total=  48.1s
    [CV]  adaboostclassifier__n_estimators=3250, score=0.995986540588948, total=  47.2s
    [CV]  adaboostclassifier__n_estimators=3250, score=0.9962522778397508, total=  47.1s
    [CV]  adaboostclassifier__n_estimators=3500, score=0.9966935621865662, total=  50.6s
    [CV]  adaboostclassifier__n_estimators=3500, score=0.997175312281995, total=  49.3s


    [Parallel(n_jobs=-1)]: Done  32 out of  35 | elapsed:  2.8min remaining:   15.9s


    [CV]  adaboostclassifier__n_estimators=3500, score=0.9975040167518305, total=  40.4s
    [CV]  adaboostclassifier__n_estimators=3500, score=0.9959615182004952, total=  39.1s
    [CV]  adaboostclassifier__n_estimators=3500, score=0.9962753529168216, total=  36.8s


    [Parallel(n_jobs=-1)]: Done  35 out of  35 | elapsed:  3.2min finished





    GridSearchCV(cv=5, error_score='raise',
           estimator=Pipeline(memory=None,
         steps=[('categoricaltransformer', CategoricalTransformer(columns=['favorite', 'interested', 'status', 'property_type', 'sale_type', 'source', 'state'])), ('dummyencoder', DummyEncoder()), ('imputer', Imputer(axis=0, copy=True, missing_values='NaN', strategy='median', verbose=0)), ('standardscaler', ...m='SAMME.R', base_estimator=None,
              learning_rate=1.0, n_estimators=50, random_state=None))]),
           fit_params=None, iid=True, n_jobs=-1,
           param_grid={'adaboostclassifier__n_estimators': [1500, 1570, 2000, 2250, 3000, 3250, 3500]},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=make_scorer(roc_auc_score, needs_threshold=True), verbose=3)




```python
pd.DataFrame({'score':abgrid.cv_results_['mean_test_score']}, index=abgrid.cv_results_['param_adaboostclassifier__n_estimators']).plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f1cdee9ac50>




![png](https://s3.amazonaws.com/ghpage/htmp/output_97_1.png)



```python
roc_auc_score(y_test, abgrid.predict(X_test))
```




    0.96757739512200769




```python
abgrid.best_estimator_, abgrid.best_params_
```




    (Pipeline(memory=None,
          steps=[('categoricaltransformer', CategoricalTransformer(columns=['favorite', 'interested', 'status', 'property_type', 'sale_type', 'source', 'state'])), ('dummyencoder', DummyEncoder()), ('imputer', Imputer(axis=0, copy=True, missing_values='NaN', strategy='median', verbose=0)), ('standardscaler', ...'SAMME.R', base_estimator=None,
               learning_rate=1.0, n_estimators=3250, random_state=None))]),
     {'adaboostclassifier__n_estimators': 3250})



## Thank you!
