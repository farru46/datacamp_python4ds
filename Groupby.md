```python
import pandas as pd
```


```python
# Aggregation and Reduction
```


```python
titanic = pd.read_csv("./datasets/titanic_kaggle.csv")
```


```python
titanic.head()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
titanic.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
    PassengerId    891 non-null int64
    Survived       891 non-null int64
    Pclass         891 non-null int64
    Name           891 non-null object
    Sex            891 non-null object
    Age            714 non-null float64
    SibSp          891 non-null int64
    Parch          891 non-null int64
    Ticket         891 non-null object
    Fare           891 non-null float64
    Cabin          204 non-null object
    Embarked       889 non-null object
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.7+ KB



```python
titanic.groupby('Pclass').count()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
    <tr>
      <th>Pclass</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>216</td>
      <td>216</td>
      <td>216</td>
      <td>216</td>
      <td>186</td>
      <td>216</td>
      <td>216</td>
      <td>216</td>
      <td>216</td>
      <td>176</td>
      <td>214</td>
    </tr>
    <tr>
      <th>2</th>
      <td>184</td>
      <td>184</td>
      <td>184</td>
      <td>184</td>
      <td>173</td>
      <td>184</td>
      <td>184</td>
      <td>184</td>
      <td>184</td>
      <td>16</td>
      <td>184</td>
    </tr>
    <tr>
      <th>3</th>
      <td>491</td>
      <td>491</td>
      <td>491</td>
      <td>491</td>
      <td>355</td>
      <td>491</td>
      <td>491</td>
      <td>491</td>
      <td>491</td>
      <td>12</td>
      <td>491</td>
    </tr>
  </tbody>
</table>
</div>




```python
titanic.groupby(['Pclass']).count()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
    <tr>
      <th>Pclass</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>216</td>
      <td>216</td>
      <td>216</td>
      <td>216</td>
      <td>186</td>
      <td>216</td>
      <td>216</td>
      <td>216</td>
      <td>216</td>
      <td>176</td>
      <td>214</td>
    </tr>
    <tr>
      <th>2</th>
      <td>184</td>
      <td>184</td>
      <td>184</td>
      <td>184</td>
      <td>173</td>
      <td>184</td>
      <td>184</td>
      <td>184</td>
      <td>184</td>
      <td>16</td>
      <td>184</td>
    </tr>
    <tr>
      <th>3</th>
      <td>491</td>
      <td>491</td>
      <td>491</td>
      <td>491</td>
      <td>355</td>
      <td>491</td>
      <td>491</td>
      <td>491</td>
      <td>491</td>
      <td>12</td>
      <td>491</td>
    </tr>
  </tbody>
</table>
</div>




```python
life = pd.read_csv("./datasets/life_expectancy_years.csv", 
                   index_col = 'country')
```


```python
# life.drop('country', axis = 'columns', inplace=True)
```


```python
regions = pd.read_csv("./datasets/regions.csv", index_col = 'Country')
```


```python
regions.head()
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
      <th>region</th>
    </tr>
    <tr>
      <th>Country</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Afghanistan</th>
      <td>South Asia</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>Europe &amp; Central Asia</td>
    </tr>
    <tr>
      <th>Algeria</th>
      <td>Middle East &amp; North Africa</td>
    </tr>
    <tr>
      <th>Andorra</th>
      <td>Europe &amp; Central Asia</td>
    </tr>
    <tr>
      <th>Angola</th>
      <td>Sub-Saharan Africa</td>
    </tr>
  </tbody>
</table>
</div>




```python
regions['region']
```




    Country
    Afghanistan                    South Asia
    Albania             Europe & Central Asia
    Algeria        Middle East & North Africa
    Andorra             Europe & Central Asia
    Angola                 Sub-Saharan Africa
                              ...            
    Vietnam               East Asia & Pacific
    Yemen          Middle East & North Africa
    Zambia                 Sub-Saharan Africa
    Zimbabwe               Sub-Saharan Africa
    South Sudan            Sub-Saharan Africa
    Name: region, Length: 197, dtype: object




```python
life.groupby(regions['region'])['2010'].mean()
```




    region
    East Asia & Pacific           70.640741
    Europe & Central Asia         76.353061
    Latin America & Caribbean     73.250000
    Middle East & North Africa    74.933333
    North America                 80.000000
    South Asia                    68.850000
    Sub-Saharan Africa            59.483333
    Name: 2010, dtype: float64




```python
from sklearn.datasets import load_iris
```


```python
iris = load_iris()
```


```python
X = pd.DataFrame(iris.data)
Y = pd.DataFrame(iris.target)
iris = pd.concat([X,Y], axis = 1)
iris.columns = ['Sepal.Length','Sepal.Width', 'Petal.Length', 'Petal.Width', 'Species']
```


```python
iris
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
      <th>Sepal.Length</th>
      <th>Sepal.Width</th>
      <th>Petal.Length</th>
      <th>Petal.Width</th>
      <th>Species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>145</th>
      <td>6.7</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>146</th>
      <td>6.3</td>
      <td>2.5</td>
      <td>5.0</td>
      <td>1.9</td>
      <td>2</td>
    </tr>
    <tr>
      <th>147</th>
      <td>6.5</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>148</th>
      <td>6.2</td>
      <td>3.4</td>
      <td>5.4</td>
      <td>2.3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>149</th>
      <td>5.9</td>
      <td>3.0</td>
      <td>5.1</td>
      <td>1.8</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>150 rows × 5 columns</p>
</div>




```python
iris.groupby('Species')['Sepal.Length'].mean()
```




    Species
    0    5.006
    1    5.936
    2    6.588
    Name: Sepal.Length, dtype: float64




```python
sales = pd.read_csv("./datasets/sales_new")
```


```python
sales
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
      <th>state</th>
      <th>month</th>
      <th>eggs</th>
      <th>salt</th>
      <th>spam</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CA</td>
      <td>1</td>
      <td>47</td>
      <td>12.0</td>
      <td>17</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CA</td>
      <td>2</td>
      <td>110</td>
      <td>50.0</td>
      <td>31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NY</td>
      <td>1</td>
      <td>221</td>
      <td>89.0</td>
      <td>72</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NY</td>
      <td>2</td>
      <td>77</td>
      <td>87.0</td>
      <td>20</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TX</td>
      <td>1</td>
      <td>132</td>
      <td>NaN</td>
      <td>52</td>
    </tr>
    <tr>
      <th>5</th>
      <td>TX</td>
      <td>2</td>
      <td>205</td>
      <td>60.0</td>
      <td>55</td>
    </tr>
  </tbody>
</table>
</div>




```python
dict = {'bread': [139, 237,326,456], 
       'butter':[20,45,70,98],
       'city':['Austin','Dallas'] * 2 ,
       'weekday':['Sun', 'Sun', 'Mon', 'Mon']}
```


```python
sales = pd.DataFrame(dict)
```


```python
sales
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
      <th>bread</th>
      <th>butter</th>
      <th>city</th>
      <th>weekday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>139</td>
      <td>20</td>
      <td>Austin</td>
      <td>Sun</td>
    </tr>
    <tr>
      <th>1</th>
      <td>237</td>
      <td>45</td>
      <td>Dallas</td>
      <td>Sun</td>
    </tr>
    <tr>
      <th>2</th>
      <td>326</td>
      <td>70</td>
      <td>Austin</td>
      <td>Mon</td>
    </tr>
    <tr>
      <th>3</th>
      <td>456</td>
      <td>98</td>
      <td>Dallas</td>
      <td>Mon</td>
    </tr>
  </tbody>
</table>
</div>




```python
sales.groupby('city')[['bread','butter']].max()
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
      <th>bread</th>
      <th>butter</th>
    </tr>
    <tr>
      <th>city</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Austin</th>
      <td>326</td>
      <td>70</td>
    </tr>
    <tr>
      <th>Dallas</th>
      <td>456</td>
      <td>98</td>
    </tr>
  </tbody>
</table>
</div>



# Agg method


```python
sales.groupby('city')[['bread', 'butter']].agg(['sum', 'max'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">bread</th>
      <th colspan="2" halign="left">butter</th>
    </tr>
    <tr>
      <th></th>
      <th>sum</th>
      <th>max</th>
      <th>sum</th>
      <th>max</th>
    </tr>
    <tr>
      <th>city</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Austin</th>
      <td>465</td>
      <td>326</td>
      <td>90</td>
      <td>70</td>
    </tr>
    <tr>
      <th>Dallas</th>
      <td>693</td>
      <td>456</td>
      <td>143</td>
      <td>98</td>
    </tr>
  </tbody>
</table>
</div>




```python
# results in a multi level column index
```


```python
titanic = pd.read_csv("./datasets/titanic_kaggle.csv")
```


```python
titanic.head()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
# titanic.columns = titanic.columns.str.lower()
new_col_names = [x.lower() for x in titanic.columns]
titanic.columns = new_col_names
```


```python
titanic.head ()
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
      <th>passengerid</th>
      <th>survived</th>
      <th>pclass</th>
      <th>name</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>ticket</th>
      <th>fare</th>
      <th>cabin</th>
      <th>embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
by_class = titanic.groupby('pclass')
```


```python
aggregated = titanic.groupby('pclass')[['age','fare']].agg(['max', 'median'])
```


```python
aggregated
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">age</th>
      <th colspan="2" halign="left">fare</th>
    </tr>
    <tr>
      <th></th>
      <th>max</th>
      <th>median</th>
      <th>max</th>
      <th>median</th>
    </tr>
    <tr>
      <th>pclass</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>80.0</td>
      <td>37.0</td>
      <td>512.3292</td>
      <td>60.2875</td>
    </tr>
    <tr>
      <th>2</th>
      <td>70.0</td>
      <td>29.0</td>
      <td>73.5000</td>
      <td>14.2500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>74.0</td>
      <td>24.0</td>
      <td>69.5500</td>
      <td>8.0500</td>
    </tr>
  </tbody>
</table>
</div>




```python
aggregated.loc[:, ('fare', 'median')]
```




    pclass
    1    60.2875
    2    14.2500
    3     8.0500
    Name: (fare, median), dtype: float64




```python
# Aggregating on index levels/fields
```


```python
gapminder = pd.read_csv("./datasets/gapminder_r.csv", 
                       index_col = ['year','continent','country']).sort_index()
```


```python
gapminder.columns
```




    Index(['lifeExp', 'pop', 'gdpPercap'], dtype='object')




```python
gapminder
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
      <th></th>
      <th></th>
      <th>lifeExp</th>
      <th>pop</th>
      <th>gdpPercap</th>
    </tr>
    <tr>
      <th>year</th>
      <th>continent</th>
      <th>country</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">1952</th>
      <th rowspan="5" valign="top">Africa</th>
      <th>Algeria</th>
      <td>43.077</td>
      <td>9279525</td>
      <td>2449.008185</td>
    </tr>
    <tr>
      <th>Angola</th>
      <td>30.015</td>
      <td>4232095</td>
      <td>3520.610273</td>
    </tr>
    <tr>
      <th>Benin</th>
      <td>38.223</td>
      <td>1738315</td>
      <td>1062.752200</td>
    </tr>
    <tr>
      <th>Botswana</th>
      <td>47.622</td>
      <td>442308</td>
      <td>851.241141</td>
    </tr>
    <tr>
      <th>Burkina Faso</th>
      <td>31.975</td>
      <td>4469979</td>
      <td>543.255241</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">2007</th>
      <th rowspan="3" valign="top">Europe</th>
      <th>Switzerland</th>
      <td>81.701</td>
      <td>7554661</td>
      <td>37506.419070</td>
    </tr>
    <tr>
      <th>Turkey</th>
      <td>71.777</td>
      <td>71158647</td>
      <td>8458.276384</td>
    </tr>
    <tr>
      <th>United Kingdom</th>
      <td>79.425</td>
      <td>60776238</td>
      <td>33203.261280</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Oceania</th>
      <th>Australia</th>
      <td>81.235</td>
      <td>20434176</td>
      <td>34435.367440</td>
    </tr>
    <tr>
      <th>New Zealand</th>
      <td>80.204</td>
      <td>4115771</td>
      <td>25185.009110</td>
    </tr>
  </tbody>
</table>
<p>1704 rows × 3 columns</p>
</div>




```python
# gapminder = gapminder.swaplevel(0,1)
```


```python
gapminder.groupby()
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
      <th>lifeExp</th>
      <th>pop</th>
      <th>gdpPercap</th>
    </tr>
    <tr>
      <th>year</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1952</th>
      <td>142</td>
      <td>142</td>
      <td>142</td>
    </tr>
    <tr>
      <th>1957</th>
      <td>142</td>
      <td>142</td>
      <td>142</td>
    </tr>
    <tr>
      <th>1962</th>
      <td>142</td>
      <td>142</td>
      <td>142</td>
    </tr>
    <tr>
      <th>1967</th>
      <td>142</td>
      <td>142</td>
      <td>142</td>
    </tr>
    <tr>
      <th>1972</th>
      <td>142</td>
      <td>142</td>
      <td>142</td>
    </tr>
    <tr>
      <th>1977</th>
      <td>142</td>
      <td>142</td>
      <td>142</td>
    </tr>
    <tr>
      <th>1982</th>
      <td>142</td>
      <td>142</td>
      <td>142</td>
    </tr>
    <tr>
      <th>1987</th>
      <td>142</td>
      <td>142</td>
      <td>142</td>
    </tr>
    <tr>
      <th>1992</th>
      <td>142</td>
      <td>142</td>
      <td>142</td>
    </tr>
    <tr>
      <th>1997</th>
      <td>142</td>
      <td>142</td>
      <td>142</td>
    </tr>
    <tr>
      <th>2002</th>
      <td>142</td>
      <td>142</td>
      <td>142</td>
    </tr>
    <tr>
      <th>2007</th>
      <td>142</td>
      <td>142</td>
      <td>142</td>
    </tr>
  </tbody>
</table>
</div>




```python
def spread(series):
    return series.max() - series.min()
```


```python
aggregator = {'pop' : 'sum', 'lifeExp' : 'mean', 'gdpPercap' : spread}
```


```python
aggregated = gapminder.groupby(level = ['year', 'continent']).agg(aggregator)
```


```python
aggregated
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
      <th></th>
      <th>pop</th>
      <th>lifeExp</th>
      <th>gdpPercap</th>
    </tr>
    <tr>
      <th>year</th>
      <th>continent</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">1952</th>
      <th>Africa</th>
      <td>237640501</td>
      <td>39.135500</td>
      <td>4426.449319</td>
    </tr>
    <tr>
      <th>Americas</th>
      <td>345152446</td>
      <td>53.279840</td>
      <td>12592.764943</td>
    </tr>
    <tr>
      <th>Asia</th>
      <td>1395357351</td>
      <td>46.314394</td>
      <td>108051.352900</td>
    </tr>
    <tr>
      <th>Europe</th>
      <td>418120846</td>
      <td>64.408500</td>
      <td>13760.699555</td>
    </tr>
    <tr>
      <th>Oceania</th>
      <td>10686006</td>
      <td>69.255000</td>
      <td>516.980020</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">1957</th>
      <th>Africa</th>
      <td>264837738</td>
      <td>41.266346</td>
      <td>5151.107104</td>
    </tr>
    <tr>
      <th>Americas</th>
      <td>386953916</td>
      <td>55.960280</td>
      <td>13302.724125</td>
    </tr>
    <tr>
      <th>Asia</th>
      <td>1562780599</td>
      <td>49.318544</td>
      <td>113173.132900</td>
    </tr>
    <tr>
      <th>Europe</th>
      <td>437890351</td>
      <td>66.703067</td>
      <td>16555.500554</td>
    </tr>
    <tr>
      <th>Oceania</th>
      <td>11941976</td>
      <td>70.295000</td>
      <td>1297.745730</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">1962</th>
      <th>Africa</th>
      <td>296516865</td>
      <td>43.319442</td>
      <td>6401.827589</td>
    </tr>
    <tr>
      <th>Americas</th>
      <td>433270254</td>
      <td>58.398760</td>
      <td>14511.008501</td>
    </tr>
    <tr>
      <th>Asia</th>
      <td>1696357182</td>
      <td>51.563223</td>
      <td>95070.111760</td>
    </tr>
    <tr>
      <th>Europe</th>
      <td>460355155</td>
      <td>68.539233</td>
      <td>18721.409021</td>
    </tr>
    <tr>
      <th>Oceania</th>
      <td>13283518</td>
      <td>71.085000</td>
      <td>958.451140</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">1967</th>
      <th>Africa</th>
      <td>335289489</td>
      <td>45.334538</td>
      <td>18359.774176</td>
    </tr>
    <tr>
      <th>Americas</th>
      <td>480746623</td>
      <td>60.410920</td>
      <td>18078.307904</td>
    </tr>
    <tr>
      <th>Asia</th>
      <td>1905662900</td>
      <td>54.663640</td>
      <td>80545.883260</td>
    </tr>
    <tr>
      <th>Europe</th>
      <td>481178958</td>
      <td>69.737600</td>
      <td>20793.791897</td>
    </tr>
    <tr>
      <th>Oceania</th>
      <td>14600414</td>
      <td>71.310000</td>
      <td>62.205720</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">1972</th>
      <th>Africa</th>
      <td>379879541</td>
      <td>47.450942</td>
      <td>20547.397706</td>
    </tr>
    <tr>
      <th>Americas</th>
      <td>529384210</td>
      <td>62.394920</td>
      <td>20151.578994</td>
    </tr>
    <tr>
      <th>Asia</th>
      <td>2150972248</td>
      <td>57.319269</td>
      <td>108990.867000</td>
    </tr>
    <tr>
      <th>Europe</th>
      <td>500635059</td>
      <td>70.775033</td>
      <td>24334.943290</td>
    </tr>
    <tr>
      <th>Oceania</th>
      <td>16106100</td>
      <td>71.910000</td>
      <td>742.592200</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">1977</th>
      <th>Africa</th>
      <td>433061021</td>
      <td>49.580423</td>
      <td>21448.892027</td>
    </tr>
    <tr>
      <th>Americas</th>
      <td>578067699</td>
      <td>64.391560</td>
      <td>22198.333199</td>
    </tr>
    <tr>
      <th>Asia</th>
      <td>2384513556</td>
      <td>59.610556</td>
      <td>58894.477140</td>
    </tr>
    <tr>
      <th>Europe</th>
      <td>517164531</td>
      <td>71.937767</td>
      <td>23453.809215</td>
    </tr>
    <tr>
      <th>Oceania</th>
      <td>17239000</td>
      <td>72.855000</td>
      <td>2100.479810</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">1982</th>
      <th>Africa</th>
      <td>499348587</td>
      <td>51.592865</td>
      <td>16902.063965</td>
    </tr>
    <tr>
      <th>Americas</th>
      <td>630290920</td>
      <td>66.228840</td>
      <td>22998.399591</td>
    </tr>
    <tr>
      <th>Asia</th>
      <td>2610135582</td>
      <td>62.617939</td>
      <td>33269.175250</td>
    </tr>
    <tr>
      <th>Europe</th>
      <td>531266901</td>
      <td>72.806400</td>
      <td>24766.834398</td>
    </tr>
    <tr>
      <th>Oceania</th>
      <td>18394850</td>
      <td>74.290000</td>
      <td>1844.598880</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">1987</th>
      <th>Africa</th>
      <td>574834110</td>
      <td>53.344788</td>
      <td>11474.532255</td>
    </tr>
    <tr>
      <th>Americas</th>
      <td>682753971</td>
      <td>68.090720</td>
      <td>28061.334415</td>
    </tr>
    <tr>
      <th>Asia</th>
      <td>2871220762</td>
      <td>64.851182</td>
      <td>27733.429980</td>
    </tr>
    <tr>
      <th>Europe</th>
      <td>543094160</td>
      <td>73.642167</td>
      <td>27802.042065</td>
    </tr>
    <tr>
      <th>Oceania</th>
      <td>19574415</td>
      <td>75.320000</td>
      <td>2881.697740</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">1992</th>
      <th>Africa</th>
      <td>659081517</td>
      <td>53.629577</td>
      <td>13111.260696</td>
    </tr>
    <tr>
      <th>Americas</th>
      <td>739274104</td>
      <td>69.568360</td>
      <td>30547.622723</td>
    </tr>
    <tr>
      <th>Asia</th>
      <td>3133292191</td>
      <td>66.537212</td>
      <td>34585.919590</td>
    </tr>
    <tr>
      <th>Europe</th>
      <td>558142797</td>
      <td>74.440100</td>
      <td>31468.223249</td>
    </tr>
    <tr>
      <th>Oceania</th>
      <td>20919651</td>
      <td>76.945000</td>
      <td>5061.441890</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">1997</th>
      <th>Africa</th>
      <td>743832984</td>
      <td>53.598269</td>
      <td>14410.653457</td>
    </tr>
    <tr>
      <th>Americas</th>
      <td>796900410</td>
      <td>71.150480</td>
      <td>34425.706099</td>
    </tr>
    <tr>
      <th>Asia</th>
      <td>3383285500</td>
      <td>68.020515</td>
      <td>39885.619960</td>
    </tr>
    <tr>
      <th>Europe</th>
      <td>568944148</td>
      <td>75.505167</td>
      <td>38090.109726</td>
    </tr>
    <tr>
      <th>Oceania</th>
      <td>22241430</td>
      <td>78.190000</td>
      <td>5947.522800</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">2002</th>
      <th>Africa</th>
      <td>833723916</td>
      <td>53.325231</td>
      <td>12280.548044</td>
    </tr>
    <tr>
      <th>Americas</th>
      <td>849772762</td>
      <td>72.422040</td>
      <td>37826.734618</td>
    </tr>
    <tr>
      <th>Asia</th>
      <td>3601802203</td>
      <td>69.233879</td>
      <td>35412.105400</td>
    </tr>
    <tr>
      <th>Europe</th>
      <td>578223869</td>
      <td>76.700600</td>
      <td>40079.763513</td>
    </tr>
    <tr>
      <th>Oceania</th>
      <td>23454829</td>
      <td>79.740000</td>
      <td>7497.953380</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">2007</th>
      <th>Africa</th>
      <td>929539692</td>
      <td>54.806038</td>
      <td>12928.932661</td>
    </tr>
    <tr>
      <th>Americas</th>
      <td>898871184</td>
      <td>73.608120</td>
      <td>41750.015936</td>
    </tr>
    <tr>
      <th>Asia</th>
      <td>3811953827</td>
      <td>70.728485</td>
      <td>46362.989780</td>
    </tr>
    <tr>
      <th>Europe</th>
      <td>586098529</td>
      <td>77.648600</td>
      <td>43420.160644</td>
    </tr>
    <tr>
      <th>Oceania</th>
      <td>24549947</td>
      <td>80.719500</td>
      <td>9250.358330</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Grouping a function of the index
```


```python
sales = pd.read_csv("./datasets/sales_data_companies_r.csv",
                   index_col = 'dt', parse_dates=True)
```


```python
sales
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
      <th>dt</th>
      <th>Company</th>
      <th>Product</th>
      <th>Units</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-02-02T08:30:00Z</td>
      <td>Hooli</td>
      <td>Software</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-02-02T21:00:00Z</td>
      <td>Mediacore</td>
      <td>Hardware</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-02-03T14:00:00Z</td>
      <td>Initech</td>
      <td>Software</td>
      <td>13</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015-02-04T15:30:00Z</td>
      <td>Streeplex</td>
      <td>Software</td>
      <td>13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-02-04T22:00:00Z</td>
      <td>Acme</td>
      <td>Coporation Hardware</td>
      <td>14</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2015-02-05T02:00:00Z</td>
      <td>Acme</td>
      <td>Coporation Software</td>
      <td>19</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2015-02-05T22:00:00Z</td>
      <td>Hooli</td>
      <td>Service</td>
      <td>10</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2015-02-07T23:00:00Z</td>
      <td>Acme</td>
      <td>Coporation Hardware</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2015-02-09T09:00:00Z</td>
      <td>Streeplex</td>
      <td>Service</td>
      <td>19</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2015-02-09T13:00:00Z</td>
      <td>Mediacore</td>
      <td>Software</td>
      <td>7</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2015-02-11T20:00:00Z</td>
      <td>Initech</td>
      <td>Software</td>
      <td>7</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2015-02-11T23:00:00Z</td>
      <td>Hooli</td>
      <td>Software</td>
      <td>4</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2015-02-16T12:00:00Z</td>
      <td>Hooli</td>
      <td>Software</td>
      <td>10</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2015-02-19T11:00:00Z</td>
      <td>Mediacore</td>
      <td>Hardware</td>
      <td>16</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2015-02-19T16:00:00Z</td>
      <td>Mediacore</td>
      <td>Service</td>
      <td>10</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2015-02-21T05:00:00Z</td>
      <td>Mediacore</td>
      <td>Software</td>
      <td>3</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2015-02-21T20:30:00Z</td>
      <td>Hooli</td>
      <td>Hardware</td>
      <td>3</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2015-02-25T00:30:00Z</td>
      <td>Initech</td>
      <td>Service</td>
      <td>10</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2015-02-26T09:00:00Z</td>
      <td>Streeplex</td>
      <td>Service</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
sales.index.strftime('%a')
```




    Index(['Mon', 'Mon', 'Tue', 'Wed', 'Wed', 'Thu', 'Thu', 'Sat', 'Mon', 'Mon',
           'Wed', 'Wed', 'Mon', 'Thu', 'Thu', 'Sat', 'Sat', 'Wed', 'Thu'],
          dtype='object')




```python
# Day wise sales
```


```python
sales.groupby(sales.index.strftime("%a"))['Units'].sum()
```




    Mon    48
    Sat     7
    Thu    59
    Tue    13
    Wed    48
    Name: Units, dtype: int64



# Groupby and Transformation


```python
colnames = ["mpg",
"cylinders",
"displacement",
"hp",
"weight",
"acceleration",
"modelyear",
"origin",
"carname"]
df_auto = pd.read_csv("datasets/auto-mpg.csv", names = colnames, na_values='?')
```


```python
df_auto
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
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>hp</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>modelyear</th>
      <th>origin</th>
      <th>carname</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>8.0</td>
      <td>307.0</td>
      <td>130.0</td>
      <td>3504.0</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>chevrolet chevelle malibu</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>8.0</td>
      <td>350.0</td>
      <td>165.0</td>
      <td>3693.0</td>
      <td>11.5</td>
      <td>70</td>
      <td>1</td>
      <td>buick skylark 320</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>8.0</td>
      <td>318.0</td>
      <td>150.0</td>
      <td>3436.0</td>
      <td>11.0</td>
      <td>70</td>
      <td>1</td>
      <td>plymouth satellite</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.0</td>
      <td>8.0</td>
      <td>304.0</td>
      <td>150.0</td>
      <td>3433.0</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>amc rebel sst</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.0</td>
      <td>8.0</td>
      <td>302.0</td>
      <td>140.0</td>
      <td>3449.0</td>
      <td>10.5</td>
      <td>70</td>
      <td>1</td>
      <td>ford torino</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>393</th>
      <td>27.0</td>
      <td>4.0</td>
      <td>140.0</td>
      <td>86.0</td>
      <td>2790.0</td>
      <td>15.6</td>
      <td>82</td>
      <td>1</td>
      <td>ford mustang gl</td>
    </tr>
    <tr>
      <th>394</th>
      <td>44.0</td>
      <td>4.0</td>
      <td>97.0</td>
      <td>52.0</td>
      <td>2130.0</td>
      <td>24.6</td>
      <td>82</td>
      <td>2</td>
      <td>vw pickup</td>
    </tr>
    <tr>
      <th>395</th>
      <td>32.0</td>
      <td>4.0</td>
      <td>135.0</td>
      <td>84.0</td>
      <td>2295.0</td>
      <td>11.6</td>
      <td>82</td>
      <td>1</td>
      <td>dodge rampage</td>
    </tr>
    <tr>
      <th>396</th>
      <td>28.0</td>
      <td>4.0</td>
      <td>120.0</td>
      <td>79.0</td>
      <td>2625.0</td>
      <td>18.6</td>
      <td>82</td>
      <td>1</td>
      <td>ford ranger</td>
    </tr>
    <tr>
      <th>397</th>
      <td>31.0</td>
      <td>4.0</td>
      <td>119.0</td>
      <td>82.0</td>
      <td>2720.0</td>
      <td>19.4</td>
      <td>82</td>
      <td>1</td>
      <td>chevy s-10</td>
    </tr>
  </tbody>
</table>
<p>398 rows × 9 columns</p>
</div>




```python
from scipy.stats import zscore
```


```python
zscore(df_auto['mpg'])[0:5]
```




    array([-0.70649818, -1.09082846, -0.70649818, -0.96271837, -0.83460828])




```python
df_auto.groupby('modelyear')['mpg'].transform(lambda x : (x - x.mean()) / x.std())
```




    0      0.000000
    1     -0.580948
    2      0.000000
    3     -0.387298
    4     -0.193649
             ...   
    393   -0.873368
    394    2.279131
    395    0.053838
    396   -0.687927
    397   -0.131603
    Name: mpg, Length: 398, dtype: float64




```python
df_auto.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 398 entries, 0 to 397
    Data columns (total 9 columns):
    mpg             398 non-null float64
    cylinders       398 non-null float64
    displacement    398 non-null float64
    hp              392 non-null float64
    weight          398 non-null float64
    acceleration    398 non-null float64
    modelyear       398 non-null int64
    origin          398 non-null object
    carname         397 non-null object
    dtypes: float64(6), int64(1), object(2)
    memory usage: 28.1+ KB



```python
from scipy.stats import zscore
```


```python
gapminder = pd.read_csv("./datasets/gapminder_r.csv",index_col = 'country')
```


```python
gapminder
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
      <th>continent</th>
      <th>year</th>
      <th>lifeExp</th>
      <th>pop</th>
      <th>gdpPercap</th>
    </tr>
    <tr>
      <th>country</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Afghanistan</th>
      <td>Asia</td>
      <td>1952</td>
      <td>28.801</td>
      <td>8425333</td>
      <td>779.445314</td>
    </tr>
    <tr>
      <th>Afghanistan</th>
      <td>Asia</td>
      <td>1957</td>
      <td>30.332</td>
      <td>9240934</td>
      <td>820.853030</td>
    </tr>
    <tr>
      <th>Afghanistan</th>
      <td>Asia</td>
      <td>1962</td>
      <td>31.997</td>
      <td>10267083</td>
      <td>853.100710</td>
    </tr>
    <tr>
      <th>Afghanistan</th>
      <td>Asia</td>
      <td>1967</td>
      <td>34.020</td>
      <td>11537966</td>
      <td>836.197138</td>
    </tr>
    <tr>
      <th>Afghanistan</th>
      <td>Asia</td>
      <td>1972</td>
      <td>36.088</td>
      <td>13079460</td>
      <td>739.981106</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Zimbabwe</th>
      <td>Africa</td>
      <td>1987</td>
      <td>62.351</td>
      <td>9216418</td>
      <td>706.157306</td>
    </tr>
    <tr>
      <th>Zimbabwe</th>
      <td>Africa</td>
      <td>1992</td>
      <td>60.377</td>
      <td>10704340</td>
      <td>693.420786</td>
    </tr>
    <tr>
      <th>Zimbabwe</th>
      <td>Africa</td>
      <td>1997</td>
      <td>46.809</td>
      <td>11404948</td>
      <td>792.449960</td>
    </tr>
    <tr>
      <th>Zimbabwe</th>
      <td>Africa</td>
      <td>2002</td>
      <td>39.989</td>
      <td>11926563</td>
      <td>672.038623</td>
    </tr>
    <tr>
      <th>Zimbabwe</th>
      <td>Africa</td>
      <td>2007</td>
      <td>43.487</td>
      <td>12311143</td>
      <td>469.709298</td>
    </tr>
  </tbody>
</table>
<p>1704 rows × 5 columns</p>
</div>




```python
gapminder.groupby('continent')[['lifeExp', 'pop']].transform(zscore)
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
      <th>year</th>
      <th>lifeExp</th>
      <th>pop</th>
      <th>gdpPercap</th>
    </tr>
    <tr>
      <th>country</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Afghanistan</th>
      <td>-1.593255</td>
      <td>-2.638406</td>
      <td>-0.332069</td>
      <td>-0.507763</td>
    </tr>
    <tr>
      <th>Afghanistan</th>
      <td>-1.303572</td>
      <td>-2.509203</td>
      <td>-0.328122</td>
      <td>-0.504811</td>
    </tr>
    <tr>
      <th>Afghanistan</th>
      <td>-1.013890</td>
      <td>-2.368691</td>
      <td>-0.323156</td>
      <td>-0.502512</td>
    </tr>
    <tr>
      <th>Afghanistan</th>
      <td>-0.724207</td>
      <td>-2.197967</td>
      <td>-0.317005</td>
      <td>-0.503717</td>
    </tr>
    <tr>
      <th>Afghanistan</th>
      <td>-0.434524</td>
      <td>-2.023446</td>
      <td>-0.309544</td>
      <td>-0.510576</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Zimbabwe</th>
      <td>0.434524</td>
      <td>1.474992</td>
      <td>-0.045197</td>
      <td>-0.526460</td>
    </tr>
    <tr>
      <th>Zimbabwe</th>
      <td>0.724207</td>
      <td>1.259086</td>
      <td>0.050931</td>
      <td>-0.530967</td>
    </tr>
    <tr>
      <th>Zimbabwe</th>
      <td>1.013890</td>
      <td>-0.224911</td>
      <td>0.096194</td>
      <td>-0.495921</td>
    </tr>
    <tr>
      <th>Zimbabwe</th>
      <td>1.303572</td>
      <td>-0.970847</td>
      <td>0.129894</td>
      <td>-0.538534</td>
    </tr>
    <tr>
      <th>Zimbabwe</th>
      <td>1.593255</td>
      <td>-0.588254</td>
      <td>0.154740</td>
      <td>-0.610138</td>
    </tr>
  </tbody>
</table>
<p>1704 rows × 4 columns</p>
</div>




```python
titanic = pd.read_csv("./datasets/titanic_kaggle.csv")
```


```python
titanic.tail()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>886</th>
      <td>887</td>
      <td>0</td>
      <td>2</td>
      <td>Montvila, Rev. Juozas</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>211536</td>
      <td>13.00</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>887</th>
      <td>888</td>
      <td>1</td>
      <td>1</td>
      <td>Graham, Miss. Margaret Edith</td>
      <td>female</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>112053</td>
      <td>30.00</td>
      <td>B42</td>
      <td>S</td>
    </tr>
    <tr>
      <th>888</th>
      <td>889</td>
      <td>0</td>
      <td>3</td>
      <td>Johnston, Miss. Catherine Helen "Carrie"</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>W./C. 6607</td>
      <td>23.45</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>889</th>
      <td>890</td>
      <td>1</td>
      <td>1</td>
      <td>Behr, Mr. Karl Howell</td>
      <td>male</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>111369</td>
      <td>30.00</td>
      <td>C148</td>
      <td>C</td>
    </tr>
    <tr>
      <th>890</th>
      <td>891</td>
      <td>0</td>
      <td>3</td>
      <td>Dooley, Mr. Patrick</td>
      <td>male</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>370376</td>
      <td>7.75</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
  </tbody>
</table>
</div>




```python
titanic.groupby(['Pclass', 'Sex'])['Age'].median()
```




    Pclass  Sex   
    1       female    35.0
            male      40.0
    2       female    28.0
            male      30.0
    3       female    21.5
            male      25.0
    Name: Age, dtype: float64




```python
# Imputing median values for age groupwise
def impute_median(series):
    return series.fillna(series.median())
```


```python
titanic.columns = [x.lower() for x in titanic.columns]
```


```python
imputed_age = titanic.groupby(['sex', 'pclass'])['age'].transform(impute_median)
```


```python
titanic['age'].tail()
```




    886    27.0
    887    19.0
    888     NaN
    889    26.0
    890    32.0
    Name: age, dtype: float64




```python
imputed_age.tail()
```




    886    27.0
    887    19.0
    888    21.5
    889    26.0
    890    32.0
    Name: age, dtype: float64



# .Apply


```python
# Gapminder dataset
```


```python
gapminder = pd.read_csv("./datasets/gapminder_r.csv", index_col = 'country')
```


```python
gapminder['gdp'] = gapminder['gdpPercap']
```


```python
regional = gapminder.groupby('continent')
```


```python
def disparity(gr):
    # Compute the spread of gr['gdp']: s
    s = gr['gdp'].max() - gr['gdp'].min()
    # Compute the z-score of gr['gdp'] as (gr['gdp']-gr['gdp'].mean())/gr['gdp'].std(): z
    z = (gr['gdp'] - gr['gdp'].mean())/gr['gdp'].std()
    # Return a DataFrame with the inputs {'z(gdp)':z, 'regional spread(gdp)':s}
    return pd.DataFrame({'z(gdp)':z , 'regional spread(gdp)':s})
```


```python
reg_disp = regional.apply(disparity)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    ~/repos_faraz/datacamp_python4ds/py36_env/lib/python3.6/site-packages/pandas/core/groupby/groupby.py in apply(self, func, *args, **kwargs)
        724             try:
    --> 725                 result = self._python_apply_general(f)
        726             except Exception:


    ~/repos_faraz/datacamp_python4ds/py36_env/lib/python3.6/site-packages/pandas/core/groupby/groupby.py in _python_apply_general(self, f)
        744         return self._wrap_applied_output(
    --> 745             keys, values, not_indexed_same=mutated or self.mutated
        746         )


    ~/repos_faraz/datacamp_python4ds/py36_env/lib/python3.6/site-packages/pandas/core/groupby/generic.py in _wrap_applied_output(self, keys, values, not_indexed_same)
        371         elif isinstance(v, DataFrame):
    --> 372             return self._concat_objects(keys, values, not_indexed_same=not_indexed_same)
        373         elif self.grouper.groupings is not None:


    ~/repos_faraz/datacamp_python4ds/py36_env/lib/python3.6/site-packages/pandas/core/groupby/groupby.py in _concat_objects(self, keys, values, not_indexed_same)
        954                 else:
    --> 955                     result = result.reindex(ax, axis=self.axis)
        956 


    ~/repos_faraz/datacamp_python4ds/py36_env/lib/python3.6/site-packages/pandas/util/_decorators.py in wrapper(*args, **kwargs)
        220         def wrapper(*args, **kwargs):
    --> 221             return func(*args, **kwargs)
        222 


    ~/repos_faraz/datacamp_python4ds/py36_env/lib/python3.6/site-packages/pandas/core/frame.py in reindex(self, *args, **kwargs)
       3975         kwargs.pop("labels", None)
    -> 3976         return super().reindex(**kwargs)
       3977 


    ~/repos_faraz/datacamp_python4ds/py36_env/lib/python3.6/site-packages/pandas/core/generic.py in reindex(self, *args, **kwargs)
       4513         return self._reindex_axes(
    -> 4514             axes, level, limit, tolerance, method, fill_value, copy
       4515         ).__finalize__(self)


    ~/repos_faraz/datacamp_python4ds/py36_env/lib/python3.6/site-packages/pandas/core/frame.py in _reindex_axes(self, axes, level, limit, tolerance, method, fill_value, copy)
       3863             frame = frame._reindex_index(
    -> 3864                 index, method, copy, level, fill_value, limit, tolerance
       3865             )


    ~/repos_faraz/datacamp_python4ds/py36_env/lib/python3.6/site-packages/pandas/core/frame.py in _reindex_index(self, new_index, method, copy, level, fill_value, limit, tolerance)
       3885             fill_value=fill_value,
    -> 3886             allow_dups=False,
       3887         )


    ~/repos_faraz/datacamp_python4ds/py36_env/lib/python3.6/site-packages/pandas/core/generic.py in _reindex_with_indexers(self, reindexers, fill_value, copy, allow_dups)
       4576                 allow_dups=allow_dups,
    -> 4577                 copy=copy,
       4578             )


    ~/repos_faraz/datacamp_python4ds/py36_env/lib/python3.6/site-packages/pandas/core/internals/managers.py in reindex_indexer(self, new_axis, indexer, axis, fill_value, allow_dups, copy)
       1250         if not allow_dups:
    -> 1251             self.axes[axis]._can_reindex(indexer)
       1252 


    ~/repos_faraz/datacamp_python4ds/py36_env/lib/python3.6/site-packages/pandas/core/indexes/base.py in _can_reindex(self, indexer)
       3361         if not self.is_unique and len(indexer):
    -> 3362             raise ValueError("cannot reindex from a duplicate axis")
       3363 


    ValueError: cannot reindex from a duplicate axis

    
    During handling of the above exception, another exception occurred:


    ValueError                                Traceback (most recent call last)

    <ipython-input-225-74e3e563a526> in <module>
    ----> 1 reg_disp = regional.apply(disparity)
    

    ~/repos_faraz/datacamp_python4ds/py36_env/lib/python3.6/site-packages/pandas/core/groupby/groupby.py in apply(self, func, *args, **kwargs)
        735 
        736                 with _group_selection_context(self):
    --> 737                     return self._python_apply_general(f)
        738 
        739         return result


    ~/repos_faraz/datacamp_python4ds/py36_env/lib/python3.6/site-packages/pandas/core/groupby/groupby.py in _python_apply_general(self, f)
        743 
        744         return self._wrap_applied_output(
    --> 745             keys, values, not_indexed_same=mutated or self.mutated
        746         )
        747 


    ~/repos_faraz/datacamp_python4ds/py36_env/lib/python3.6/site-packages/pandas/core/groupby/generic.py in _wrap_applied_output(self, keys, values, not_indexed_same)
        370             return DataFrame()
        371         elif isinstance(v, DataFrame):
    --> 372             return self._concat_objects(keys, values, not_indexed_same=not_indexed_same)
        373         elif self.grouper.groupings is not None:
        374             if len(self.grouper.groupings) > 1:


    ~/repos_faraz/datacamp_python4ds/py36_env/lib/python3.6/site-packages/pandas/core/groupby/groupby.py in _concat_objects(self, keys, values, not_indexed_same)
        953                     result = result.take(indexer, axis=self.axis)
        954                 else:
    --> 955                     result = result.reindex(ax, axis=self.axis)
        956 
        957         elif self.group_keys:


    ~/repos_faraz/datacamp_python4ds/py36_env/lib/python3.6/site-packages/pandas/util/_decorators.py in wrapper(*args, **kwargs)
        219         @wraps(func)
        220         def wrapper(*args, **kwargs):
    --> 221             return func(*args, **kwargs)
        222 
        223         kind = inspect.Parameter.POSITIONAL_OR_KEYWORD


    ~/repos_faraz/datacamp_python4ds/py36_env/lib/python3.6/site-packages/pandas/core/frame.py in reindex(self, *args, **kwargs)
       3974         kwargs.pop("axis", None)
       3975         kwargs.pop("labels", None)
    -> 3976         return super().reindex(**kwargs)
       3977 
       3978     def drop(


    ~/repos_faraz/datacamp_python4ds/py36_env/lib/python3.6/site-packages/pandas/core/generic.py in reindex(self, *args, **kwargs)
       4512         # perform the reindex on the axes
       4513         return self._reindex_axes(
    -> 4514             axes, level, limit, tolerance, method, fill_value, copy
       4515         ).__finalize__(self)
       4516 


    ~/repos_faraz/datacamp_python4ds/py36_env/lib/python3.6/site-packages/pandas/core/frame.py in _reindex_axes(self, axes, level, limit, tolerance, method, fill_value, copy)
       3862         if index is not None:
       3863             frame = frame._reindex_index(
    -> 3864                 index, method, copy, level, fill_value, limit, tolerance
       3865             )
       3866 


    ~/repos_faraz/datacamp_python4ds/py36_env/lib/python3.6/site-packages/pandas/core/frame.py in _reindex_index(self, new_index, method, copy, level, fill_value, limit, tolerance)
       3884             copy=copy,
       3885             fill_value=fill_value,
    -> 3886             allow_dups=False,
       3887         )
       3888 


    ~/repos_faraz/datacamp_python4ds/py36_env/lib/python3.6/site-packages/pandas/core/generic.py in _reindex_with_indexers(self, reindexers, fill_value, copy, allow_dups)
       4575                 fill_value=fill_value,
       4576                 allow_dups=allow_dups,
    -> 4577                 copy=copy,
       4578             )
       4579 


    ~/repos_faraz/datacamp_python4ds/py36_env/lib/python3.6/site-packages/pandas/core/internals/managers.py in reindex_indexer(self, new_axis, indexer, axis, fill_value, allow_dups, copy)
       1249         # some axes don't allow reindexing with dups
       1250         if not allow_dups:
    -> 1251             self.axes[axis]._can_reindex(indexer)
       1252 
       1253         if axis >= self.ndim:


    ~/repos_faraz/datacamp_python4ds/py36_env/lib/python3.6/site-packages/pandas/core/indexes/base.py in _can_reindex(self, indexer)
       3360         # trying to reindex on an axis with duplicates
       3361         if not self.is_unique and len(indexer):
    -> 3362             raise ValueError("cannot reindex from a duplicate axis")
       3363 
       3364     def reindex(self, target, method=None, level=None, limit=None, tolerance=None):


    ValueError: cannot reindex from a duplicate axis



```python
# Group by and filtering
```


```python
titanic = pd.read_csv("./datasets/titanic_kaggle.csv")
titanic.columns = [x.lower() for x in titanic.columns]
```


```python
titanic
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
      <th>passengerid</th>
      <th>survived</th>
      <th>pclass</th>
      <th>name</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>ticket</th>
      <th>fare</th>
      <th>cabin</th>
      <th>embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>886</th>
      <td>887</td>
      <td>0</td>
      <td>2</td>
      <td>Montvila, Rev. Juozas</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>211536</td>
      <td>13.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>887</th>
      <td>888</td>
      <td>1</td>
      <td>1</td>
      <td>Graham, Miss. Margaret Edith</td>
      <td>female</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>112053</td>
      <td>30.0000</td>
      <td>B42</td>
      <td>S</td>
    </tr>
    <tr>
      <th>888</th>
      <td>889</td>
      <td>0</td>
      <td>3</td>
      <td>Johnston, Miss. Catherine Helen "Carrie"</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>W./C. 6607</td>
      <td>23.4500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>889</th>
      <td>890</td>
      <td>1</td>
      <td>1</td>
      <td>Behr, Mr. Karl Howell</td>
      <td>male</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>111369</td>
      <td>30.0000</td>
      <td>C148</td>
      <td>C</td>
    </tr>
    <tr>
      <th>890</th>
      <td>891</td>
      <td>0</td>
      <td>3</td>
      <td>Dooley, Mr. Patrick</td>
      <td>male</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>370376</td>
      <td>7.7500</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 12 columns</p>
</div>




```python
by_sex = titanic.groupby("sex")
```


```python
def c_deck_survival(gr):

    c_passengers = gr['cabin'].str.startswith('C').fillna(False)

    return gr.loc[c_passengers, 'survived'].mean()
```


```python
c_surv_by_sex = by_sex.apply(c_deck_survival)
```


```python
print(c_surv_by_sex)
```

    sex
    female    0.888889
    male      0.343750
    dtype: float64



```python
sales = pd.read_csv("./datasets/sales_data_companies_r.csv")
```


```python
sales
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
      <th>dt</th>
      <th>Company</th>
      <th>Product</th>
      <th>Units</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-02-02T08:30:00Z</td>
      <td>Hooli</td>
      <td>Software</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-02-02T21:00:00Z</td>
      <td>Mediacore</td>
      <td>Hardware</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-02-03T14:00:00Z</td>
      <td>Initech</td>
      <td>Software</td>
      <td>13</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015-02-04T15:30:00Z</td>
      <td>Streeplex</td>
      <td>Software</td>
      <td>13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-02-04T22:00:00Z</td>
      <td>Acme</td>
      <td>Coporation Hardware</td>
      <td>14</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2015-02-05T02:00:00Z</td>
      <td>Acme</td>
      <td>Coporation Software</td>
      <td>19</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2015-02-05T22:00:00Z</td>
      <td>Hooli</td>
      <td>Service</td>
      <td>10</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2015-02-07T23:00:00Z</td>
      <td>Acme</td>
      <td>Coporation Hardware</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2015-02-09T09:00:00Z</td>
      <td>Streeplex</td>
      <td>Service</td>
      <td>19</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2015-02-09T13:00:00Z</td>
      <td>Mediacore</td>
      <td>Software</td>
      <td>7</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2015-02-11T20:00:00Z</td>
      <td>Initech</td>
      <td>Software</td>
      <td>7</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2015-02-11T23:00:00Z</td>
      <td>Hooli</td>
      <td>Software</td>
      <td>4</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2015-02-16T12:00:00Z</td>
      <td>Hooli</td>
      <td>Software</td>
      <td>10</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2015-02-19T11:00:00Z</td>
      <td>Mediacore</td>
      <td>Hardware</td>
      <td>16</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2015-02-19T16:00:00Z</td>
      <td>Mediacore</td>
      <td>Service</td>
      <td>10</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2015-02-21T05:00:00Z</td>
      <td>Mediacore</td>
      <td>Software</td>
      <td>3</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2015-02-21T20:30:00Z</td>
      <td>Hooli</td>
      <td>Hardware</td>
      <td>3</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2015-02-25T00:30:00Z</td>
      <td>Initech</td>
      <td>Service</td>
      <td>10</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2015-02-26T09:00:00Z</td>
      <td>Streeplex</td>
      <td>Service</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Read the CSV file into a DataFrame: sales
sales = pd.read_csv('./datasets/sales_data_companies_r.csv', index_col='dt', parse_dates=True)

# Group sales by 'Company': by_company
by_company = sales.groupby('Company')

# Compute the sum of the 'Units' of by_company: by_com_sum
by_com_sum = by_company['Units'].sum()
print(by_com_sum)

# Filter 'Units' where the sum is > 35: by_com_filt
by_com_filt = by_company.filter(lambda g:g['Units'].sum() > 35)
print(by_com_filt)
```

    Company
    Acme         34
    Hooli        30
    Initech      30
    Mediacore    45
    Streeplex    36
    Name: Units, dtype: int64
                                 Company   Product  Units
    dt                                                   
    2015-02-02 21:00:00+00:00  Mediacore  Hardware      9
    2015-02-04 15:30:00+00:00  Streeplex  Software     13
    2015-02-09 09:00:00+00:00  Streeplex   Service     19
    2015-02-09 13:00:00+00:00  Mediacore  Software      7
    2015-02-19 11:00:00+00:00  Mediacore  Hardware     16
    2015-02-19 16:00:00+00:00  Mediacore   Service     10
    2015-02-21 05:00:00+00:00  Mediacore  Software      3
    2015-02-26 09:00:00+00:00  Streeplex   Service      4



```python
titanic = pd.read_csv("./datasets/titanic_kaggle.csv")
```


```python
titanic.columns = [x.lower() for x in titanic.columns]
```


```python
under10 = (titanic['age'] < 10).map({True:'under 10', False:'over 10'})
```


```python
under10.value_counts()
```




    over 10     829
    under 10     62
    Name: age, dtype: int64




```python
survived_mean_1 = titanic.groupby(under10)['survived'].mean()
print(survived_mean_1)
```

    age
    over 10     0.366707
    under 10    0.612903
    Name: survived, dtype: float64



```python
survived_mean_2 = titanic.groupby([under10, 'pclass'])['survived'].mean()
print(survived_mean_2)
```

    age       pclass
    over 10   1         0.629108
              2         0.419162
              3         0.222717
    under 10  1         0.666667
              2         1.000000
              3         0.452381
    Name: survived, dtype: float64



```python
users=pd.read_csv("./datasets/users.csv")
```


```python
users
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
      <th>weekday</th>
      <th>city</th>
      <th>visitors</th>
      <th>signups</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sun</td>
      <td>Austin</td>
      <td>139</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Sun</td>
      <td>Dallas</td>
      <td>237</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mon</td>
      <td>Austin</td>
      <td>326</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Mon</td>
      <td>Dallas</td>
      <td>456</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
