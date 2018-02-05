---
title:  Central VA traffic analysis
tags:
  - analysis
  - tweets
  - simple
---

This is a simple analysis of Tweets from @511centralva to analyze traffic conditions in the Central VA area. The is the first attempt without any NLP and utilizes regex to parse tweets.

The aim is to obtain accident prone zones and times during the day.

<!--more-->
```python
import tweepy
import pandas as pd
pd.set_option('display.max_colwidth', 146)
from matplotlib import pyplot as plt
from datetime import datetime as dt
import pickle
```


```python
at = lambda :dt.now().strftime("%Y%m%d%H%M")
at()
```




    '201711161853'




```python
consumer_key = 'YOUR CONSUMER KEY HERE'
consumer_secret = 'YOUR CONSUMER SECRET HERE' 
access_token = 'YOUR ACCESS TOKEN'
access_token_secret = 'YOUR TOKEN SECRET' 
```


```python
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
```

### Get all tweets from 511 Central VA as of 18:50pm Nov 16


```python
js = api.user_timeline('511centralva')
len(js)
```




    20




```python
while True:
    temp = api.user_timeline('511centralva', count=200, max_id=js[-1]._json['id'])
    if js[-1]._json['id'] == temp[-1]._json['id']:
        break
    else:
        js += temp
len(js), js[-1]._json['id']
```




    (3267, 923533267948720128)



Save the tweets for future use

```python
fname = 'data_' + at()+ '.pkl'
with open(fname , 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(js, f, pickle.HIGHEST_PROTOCOL)
```


Get old tweets for analysis

```python
## reading back

with open('data_201710141356.pkl', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    older = pickle.load(f)
len(older)
```




    3251


Read the tweets into a pandas dataframe for analysis

```python
js_dict = {
    'id': [_.id for _ in older],
    'screen_name': [ _.user.screen_name for _ in older],
    'created_at': [_.created_at for _ in older],
    'text': [_.text for _ in older]
}
olddf = pd.DataFrame(js_dict)
olddf.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>created_at</th>
      <th>id</th>
      <th>screen_name</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017-10-14 17:30:26</td>
      <td>919254082661044224</td>
      <td>511centralva</td>
      <td>Cleared: Accident: NB on US-17 (George Washington Memorial Hwy) in Gloucester Co.1:30PM</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017-10-14 17:28:25</td>
      <td>919253572272971776</td>
      <td>511centralva</td>
      <td>Cleared: Incident: NB on I-95 at MM53 in Colonial Heights.1:28PM</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017-10-14 17:20:18</td>
      <td>919251531861512193</td>
      <td>511centralva</td>
      <td>Cleared: Incident: NB on I-195 at MM2 in Richmond.1:20PM</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017-10-14 17:18:25</td>
      <td>919251057817083904</td>
      <td>511centralva</td>
      <td>Cleared: Incident: SB on I-295 at MM14 in Chesterfield Co.1:18PM</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017-10-14 17:14:24</td>
      <td>919250044829782017</td>
      <td>511centralva</td>
      <td>Incident: NB on I-195 at MM2 in Richmond. No lanes closed.1:14PM</td>
    </tr>
  </tbody>
</table>
</div>




```python
js_dict = {
    'id': [_.id for _ in js],
    'screen_name': [ _.user.screen_name for _ in js],
    'created_at': [_.created_at for _ in js],
    'text': [_.text for _ in js]
}
data = pd.DataFrame(js_dict)
data.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>created_at</th>
      <th>id</th>
      <th>screen_name</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017-11-16 23:52:20</td>
      <td>931308990599979008</td>
      <td>511centralva</td>
      <td>Accident: SB on I-95 at MM75 in Richmond. Right shoulder closed.6:52PM</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017-11-16 23:52:20</td>
      <td>931308988968214528</td>
      <td>511centralva</td>
      <td>Update: Accident: SB on US-17 at MM112 in Essex Co. No lanes closed.6:52PM</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017-11-16 23:52:20</td>
      <td>931308987340992517</td>
      <td>511centralva</td>
      <td>Disabled Vehicle: SB on I-95 at MM75 in Richmond. No lanes closed.6:52PM</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017-11-16 23:52:19</td>
      <td>931308985784946688</td>
      <td>511centralva</td>
      <td>Cleared: Accident: NB on US-17 at MM119 in Essex Co.6:52PM</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017-11-16 23:50:26</td>
      <td>931308511312666625</td>
      <td>511centralva</td>
      <td>Cleared: Disabled Vehicle: WB on I-64 at MM218 in New Kent Co.6:50PM</td>
    </tr>
  </tbody>
</table>
</div>

  
### Perform some EDA 

What is the frequency of different kinds of tweets?

```python
data.text.map(lambda x: x.split(':')[0]).value_counts()
```




    Cleared                     1252
    Update                       756
    Accident                     714
    Incident                     237
    Disabled Vehicle             111
    Advisory                      58
    bridge opening                57
    utility work                  18
    Delay                         17
    Vehicle Fire                  15
    Disabled Tractor Trailer      13
    brush fire                     4
    maintenance                    3
    special event                  3
    signal installation            2
    bridge inspection              2
    Closed                         1
    paving operations              1
    bridge work                    1
    road widening work             1
    patching                       1
    Name: text, dtype: int64






```python
len(olddf.append(data))
dummy = pd.concat([data, olddf], ignore_index=True)
dummy = dummy.drop_duplicates('id')
len(dummy), dummy.index.has_duplicates
```




    (6484, False)


### Analyze the structure of tweets

The tweets' key elements appear to be separated by ':' so lets check what is the distribution of that split.

```python
dummy['col_cnt'] = dummy.text.map(lambda x:len(x.split(':')))
```


```python
dummy.col_cnt.value_counts()
```




    4    1979
    3    1200
    5      88
    Name: col_cnt, dtype: int64


### Some sample tweets with different ':'' counts


```python
dummy[dummy.col_cnt == 3].head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>created_at</th>
      <th>id</th>
      <th>screen_name</th>
      <th>text</th>
      <th>col_cnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017-11-16 23:52:20</td>
      <td>931308990599979008</td>
      <td>511centralva</td>
      <td>Accident: SB on I-95 at MM75 in Richmond. Right shoulder closed.6:52PM</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017-11-16 23:52:20</td>
      <td>931308987340992517</td>
      <td>511centralva</td>
      <td>Disabled Vehicle: SB on I-95 at MM75 in Richmond. No lanes closed.6:52PM</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2017-11-16 23:40:17</td>
      <td>931305957505847296</td>
      <td>511centralva</td>
      <td>Accident: NB on US-17 at MM112 in Essex Co. No lanes closed.6:40PM</td>
      <td>3</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2017-11-16 23:36:23</td>
      <td>931304974671368192</td>
      <td>511centralva</td>
      <td>Disabled Vehicle: WB on I-64 at MM218 in New Kent Co. 1 travel lane closed.6:36PM</td>
      <td>3</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2017-11-16 23:22:24</td>
      <td>931301456132562944</td>
      <td>511centralva</td>
      <td>Accident: NB on I-95 at MM78 in Richmond. Right shoulder closed.6:22PM</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
dummy[dummy.col_cnt == 4].head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>created_at</th>
      <th>id</th>
      <th>screen_name</th>
      <th>text</th>
      <th>col_cnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2017-11-16 23:52:20</td>
      <td>931308988968214528</td>
      <td>511centralva</td>
      <td>Update: Accident: SB on US-17 at MM112 in Essex Co. No lanes closed.6:52PM</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017-11-16 23:52:19</td>
      <td>931308985784946688</td>
      <td>511centralva</td>
      <td>Cleared: Accident: NB on US-17 at MM119 in Essex Co.6:52PM</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017-11-16 23:50:26</td>
      <td>931308511312666625</td>
      <td>511centralva</td>
      <td>Cleared: Disabled Vehicle: WB on I-64 at MM218 in New Kent Co.6:50PM</td>
      <td>4</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2017-11-16 23:26:27</td>
      <td>931302475562344448</td>
      <td>511centralva</td>
      <td>Cleared: Accident: WB on I-64 at MM187 in Richmond.6:24PM</td>
      <td>4</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2017-11-16 23:06:31</td>
      <td>931297456817606658</td>
      <td>511centralva</td>
      <td>Cleared: Accident: NB on I-95 at MM87 in Hanover Co.6:06PM</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
dummy[dummy.col_cnt == 3]['text'].map(lambda x: x.split(':')[0]).value_counts()
```




    Accident                    714
    Incident                    237
    Disabled Vehicle            111
    bridge opening               57
    utility work                 18
    Delay                        17
    Vehicle Fire                 15
    Disabled Tractor Trailer     13
    brush fire                    4
    special event                 3
    maintenance                   3
    signal installation           2
    bridge inspection             2
    paving operations             1
    bridge work                   1
    patching                      1
    road widening work            1
    Name: text, dtype: int64




```python
dummy[dummy.col_cnt == 4]['text'].map(lambda x: x.split(':')[0]).value_counts()
```




    Cleared     1193
    Update       727
    Advisory      58
    Closed         1
    Name: text, dtype: int64




```python
dummy[dummy.col_cnt == 5]['text'].map(lambda x: x.split(':')[0]).value_counts()
```




    Cleared    59
    Update     29
    Name: text, dtype: int64






```python
texts = dummy[dummy.text.str.contains('brush')]['text']
```

### Form the Regex to parse tweet texts


```python
import re
pattern = re.compile(r'((?P<status>\w+): )?'
                     r'(?P<advisory>(Advisory|Closed): )?'
                     r'(?P<type>(\w*\s?)+): '
                     r'(?P<direction>[NEWS]B )?on '
                     r'(?P<hwy>.*)( at)? '
                     r'((?P<loc>.*)) in '
                     r'(?P<city>[A-Za-z0-9 ]+).'
                     r'(?P<comment>[a-zA-Z0-9\.&;/ ]+.)?'
                     r'(?P<time>\d+:\d+[AP]M)$'
                    )
```


Test the regular expression to see if it works

```python
attrs = ['status', 'type', 'direction', 'hwy', 'loc', 'city', 'comment', 'time']
for a in attrs:
    t = 'Update: Accident: EB on I-64 at MM199 in Henrico Co. All travel lanes closed. Delay 1 mi.5:52PM'
    t = 'Cleared: Accident: WB on I-64 at MM187 in Richmond.6:24PM'
    print(f'{a} -> {pattern.match(t).group(a)}')
```

    status -> Cleared
    type -> Accident
    direction -> WB 
    hwy -> I-64 at
    loc -> MM187
    city -> Richmond
    comment -> None
    time -> 6:24PM


Some texts throw the parser for a spin, so clean those handful of offensive tweets

```python
dummy.iloc[1520]
dummy.drop(1520, inplace=True)
```


```python
dummy.drop(1519, inplace=True)
dummy.iloc[1519]
```




    created_at                                                        2017-11-07 13:50:25
    id                                                                 927896021925023750
    screen_name                                                              511centralva
    text           Accident: NB on I-195 at MM3 in Richmond. Right shoulder closed.8:50AM
    status                                                                           None
    type                                                                         Accident
    direction                                                                         NB 
    hwy                                                                          I-195 at
    loc                                                                               MM3
    city                                                                         Richmond
    comment                                                        Right shoulder closed.
    time                                                                           8:50AM
    Name: 1529, dtype: object




```python
dummy.iloc[1518]
```




    created_at                                                                                                                                 2017-11-07 13:50:26
    id                                                                                                                                          927896023451783168
    screen_name                                                                                                                                       511centralva
    text           maintenance: On Diascund Road North and South at Hockaday Road in New Kent Co. All NB &amp; all SB travel lanes closed. Potential Delays.8:50AM
    status                                                                                                                                                     NaN
    type                                                                                                                                                       NaN
    direction                                                                                                                                                  NaN
    hwy                                                                                                                                                        NaN
    loc                                                                                                                                                        NaN
    city                                                                                                                                                       NaN
    comment                                                                                                                                                    NaN
    time                                                                                                                                                       NaN
    Name: 1528, dtype: object




```python
dummy.drop(1518, inplace=True)
```


```python
dummy.drop(1527, inplace=True)
dummy.drop(1528, inplace=True)
```


```python
dummy[dummy.text.str.contains('Diascund')]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>created_at</th>
      <th>id</th>
      <th>screen_name</th>
      <th>text</th>
      <th>status</th>
      <th>type</th>
      <th>direction</th>
      <th>hwy</th>
      <th>loc</th>
      <th>city</th>
      <th>comment</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1146</th>
      <td>2017-11-09 21:34:22</td>
      <td>928737552491835392</td>
      <td>511centralva</td>
      <td>Cleared: maintenance: NB On Diascund Road North and South at Hockaday Road in New Kent Co.4:34PM</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### Parse additional information from tweets text into data frame columns

Some tweets are still little wierd for regex parser, and we will skip those as well. But if we skip them, we will display them as well to keep track of the ones that were skipped.

```python
for idx in dummy.index:
    if idx%500 == 0:
        print(idx)
    try:
        m = pattern.match(dummy.loc[idx,'text'])
        for a in attrs:
            dummy.loc[idx,a] = m.group(a)
    except AttributeError:
        print (dummy.loc[idx,'text'])
```

    0
    500
    Cleared: Incident: NB (South Hopewell Street) in Hopewell.4:52PM
    Incident: NB (South Hopewell Street) in Hopewell. No lanes closed.4:26PM
    1000
    Cleared: maintenance: NB On Diascund Road North and South at Hockaday Road in New Kent Co.4:34PM
    1500
    2000
    2500
    3000
    3500
    Update: bridge repair: EB On Scotts Road East and West between Canton Road and Level Green Road in Henrico Co. All EB &amp; all WB travel10:32PM
    bridge repair: On Scotts Road East and West between Canton Road and Level Green Road in Henrico Co. All EB &amp; all WB travel lanes close9:52PM
    4000
    bridge work: at Gwynn's Island Bridge in Mathews Co. All NB &amp; all SB travel lanes closed. Potential Delays.11:26AM
    4500
    Cleared: bridge repair: EB On Scotts Road East and West between Canton Road and Level Green Road in Henrico Co.9:58AM
    Cleared: Incident: SB (Coleman Memorial Bridge) in Gloucester Co.8:08AM
    Incident: SB (Coleman Memorial Bridge) in Gloucester Co. No lanes closed.7:30AM
    5000
    5500
    6000
    6500



```python
import seaborn as sns
plt.style.use('fivethirtyeight')
```

Finally the output of parsing:

```python
dummy.tail()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>created_at</th>
      <th>id</th>
      <th>screen_name</th>
      <th>text</th>
      <th>status</th>
      <th>type</th>
      <th>direction</th>
      <th>hwy</th>
      <th>loc</th>
      <th>city</th>
      <th>comment</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6513</th>
      <td>2017-09-21 06:36:24</td>
      <td>910754565540151296</td>
      <td>511centralva</td>
      <td>Update: Incident: NB on I-95 at MM75 in Richmond. 1 SB travel lane closed.2:36AM</td>
      <td>Update</td>
      <td>Incident</td>
      <td>NB</td>
      <td>I-95 at</td>
      <td>MM75</td>
      <td>Richmond</td>
      <td>1 SB travel lane closed.</td>
      <td>2:36AM</td>
    </tr>
    <tr>
      <th>6514</th>
      <td>2017-09-21 06:24:24</td>
      <td>910751549248466944</td>
      <td>511centralva</td>
      <td>Incident: NB on I-95 at MM75 in Richmond. No lanes closed.2:24AM</td>
      <td>None</td>
      <td>Incident</td>
      <td>NB</td>
      <td>I-95 at</td>
      <td>MM75</td>
      <td>Richmond</td>
      <td>No lanes closed.</td>
      <td>2:24AM</td>
    </tr>
    <tr>
      <th>6515</th>
      <td>2017-09-21 05:44:25</td>
      <td>910741486081384448</td>
      <td>511centralva</td>
      <td>Cleared: bridge opening: NB on VA-156 at B. Harrison Bridge in Prince George Co.1:44AM</td>
      <td>Cleared</td>
      <td>bridge opening</td>
      <td>NB</td>
      <td>VA-156 at B. Harrison</td>
      <td>Bridge</td>
      <td>Prince George Co</td>
      <td>None</td>
      <td>1:44AM</td>
    </tr>
    <tr>
      <th>6516</th>
      <td>2017-09-21 05:32:19</td>
      <td>910738441587101697</td>
      <td>511centralva</td>
      <td>bridge opening: on VA-156 at B. Harrison Bridge in Prince George Co. All NB &amp;amp; all SB travel lanes closed. Potential Delays.1:32AM</td>
      <td>None</td>
      <td>bridge opening</td>
      <td>None</td>
      <td>VA-156 at B. Harrison</td>
      <td>Bridge</td>
      <td>Prince George Co</td>
      <td>All NB &amp;amp; all SB travel lanes closed. Potential Delays.</td>
      <td>1:32AM</td>
    </tr>
    <tr>
      <th>6517</th>
      <td>2017-09-21 03:10:25</td>
      <td>910702730670374914</td>
      <td>511centralva</td>
      <td>Cleared: Incident: NB on I-95 at MM54 in Colonial Heights.11:10PM</td>
      <td>Cleared</td>
      <td>Incident</td>
      <td>NB</td>
      <td>I-95 at</td>
      <td>MM54</td>
      <td>Colonial Heights</td>
      <td>None</td>
      <td>11:10PM</td>
    </tr>
  </tbody>
</table>
</div>




```python
dummy.status.hasnans, dummy.type.hasnans
```




    (True, False)




```python
for idx in dummy[dummy.type.isnull()].index:
    dummy.drop(idx, inplace=True)
```


```python
reports = dummy[dummy.status.isnull()]
len(reports)
```




    2348




```python
reports.created_at.dt.date.value_counts().plot()
plt.title('Daily incident reported since Sep 21, 2017')
plt.ylabel('Reported Count')
plt.xlabel('Date')
plt.show()
```


![png](https://s3.amazonaws.com/cva-traffic-analysis/output_45_0.png)



```python
reports.type.value_counts(dropna=False)[:10].plot(kind='bar')
plt.ylabel('Reported Count')
plt.title('Top 10 types of traffic related events in Central VA')
plt.show()
```


![png](https://s3.amazonaws.com/cva-traffic-analysis/output_46_0.png)



```python
reports.city.value_counts(dropna=False)[:10].plot(kind='bar')
plt.ylabel('Reported Count')
plt.title('Top 10 areas for traffic related events in Central VA')
plt.show()
```


![png](https://s3.amazonaws.com/cva-traffic-analysis/output_47_0.png)



```python
temp = reports.groupby(['type', 'city'])['id'].count().sort_values(ascending=False)[:20].to_frame()
temp.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>id</th>
    </tr>
    <tr>
      <th>type</th>
      <th>city</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">Accident</th>
      <th>Henrico Co</th>
      <td>281</td>
    </tr>
    <tr>
      <th>Richmond</th>
      <td>246</td>
    </tr>
    <tr>
      <th>Chesterfield Co</th>
      <td>222</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Incident</th>
      <th>Henrico Co</th>
      <td>147</td>
    </tr>
    <tr>
      <th>Chesterfield Co</th>
      <td>141</td>
    </tr>
  </tbody>
</table>
</div>




```python
def hitype(x):
    return (x in ['Accident', 'Incident', 'Disabled Vehicle'])
sns.countplot(data=reports[reports.type.map(hitype)], x='city', hue='type', order=reports.city.value_counts().iloc[:5].index)
plt.title('Top traffic incidents by area')
plt.show()
```


![png](https://s3.amazonaws.com/cva-traffic-analysis/output_49_0.png)


### Which county has the most accidents?

```python
dummy.city.value_counts(dropna=False)[:10].plot(kind='bar')
plt.show()
```


![png](https://s3.amazonaws.com/cva-traffic-analysis/output_51_0.png)


### Where are the most accident prone zones?

Drive safe here!

```python
reports[reports.type == 'Accident'].groupby(['hwy','loc'])['id'].count().sort_values(ascending=False)[:10].plot(kind='bar')
plt.title('Accident prone areas')
plt.xlabel('Highway, Milemarker')
plt.ylabel('Accident Count')
plt.show()
```


![png](https://s3.amazonaws.com/cva-traffic-analysis/output_53_0.png)







```python
sns.countplot(data=reports[reports.type == 'Accident'], x='hour', palette='Reds_d')
plt.title('Accident frequency through the day')
plt.show()
```


![png](https://s3.amazonaws.com/cva-traffic-analysis/output_60_0.png)

### Summary

Drive safe all the time! But especially around MM78 , MM74 and MM79 on I-95. Also apparently there 8am and 5pm are when most accidents happen. Do dont rush to office and relax when you drive back home. Unwind!
