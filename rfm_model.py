#rfm model
#Recency(최신), Frequency(회수), Monetary(화폐) segmentation
import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')

data.head()

data.tail()

data.info()

data= data[pd.notnull(data['CUSTOMER_ID'])]

filtered_data=data[['CONTY','ID']].drop_duplicates()

filtered_data.info()

filtered_data.MCC.value_counts()[:10].plot(kind='bar')

usa_data=data[data.CONTY=='United States']

usa_data.info()

usa_data.describe()

usa_data=usa_data[['BUYER_ID','ORDER_DATE','QUANTITY','PRICE']]

usa_data['TotalPrice'] = usa_data['QUANTITY'] * usa_data['PRICE']

usa_data['ORDER_DATE'].min(),usa_data['ORDER_DATE'].max()

PRESENT = dt.datetime(2020,5,1)

usa_data['ORDER_DATE'] = pd.to_datetime(usa_data['ORDER_DATE'])

usa_data.head()

usa_data['TotalPrice'] = usa_data['TotalPrice'].astyMpe(int)

rfm= usa_data.groupby('BUYER_ID').agg({'ORDER_DATE': lambda date: (PRESENT - date.max()).days,
                                        'QUANTITY': lambda num: num.sum(),
                                        'TotalPrice': lambda price: price.sum()})

rfm.head()
rfm.tail()

# Change the name of columns
rfm.columns=['recency','frequency','monetary']

rfm.head()

rfm['recency'] = rfm['recency'].astype(int)
rfm['frequency'] = rfm['frequency'].astype(object)
rfm.info()

rfm['r_quartile'] = pd.qcut(rfm['recency'], 4, ['1','2','3','4'])
rfm['f_quartile'] = pd.qcut(rfm['frequency'], 4, ['3','2','1'],duplicates ='drop')
rfm['m_quartile'] = pd.qcut(rfm['monetary'], 4, ['4','3','2','1'])

rfm.head()

rfm['RFM_Score'] = rfm.r_quartile.astype(str)+ rfm.f_quartile.astype(str) + rfm.m_quartile.astype(str)
rfm.head()

rfm[rfm['RFM_Score']=='111'].sort_values('monetary', ascending=False).head(100)

