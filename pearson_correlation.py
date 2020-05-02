#import libraries
import pandas as pd

#read csv
data = pd.read_csv('data.csv')

data.head(20)

#df = pd.DataFrame(data).T
#corr = df.corr(method = 'pearson')
#print(corr)
data.corr(method = 'pearson')
