#!/usr/bin/env python
# coding: utf-8

# In[2]:


import findspark
findspark.init()


# In[3]:


import pyspark
from pyspark.sql import SparkSession
spark=SparkSession.builder.master("local").appName('final').getOrCreate()
sc=spark.sparkContext


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import matplotlib.pyplot as plt


# In[5]:


from pandas.core.frame import DataFrame
import seaborn as sns


# In[6]:


import os 
os.chdir('C:\\Users\\wuzix\\Desktop\\final big\\archive')


# In[7]:


df = spark.read.option('header','true').csv('C:\\Users\\wuzix\\Desktop\\final big\\archive\\movies_metadata.csv',inferSchema=True)
df.show()


# In[8]:


df.printSchema()


# In[ ]:


rating = spark.read.option('header','true').csv('ratings.csv',inferSchema=True)


# In[ ]:


rating2 = rating.groupby('movieId').agg(F.mean('rating')).select(['movieId','avg(rating)'])
df1 =rating2.join(df, df.id==rating2.movieId, how='inner')


# In[9]:


df1 = df1.filter((df.revenue > 1000000)&(df.budget > 100000)&(df.runtime > 0) ).select(['genres','id','imdb_id','original_language','popularity', 'release_date', 'revenue', 'budget', 'runtime','status', 'title','vote_average' , 'vote_count'])


# In[10]:


import pandas as pd


# In[11]:


from pyspark.sql import functions as F
df2=df1.withColumn('ROI', (F.col('revenue')-F.col('budget'))/F.col('budget'))
df2.show()


# In[ ]:





# genre --- finance

# In[9]:


genre = ['Horror', 'Mystery', 'Action', 'Adventure', 'Fantasy', 'Comedy', 'Thriller', 'Documentary', 'animation', 'romance', 'family', 'western','music' , 'crime', 'history', 'war']
genre = [s.capitalize() for s in genre]
for name in genre:
    locals()[name] = df2.filter(F.col("genres").contains(name))


# In[10]:


from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

data = []
for name in genre:
    data.append((name,locals()[name].agg(F.mean('revenue')).collect()[0][0],locals()[name].agg(F.mean('budget')).collect()[0][0],locals()[name].agg(F.mean('ROI')).collect()[0][0]))


# In[11]:


genre_fin_df = sqlContext.createDataFrame(data, ('genre', 'avg_revenue', 'avg_budget', 'avg_ROI'))


# In[12]:


genre_fin_df.show()


# In[13]:


genre_fin_df.orderBy(genre_fin_df.avg_ROI.desc()).show(3)


# In[14]:


genre_fin_df.orderBy(genre_fin_df.avg_revenue.desc()).show(3)


# In[15]:


genre_fin_df.orderBy(genre_fin_df.avg_budget.desc()).show(3)


# In[16]:


rev_data = sorted(dict(zip([genre_fin_df.collect()[i][0] for i in range(16)],[genre_fin_df.collect()[i][1] for i in range(16)])).items(), key = lambda x:x[1],reverse = True)
bud_data = sorted(dict(zip([genre_fin_df.collect()[i][0] for i in range(16)],[genre_fin_df.collect()[i][2] for i in range(16)])).items(), key = lambda x:x[1],reverse = True)
ROI_data = sorted(dict(zip([genre_fin_df.collect()[i][0] for i in range(16)],[genre_fin_df.collect()[i][3] for i in range(16)])).items(), key = lambda x:x[1],reverse = True)

rev_data


# In[17]:


plt.figure(figsize=(10,5))
plt.bar(range(len(rev_data)),[i[1] for i in rev_data], tick_label=[i[0] for i in rev_data], width= 0.5)
plt.title('avg_revenue')
plt.xticks(rotation=300)
plt.show()


# In[18]:


plt.figure(figsize=(10,5))
plt.bar(range(len(bud_data)),[i[1] for i in bud_data], tick_label=[i[0] for i in bud_data], width= 0.5)
plt.title('avg_budget')
plt.xticks(rotation=300)
plt.show()


# In[19]:


plt.figure(figsize=(10,5))
plt.bar(range(len(ROI_data)),[i[1] for i in ROI_data], tick_label=[i[0] for i in ROI_data], width= 0.5)
plt.title('avg_ROI')
plt.xticks(rotation=300)
plt.show()


# popularity --- finance

# In[20]:


df2.count()
#len(df2.columns)


# In[21]:


df_new = df2.withColumn("popularity",df2.popularity.cast('double'))
df_new = df_new.withColumn("runtime",df_new.runtime.cast('double'))
df_new = df_new.withColumn("vote_average",df_new.vote_average.cast('double'))
df_new = df_new.withColumn("vote_count",df_new.vote_count.cast('double'))


# In[22]:


df_new.show()


# In[23]:


pop_data = [df_new.collect()[i][4] for i in range(4359)]
ROI_row_data = [df_new.collect()[i][13] for i in range(4359)]


# In[31]:


plt.figure(figsize = (10,6))
plt.scatter(pop_data, ROI_row_data ,color="blue")  
plt.xlim((-10, 300))
plt.ylim((-2, 50))
plt.title('popularity--ROI')
plt.grid()
plt.show() 


# In[91]:


a={'ROI':ROI_row_data, 'RUNTIME':pop_data}
data1=DataFrame(a)
g1 = sns.lmplot(data = data1,x='RUNTIME',y='ROI',height=7,aspect=1.6,palette='Set1',scatter_kws=dict(s=60, linewidths=.7, edgecolors='black'))
sns.set(style="whitegrid", font_scale=1.5)
g1.set(xlim=(-1, 55), ylim=(-50, 450))
g1.fig.set_size_inches(10, 6)
g1.tight_layout()
plt.title("RUNTIME --- ROI")
plt.show()


# In[27]:


import scipy.stats as stats

r,p = stats.pearsonr(ROI_row_data,pop_data)  
print('corr = %6.3f，p_value = %6.3f'%(r,p))


# runtine --- finance

# In[28]:


runtime_data = [df_new.collect()[i][8] for i in range(4359)]


# In[30]:


plt.figure(figsize = (10,6))
plt.scatter(runtime_data, ROI_row_data,color="blue") 
plt.xlim((0, 300))
plt.ylim((-2, 50))
plt.title('runtime--ROI')
plt.grid() 
plt.show() 


# In[92]:


b={'ROI':ROI_row_data, 'RUNTIME':runtime_data}
data2=DataFrame(b)
g2 = sns.lmplot(data = data2,x='RUNTIME',y='ROI',height=7,aspect=1.6,palette='Set1',scatter_kws=dict(s=60, linewidths=.7, edgecolors='black'))
sns.set(style="whitegrid", font_scale=1.5)
g2.set(xlim=(50, 260), ylim=(-50, 450))
g2.fig.set_size_inches(10, 6)
g2.tight_layout()
plt.title("RUNTIME --- ROI")
plt.show()


# In[34]:


import scipy.stats as stats

r,ru = stats.pearsonr(ROI_row_data,runtime_data)  
print('corr = %6.3f，p_value = %6.3f'%(r,ru))


# rate --- finance

# In[36]:


df_vote = df_new.filter(df_new.vote_count > 100)
df_vote.count()


# In[37]:


rate_data = [df_vote.collect()[i][11] for i in range(3321)]


# In[39]:


ROI_row_data1 = [df_vote.collect()[i][13] for i in range(3321)]


# In[81]:


plt.figure(figsize = (10,6))
plt.scatter(rate_data, ROI_row_data1,color="Red") 
plt.xlim((2, 10))
plt.ylim((-2, 500))
plt.title('rate--ROI')
plt.grid() 
plt.show() 


# In[43]:


import scipy.stats as stats

r1,ra = stats.pearsonr(ROI_row_data1,rate_data)  
print('corr = %6.3f，p_value = %6.3f'%(r1,ra))


# In[93]:


c={'ROI':ROI_row_data1, 'RATE':rate_data}
data3=DataFrame(c)
g3 = sns.lmplot(data = data3,x='RATE',y='ROI',height=7,aspect=1.6,palette='Set1',scatter_kws=dict(s=60, linewidths=.7, edgecolors='black'))
sns.set(style="whitegrid", font_scale=1.5)
g3.set(xlim=(2, 10), ylim=(-50, 450))
g3.fig.set_size_inches(10, 6)
g3.tight_layout()
plt.title("RATE --- ROI")
plt.show()

