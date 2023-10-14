#!/usr/bin/env python
# coding: utf-8

# In[6]:


#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


# In[12]:


index = ['Apple', 'Banana', 'Orange']
quantity = [34, 20, 30]
pd.Series(data = quantity, index = index)


# In[13]:


dict = {'A':30, 'B':40, 'C':50}
index = ['A', 'B', 'D']
pd.Series(data = dict, index = index)


# In[16]:


s1 = pd.Series([1, 2, 5, 6.5])
s2 = pd.Series(['first', 35, 'college', 62.5])
s2


# In[19]:


a1 = ['Hogwarts', 'Durmstrang', 'Beauxbatons']
a2 = ['Hogwarts', 'Durmstrang', 'Beauxbatons']
a3 = ['Hogwarts', 'Durmstrang', 'Beauxbatons']
school = [a1, a2, a3]
inst = ['School_1', 'School_2', 'School_3']
Muggle_data = pd.DataFrame(data = school, columns = inst)
Muggle_data


# In[22]:


Muggle_data.info()


# In[7]:


n=100
p=0.09
k=np.arange(0,101)
k


# In[10]:


from scipy.stats import binom
binomial=binom.pmf(k=k,n=n,p=p)
binomial


# In[ ]:




