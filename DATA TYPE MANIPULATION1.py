#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[5]:


8//6


# In[12]:


a = ["a", "b", "c"]
b = "Hello"
a.extend(b)
a


# In[13]:


a.insert(4,b)
a


# In[15]:


List = [1, 2, 3, 4, 5, 6, 7, 8, 9,10]
print(List[::2])


# In[16]:


print(List[1:10:2])


# In[17]:


print(List[1:9:2])


# In[18]:


print(List[1::2])


# In[20]:


Employee = {1: {'Name': 'John', 'Age': '35', 'Gender': 'Male'}, 2: {'Name': 'Mike', 'Age': '22', 'Gender': 'Male'}}
Employee


# In[22]:


Employee = {1: {'Name': 'John', 'Age': '35', 'Gender': 'Male'}, 2: {'Name': 'Mike', 'Age': '22', 'Gender': 'Male'}}
Employee[0][1]


# In[23]:


Employee[1]['Age']


# In[24]:


Employee[0][‘Age’]


# In[25]:


Employee[1][Age]


# In[26]:


var = 10
for i in range(5):
    for j in range(2, 5, 1):
        if var % 2 == 0:
            continue
            var += 1
    var+=1
print(var)


# In[28]:


func=lambda x:bool(x%2)
print(func(10), func(21))


# In[29]:


print(func(5), func(7))


# In[30]:


print(func(4), func(8))


# In[31]:


print(func(21), func(10))


# In[33]:


def f(x):
    def f1(*args, **kwargs):
           print("Hello")
           return x(*args, **kwargs)
    return f1

