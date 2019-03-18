
# coding: utf-8

# In[25]:


import random
import json


# In[26]:


def random_dropout_generator(start_num,end_num):
    def random_dropout_list():
        while(True):
            yield [x for x in range(start_num,end_num) if random.random()>0.5]
    return random_dropout_list()


# In[29]:


def random_dropout_creat_file(file_name,start_num,end_num,iterations=50000):
    with open(file_name,'w') as f:
        generator = random_dropout_generator(start_num,end_num)
        l = [next(generator) for y in range(iterations)]
        json.dump(l,f)


# In[33]:




