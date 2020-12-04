#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats
import math
import statistics as st
from sklearn.datasets import load_iris
get_ipython().system('pip install pingouin')
import pingouin as pg


get_ipython().system('pip install factor_analyzer')
from factor_analyzer import FactorAnalyzer


# In[9]:


df = pd.read_csv('Personality_VTO.csv')
df = df.rename(columns={"Let's start by checking this e-commerce webpage: https://www.ray-ban.com/usa/sunglasses/view-all/plp. Select the model you like most. Which model did you choose?":'Model', 
                        'Did you manage to use Virtual Try-On?':'Used VTO 1', 
                        'Does it work now?':'Used VTO 2',
                       'Using Virtual Try-On technology in online stores is a good/bad idea.':'A1',
                        'Using Virtual Try-On technology in online stores is a foolish/wise idea.':'A2',
                        'I dislike/like the idea of Virtual Try On in online stores. ':'A3',
                        'Using Virtual Try-On would be unpleasant/pleasant. ':'A4',
                        'Using the Virtual Try-On technology will be of no benefit for me.':'P1',
                        'The advantages of Virtual Try-On will outweigh the disadvantages. ':'P2',
                        'Overall, using Virtual Try-On will be advantageous. ':'P3',
                        'Using Virtual Try-On technology will improve my shopping.':'P4',
                        'After seeing the website and using the Virtual Try-On technology, how likely are you to buy sunglasses from this online store?':'BI2',
                        'I would be willing to purchase sunglasses through this online store.':'BI3',
                        'Assuming the products on the website suit your taste or needs, how willing would you be to purchase products from this online store?':'BI1',
                        'In the future, I would buy my sunglasses from this online store. ':'BI4',
                        'I see myself as someone who has few artistic interests':'O1',
                        'I see myself as someone who has active imagination.':'O2',
                        'I consider myself as someone original, who comes up with new ideas.':'O3',
                        "Have you ever used RayBan's Virtual Try-On feature before?":'Previous Exposure',
                        "Optional question. What’s your opinion on Virtual Try-On technology?":'Open Question',
                       'Gender:':'Gender',
                       'Age:':'Age'
                       })
#cleaning

#removing incomplete questionnaires: there's incomplete answers for A1, A2, BI2
df.drop(df.index[df["Timestamp"]=="01/10/2020 12:56:21"], inplace = True)
df.drop(df.index[df["Timestamp"]=="03/11/2020 15:02:59"], inplace = True)
df.drop(df.index[df["Timestamp"]=="26/10/2020 19:17:49"], inplace = True)

#removing out-of-range age
old = df[df['Age']>=36].index
df.drop(old, inplace=True)

#removing those who did not manage to use the VTO feature
na = df[df['Used VTO 2']=="No"].index
df.drop(na, inplace=True)
df.reset_index(drop=True)

###account for reverse-coded items
df['P1'] = df['P1'].replace({1:8, 2:7, 3:6, 4:5})
df['O1'] = df['O1'].replace({1:5, 2:4})

df.head()


# In[ ]:


#computing basic stats
df.describe()


# In[ ]:


#create a dataset from just the answers
data = df.loc[:, 'A1':'O3']
data 


# # Basic Statistics

# In[ ]:


print("Age mean:", round(df['Age'].mean(), 2), ", Age variance:", st.variance(df["Age"]))

print("Females:", len(df.loc[df['Gender'] == 'Female']), ",",
      "Males:", len(df.loc[df['Gender'] == 'Male']),",",
      "Other:", len(df.loc[df['Gender'] == 'Other']),",",
      "Prefer Not To Say:", len(df.loc[df['Gender'] == 'Prefer not to say'])
     )

print()

#should be 23.09 mean (not 23.65)


# # Computing scores

# In[10]:


##Scoring each dimension

#Attitude
A = df["A1"] + df["A2"] + df["A3"] + df["A4"]
df.insert(8, "A", A, True)

#Perceived Usefulness
P= (df["P1"]) + df["P2"] + df["P3"] + df["P4"]
df.insert(13, "P", P, True)

#Behavioral Intention
BI= df["BI1"] + df["BI2"] + df["BI3"] + df["BI4"]
df.insert(18, "BI", BI, True)

#Openness 
O =  df["O2"] + df["O3"] + df["O1"]
df.insert(22, "O", O, True)

#creating a new table just for scores
scores = df[["A", "P", "BI", "O"]]


# In[5]:


scores.describe()


# In[12]:


#dataset.to_csv(r'C:\Users\Tania\Downloads\Python\Memoire\ENTER_NAME.csv')


# ### Cronbach's alpha
# using: https://towardsdatascience.com/cronbachs-alpha-theory-and-application-in-python-d2915dd63586

# In[ ]:


#verifying the correlations
corr_matrix = scores.corr()
print(corr_matrix)


# In[ ]:


def cronbach_alpha(df):
    # Transforming the df into a correlation matrix
    df_corr = df.corr()
    
    # Calculating N: the number of variables equals the number of columns in the df
    N = df.shape[1]
    
    # Calculating R: loop through the columns and append every relevant correlation to an array calles "r_s". Then, we'll calculate the mean of "r_s"
    rs = np.array([])
    for i, col in enumerate(df_corr.columns):
        sum_ = df_corr[col][i+1:].values
        rs = np.append(sum_, rs)
    mean_r = np.mean(rs)
    
   # 3. Use the formula to calculate Cronbach's Alpha 
    cronbach_alpha = (N * mean_r) / (1 + (N - 1) * mean_r)
    return cronbach_alpha


# In[ ]:


#alpha attitude
df_attitude = df[['A1', 'A2', 'A3', 'A4']]
cronbach_alpha(df_attitude)


# In[ ]:


#alpha PU
df_pu = df[['P1', 'P2', 'P3', 'P4']]
cronbach_alpha(df_pu)


# In[ ]:


#alpha BI
df_bi = df[['BI1', 'BI2', 'BI3', 'BI4']]
cronbach_alpha(df_bi)


# In[ ]:


#alpha O
df_o = df[['O1', 'O2', 'O3']]
cronbach_alpha(df_o)


# # Factor loading

# In[ ]:


# Create a dataset with only relevant columns
x = df.drop(['Timestamp', 'Model', 'Used VTO 1', 'Used VTO 2','Gender', 'Age', 'Previous Exposure',
       'Open Question', 'A', 'P', 'O', 'BI'], axis=1)

#check if it is relevant to use factor analysis
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
chi_square_value,p_value=calculate_bartlett_sphericity(x)
chi_square_value, p_value

#The test was statistically significant, so we can proceed to the Factor Loading analysis.


# In[ ]:


x.info()


# # method YT
# https://www.youtube.com/watch?v=ttBs_wfw_6U

# In[ ]:


fa = FactorAnalyzer(n_factors=5, rotation="varimax")
fa.fit(x)

#get the loadings (matrix like)
loadings = fa.loadings_

#get Eigenvector and EighenValues
ev, v = fa.get_eigenvalues()

#do a scree plot
xvals = range(1, x.shape[1]+1)
plt.scatter(xvals, ev)
plt.plot(xvals, ev)
plt.title('Scree plot')
plt.xlabel('Factor')
plt.ylabel('EigenValue')
plt.grid()


# In[ ]:


#display the factor loadings
pd.DataFrame.from_records(loadings)


# # method article
# https://towardsdatascience.com/factor-analysis-a-complete-tutorial-1b7621890e42

# In[ ]:


#determine if data suited for factor analyssi
from factor_analyzer.factor_analyzer import calculate_kmo
kmo_all,kmo_model=calculate_kmo(x)
kmo_model
#0.84, which is excellent. 


# In[ ]:


#19 columns containing the survey answers
fa = FactorAnalyzer()
fa.fit(x)
#Getting Eigen values and plotting'em
ev, v = fa.get_eigenvalues()
ev
plt.plot(range(1,x.shape[1]+1),ev)


# In[ ]:


fa = FactorAnalyzer(4, rotation='varimax')
fa.fit(x)
loads = fa.loadings_
pd.DataFrame.from_records(loads)
#A, P, BI, O


# In[ ]:


#cronbach - new method

#Create the factors
factor1 = x[['A1', 'A2', 'A3', 'A4']]
factor2 = x[['P1', 'P2', 'P3', 'P4']]
factor3 = x[['BI1', 'BI2', 'BI3','BI4']]
factor4 = x[['O1', 'O2', 'O3']]
#Get cronbach alpha
factor1_alpha = pg.cronbach_alpha(factor1)
factor2_alpha = pg.cronbach_alpha(factor2)
factor3_alpha = pg.cronbach_alpha(factor3)
factor4_alpha = pg.cronbach_alpha(factor4)
print(factor1_alpha, factor2_alpha, factor3_alpha, factor4_alpha)

#the alphas evaluated are 0.84, 0.68, 0.86, 0.65)


# # Covariance: ANCOVA
# https://www.statology.org/ancova-python/

# # Hypothesis Testing using Pearson

# In[ ]:


# defining the pearson correlation function
def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    corr_mat = np.corrcoef(x, y)
    # Return entry [0,1]
    return corr_mat[0,1]


# # # H1

# In[ ]:


# Hypothesis 1: Consumers with a higher Openness will have a higher Perceived Usefulness of the VTO feature from an online store. 
# Pearson Coefficient: Openness & Perceived Usefulness
r_OPU = pearson_r(O, P)
print("PEARSON COEFFICIENT")
print("Openness & Perceived Usefulness:", r_OPU)


# In[ ]:


# Initialize permutation replicates: perm_replicates for r_OPU
perm_replicates1=np.empty(10000)
# Draw replicates
for i in range(0,10000):
#Permute illiteracy measurments: illiteracy_permuted
    O_permuted = np.random.permutation(O)
#Compute Pearson correlation
    perm_replicates1[i] = pearson_r(O_permuted, P)
#Compute p-value: p
p = np.sum(perm_replicates1 >= r_OPU)/len(perm_replicates1)
print('p-val =',p)


# ## H2

# In[ ]:


#Perceived Usefulness & Attitude
r_PUA = pearson_r(P, A)
print("PEARSON COEFFICIENT")
print("Perceived Usefulness & Attitude:", r_PUA)


# In[ ]:


# Initialize permutation replicates: perm_replicates for r_PUA
perm_replicates2=np.empty(10000)
# Draw replicates
for i in range(0,10000):
#Permute illiteracy measurments: illiteracy_permuted
    P_permuted = np.random.permutation(P)
#Compute Pearson correlation
    perm_replicates2[i] = pearson_r(P_permuted, A)
#Compute p-value: p
p = np.sum(perm_replicates2 >= r_PUA)/len(perm_replicates2)
print('p-val =',p)


# ## H3

# In[ ]:


# Hypothesis 3: Attitude mediates Intention to purchase from the online store using VTO.
#Attitude & Behavioral Intention
r_ABI = pearson_r(A, BI)
print("PEARSON COEFFICIENT")
print("Attitude & Behavioral Intention:",r_ABI)


# In[ ]:


# Initialize permutation replicates: perm_replicates for r_ABI
perm_replicates3=np.empty(10000)
# Draw replicates
for i in range(0,10000):
#Permute illiteracy measurments: illiteracy_permuted
    A_permuted = np.random.permutation(A)
#Compute Pearson correlation
    perm_replicates3[i] = pearson_r(A_permuted, BI)
#Compute p-value: p
p = np.sum(perm_replicates3 >= r_ABI)/len(perm_replicates3)
print('p-val =',p)


# # SEM
# using AMOS: https://www.youtube.com/watch?v=xAVHnSMxW0c
