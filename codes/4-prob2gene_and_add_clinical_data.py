#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re
import os
import sys
import time
import pickle 
import requests
import numpy as np
import pandas as pd
from ipywidgets import widgets


# In[ ]:


MAIN_DIR = '.'
DATASETS = ['GSE11121',
            'GSE18864',
            'GSE20711',
            'GSE23593',
            'GSE27120',
            'GSE32646',
            'GSE36771', 
            'GSE42568', 
            'GSE50948', 
            'GSE5460', 
            'GSE11001',
            'GSE87007',
            'GSE88770',
            'GSE7390', 
            'GSE78958',
            'GSE45255',
            'GSE61304',
            'GSE63471',
            'GSE21653',
            'GSE26639',
            'GSE17907',
            'GSE10810',
            'GSE25066',
            'GSE47109',
            'GSE95700',
            'GSE5327',
            'GSE48390',
            'GSE58984',
            'GSE103091',
            'GSE45827',
            'GSE65194', 
            'GSE1456',
            'GSE102484']


# In[ ]:


annot = pd.read_csv('prob2gene.csv')
probe_to_symbol = pd.Series(annot['V2'].values, index=annot['V1']).to_dict()


# In[ ]:


df = pd.read_csv(f'{MAIN_DIR}/merged/merged_COMBAT_rma.matrix.csv', index_col=0).transpose()    
df.shape


# ### Change the column names to their corresponding genes

# In[ ]:


result = []
for id, row in df.iterrows():
    genes = {}
    
    # Iterate over columns
    for probe, value in row.iteritems():
        if probe_to_symbol.get(probe):
            if probe_to_symbol[probe] in genes:
                genes[probe_to_symbol[probe]].append(float(value))  # if we alread have the gene in our dictionary
            else:
                genes[probe_to_symbol[probe]] = [float(value)] # create a new list for the values of this gene
                
        # Just for the case of probes with 'X' at the beginning        
        if probe_to_symbol.get(probe[1:]):
            if probe_to_symbol[probe[1:]] in genes:
                genes[probe_to_symbol[probe[1:]]].append(float(value))
            else:
                genes[probe_to_symbol[probe[1:]]] = [float(value)]
                
    res_genes = {'sample_id': id.split('.')[0]}
    for k, v in genes.items():
        res_genes[k] = sum(v) / float(len(v))

    result.append(res_genes)
    
df_genes = pd.DataFrame(result)
df_genes.head(5)


# In[ ]:


# deleting AFF 
unwanted =df_genes.filter(regex='^AFFX')
df_genes.drop(unwanted, axis=1, inplace=True)

# deleting -at
unwanted =df_genes.filter(regex='_at')
df_genes.drop(unwanted, axis=1, inplace=True)
df_genes.head(5) 

unwanted = df_genes.filter(regex='---')
df_genes.drop(unwanted, axis=1, inplace=True)
df_genes.head(5) 


# In[ ]:


# selecting 'sample_id' as index
df_genes =df_genes.set_index('sample_id')
df_genes.head(5)


# In[ ]:


df_genes.shape


# ## Add clinical data

# In[ ]:


df_total = None
df_clinical_data = pd.DataFrame()

for dataset in DATASETS:
    print(f'Loading {dataset} cancer data...')
    
    # loading clinical data
    df_cancer_ann = pd.read_csv(f'{MAIN_DIR}/main_data/{dataset}/{dataset}_annotation.csv') 
    #df_cancer_ann['Dataset'] = cancer
    df_cancer_ann.rename(columns={'sample':'sample_id'}, inplace=True) 
    
    # select important annotations
    df_important_ann = df_cancer_ann[['sample_id', 'tissue', 'grade', 'stage', 'subtype', 'tumor size(mm)',
                                      'age at diagnosis(yrs)', 'death event time(yrs)', 'death event', 'er',
                                      'pr', 'her2', 't.rfs', 'e.rfs',  't.tdm', 'e.tdm']]
    
    # all clinical data
    df_clinical_data = df_clinical_data.append(df_important_ann, ignore_index = True) 

df_clinical_data = df_clinical_data.replace({'subtype': {' Basal': 'Basal', 'lumB': 'LumB', 'lumA': 'LumA', ' LumA': 'LumA', ' LumB':'LumB', ' HER2':'HER2', 'Normal': -1, '-1':-1}})
df_clinical_data = df_clinical_data.replace({'subtype': {'Her2':'HER2', ' Her2': 'HER2'}})
df_clinical_data = df_clinical_data.replace({'subtype': {'Basal':3, 'HER2':2, 'LumB':1, 'LumA':0 }})


# In[ ]:


df_clinical_data


# In[ ]:


# save data as dict 
with open(f'/data/Elmira_Data/MALANI/cancer_data/clinical_data/clean_data/breast_clinical_data.pkl', 'wb') as fp:
    pickle.dump(df_clinical_data, fp)


# In[ ]:


# merge clinical data and gene experession
cancer_GE_Clinical_data = pd.DataFrame()
cancer_GE_Clinical_data = df_genes.merge(df_all_clinical, on=['sample_id'])

# remove dublicated rowes
cancer_GE_Clinical_data = cancer_GE_Clinical_data.drop_duplicates(['sample_id'], keep='last')
cancer_GE_Clinical_data.shape


# In[ ]:


# Tumour samples
indexName = df_all_data[df_all_data['tissue'] == 1].index
df_all_cancer = df_all_data.loc[indexName]
df_all_cancer.to_pickle('df_tumour_expression.pkl')
df_all_cancer.to_csv('df_tumour_expression.csv')


# In[ ]:


# Normal samples
indexNames = df_all_data[df_all_data['tissue'] == 0].index
df_all_normal = df_all_data.loc[indexNames]
df_all_normal.to_pickle('df_normal_sample_expression.pkl')
df_all_normal.to_csv('df_normal_sample_expression.csv')

