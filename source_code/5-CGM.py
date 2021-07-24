#!/usr/bin/env python
# coding: utf-8


import random
import collections
import numpy as np
import pandas as pd

# Plotting
import seaborn as sns
from matplotlib import pyplot
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from ggplot import *
import chart_studio.plotly as py
get_ipython().run_line_magic('matplotlib', 'inline')

from scipy.stats import chi2_contingency
from matplotlib import rc

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_regression,  mutual_info_classif, chi2 
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.decomposition import PCA
from sklearn.metrics import matthews_corrcoef
from sklearn import metrics
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    silhouette_score,
    silhouette_samples,
    calinski_harabasz_score,
    davies_bouldin_score,
    roc_curve,
    matthews_corrcoef,
    auc,
)

from sklearn.model_selection import cross_validate
import scipy

# include if using a Jupyter notebook
get_ipython().run_line_magic('matplotlib', 'inline')

import pickle
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, plot_importance


OUTLIER_DETECTION = 'YES'     # ['YES', 'NO']
NUM_RUNS = 10
MAIN_DIR = '.'

df_final_prediction = None
plt.rcParams.update({'font.size': 16})
np.random.seed(40)

df_total = pd.read_pickle(f'{MAIN_DIR}/main_data/tumour_gene_expression_and_clinical.pkl')

def long_rank(df, label):
    T1= df[df[label]==0]['t.rfs']
    E1=df[df[label]==0]['e.rfs']

    T2= df[df[label]==1]['t.rfs']
    E2=df[df[label]==1]['e.rfs']
    return (T1,E1,T2,E2)

def load_analyzed_data_from_R():
    df_total = None

    print('Loading cancer data...')
    
    # load all cancer data
    df_total = pd.read_pickle(f'{MAIN_DIR}/main_data/tumour_gene_expression_and_clinical.pkl')

    # load all normal data
    df_total_normal = pd.read_pickle(f'{MAIN_DIR}/main_data/normal_gene_expression_and_clinical.pkl')

    num_normal = df_total_normal.shape[0]
    num_tumor = df_total.shape[0]
    print(f"\x1b[1;30m Normal: {num_normal}     Tumor: {num_tumor} \x1b[0m")

    # reset index
    df_total.set_index("sample_id")
    df_total_normal.set_index("sample_id")

    return df_total_normal, df_total


def clean_data(df_total, num_of_genes_to_select=None):
    # deleting AFF
    unwanted_Aff = df_total.filter(regex="^AFFX")
    print(f'Removing {unwanted_Aff.shape} rows with AFFX from df_total')
    df_total.drop(unwanted_Aff, axis=1, inplace=True)

    # deleting -at
    unwanted_at = df_total.filter(regex="_at")
    print(f'Removing {unwanted_at.shape} rows with _at from df_total')
    df_total.drop(unwanted_at, axis=1, inplace=True)

    df_clean = df_total.drop(columns=["sample_id"])
        
    # remove normal cells
    df_clean = df_clean[df_clean.tissue != 0]
  
    # CGM genes
    with open (f'{MAIN_DIR}/main_data/CGM_biomarkers.pkl', 'rb') as fp:
        gene_list = pickle.load(fp)
        gene_list_selected_number = gene_list[0:num_of_genes_to_select]
        
    annotations = ['tissue', 'stage', 'grade','subtype','PAM50','SCMOD2', 'tumor size(mm)', 'age at diagnosis(yrs)',
                   'death event time(yrs)', 'death event', 'er', 'pr', 'her2', 't.rfs', 'e.rfs', 't.tdm','e.tdm',
                   'EndoPredict','oncotypedx','GGI']
    
    df_genes = gene_list_selected_number + annotations
    df_clean = df_clean[df_genes]
       
    print('Setting aside grade 2 and null-grades for final prediction')
    global df_final_prediction
    df_final_prediction = df_clean[(df_clean["grade"] == 2) | (df_clean["grade"].isnull())].copy()
    
    print('Selecting just columns that have grade')
    df_clean = df_clean.loc[df_clean.grade.notnull()]
    print(f'shape of df_final_prediction: {df_final_prediction.shape}')
  
    df_clean = df_clean[df_clean.grade != 2]
    df_clean = df_clean.replace({"grade": {1: 0}})
    df_clean = df_clean.replace({"grade": {3: 1}})    
    
    print (f"Num of each grade: {df_clean['grade'].value_counts()}") 
    
    return df_clean


def read_input(num_of_genes_to_select=None):
    df_total_normal, df_total = load_analyzed_data_from_R()
    
    return clean_data(df_total_normal, num_of_genes_to_select), clean_data(df_total, num_of_genes_to_select)


df_total_normal, df_total = load_analyzed_data_from_R()
df_clean_normal, df = read_input()


# cleaning data
df_total = df_total.replace({'er':{' NA':np.nan,' EV':np.nan, '1':1, '0':0},
                               'pr':{' NA':np.nan,' EV':np.nan, '1':1, '0':0},
                               'her2':{' NA':np.nan,' EV':np.nan, '1':1, '0':0},
                               'stage':{0:np.nan, ' NA':np.nan, 0:np.nan},
                               'tumor size(mm)':{'>= 20':25, '< 20':10, ' --':np.nan, ' NA':np.nan },
                                'subtype':{' NA':np.nan},
                                 'age at diagnosis(yrs)':{' NA':np.nan, '>=50':60, '<50':40}})

df_total['age at diagnosis(yrs)'] = pd.to_numeric(df_total['age at diagnosis(yrs)'])
df_total['tumor size(mm)'] = pd.to_numeric(df_total['tumor size(mm)'])
df_total.loc[df_total['tumor size(mm)'] < 20, 'tumor size(mm)'] = 0
df_total.loc[df_total['tumor size(mm)'] >= 20, 'tumor size(mm)'] = 1

df_total.loc[df_total['age at diagnosis(yrs)'] < 50, 'age at diagnosis(yrs)'] = 0
df_total.loc[df_total['age at diagnosis(yrs)'] >= 50, 'age at diagnosis(yrs)'] = 1

df_total['stage'] = pd.to_numeric(df_total['stage'])
df_total['subtype'] = pd.to_numeric(df_total['subtype'])


# ### Developing ML model 

# CGM biomarkers
with open(f'{MAIN_DIR}/main_data/CGM_biomarkers.pkl', 'rb') as f:
    all_genes = pickle.load(f)
best_CGM = all_genes[:70]


# prediction dataset (grade 2 and unknown grade samples )
df_2_unkown = df_total[~df_total['grade'].isin([1,3])]
# developmental dataset (grade 1 and 2)
df_3_1 = df_total[df_total['grade'].isin([1,3])]

selected_features = best_CGM + ['grade']
df_developmental = df_3_1[selected_features]
df_prediction = df_2_unkown[selected_features]


### Function

def perform_outlier_detection(X, X_val):
    if OUTLIER_DETECTION == 'YES':
        import pyod
        from pyod.models.knn import KNN
        from pyod.utils.data import evaluate_print
        from pyod.utils.example import visualize

        outlier_fraction = 0.05

        clf = KNN(contamination = outlier_fraction)
        clf.fit(X)

        # get the prediction labels and outlier scores of the training data
        y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
        y_train_scores = clf.decision_scores_  # raw outlier scores

        # get the prediction on the val data
        y_val_pred = clf.predict(X_val)  # outlier labels (0 or 1)
        y_val_scores = clf.decision_function(X_val)  # outlier scores

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        pca = PCA(n_components=2)
        X_val_pca = pca.fit_transform(X_val)

        # visualize the results
        visualize('KNN', X_pca, y_train_pred, X_val_pca, y_val_pred, y_train_pred,
                  y_val_pred, show_figure=True, save_figure=True)
        
        train_cnt = np.bincount(y_train_pred)
        val_cnt = np.bincount(y_val_pred)
        print(f'# of inliers in train set: {train_cnt[0]}, # of outliers in train set: {train_cnt[1]}')
        print(f'# of inliers in val set: {val_cnt[0]},     # of outliers in val set: {val_cnt[1]}')
        
        return y_train_pred, y_val_pred, clf
    else:
        print('Outlier detection is disabled in code options.')
        return None, None, None


def cross_validate(model, X, y, NUM_RUNS):
    cv_results = cross_validate(model, X, y, cv=NUM_RUNS)
    return cv_results

def balance_df(X, y, random_state):
    print(f"Before Balancing, counts of label '1': {sum(y == 1)}")
    print(f"Before Balancing, counts of label '0': {sum(y == 0)}\n")

    saved_cols = X.columns
    if random_state:
        sm = SMOTE(random_state=42)
    else:
        sm = SMOTE()
        
    X_res, y_res = sm.fit_sample(X.copy(), y.copy().ravel())

    print(f"After Balancing, counts of label '1': {sum(y_res == 1)}")
    print(f"After Balancing, counts of label '0': {sum(y_res == 0)}")

    return pd.DataFrame(X_res, columns=saved_cols), pd.Series(y_res)


def select_k_best(X, y, num_features, fs):
     # fs f_classif ---> [f_classif, f_regression, mutual_info_regression,  mutual_info_classif, chi2 ]
        
    select_k_best_classifier = SelectKBest(score_func=fs, k=num_features)
    X_new = select_k_best_classifier.fit_transform(X, y)

    mask = select_k_best_classifier.get_support() #list of booleans
    new_features = [] # The list of your K best features

    for bool, feature in zip(mask, X.columns):
        if bool:
            new_features.append(feature)

    X_new = X[np.intersect1d(X.columns, new_features)]
    # Selecting important genes in test set: X_test_new
    # X_test_new = X_test[np.intersect1d(X_test.columns, new_features)]
    
    return num_features, new_features, X_new

def select_lasso(X_filtered, y):
    # We use the base estimator LassoCV since the L1 norm promotes sparsity of features.
    clf = LassoCV(tol=0.01)

    # Set a minimum threshold of 0.25
    sfm = SelectFromModel(clf)
    sfm.fit(X_filtered, y)
    X_new = sfm.transform(X_filtered)
    
    n_features = X_new.shape[1]
    
    new_features = []

    for i in range(n_features):
        for col in X_filtered:
            if np.array_equal(X_filtered[col].as_matrix(), X_new[:, i]):
                new_features.append(col)     
    
    print (new_features)
    # Selecting important genes in test set: X_test_new
    # X_test_new = X_test_Normalized[np.intersect1d(X_test_Normalized.columns, new_features)]
    return len(new_features), new_features, X_new


def calculate_metrics(y, y_pred):
    result = {}
    result['accuracy'] = metrics.accuracy_score(y, y_pred)

    try:
        result['ROC'] = metrics.roc_auc_score(y, y_pred)
    except:
        pass

    result['precision'] = precision_score(y, y_pred, average='macro')
    result['recall'] = recall_score(y, y_pred, average='macro')
    result['f1_score'] = f1_score(y, y_pred, average='macro')                                  
    result['matthews_corrcoef'] = matthews_corrcoef(y, y_pred)
    try:
        TN = confusion_matrix(y, y_pred)[0,0]
        FP = confusion_matrix(y, y_pred)[0,1]
        specifity = TN/(TN+FP)
        dic['specifity'] = specifity
    except:
        pass

    result['confusion_matrix'] = confusion_matrix(y, y_pred)
    
    try:
        fpr, tpr, _ = roc_curve(y, y_pred)
        roc_auc = auc(fpr, tpr)
        result['fpr'] = fpr
        result['tpr'] = tpr
        result['roc_auc'] = roc_auc
    except:
        pass
    
    return result

def plot_ROC_curve(r):
    plt.figure(figsize=(8, 6), dpi=100)
    plt.plot(r['fpr'], r['tpr'], color='darkorange', lw=2, label='Neural Network ROC curve (area = %0.2f)' % r['roc_auc'])
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    return plt

def plot_learning_curve(model):
    # retrieve performance metrics
    results = model.evals_result()
    epochs = len(results['validation_0']['error'])
    x_axis = range(0, epochs)
    # plot log loss
    fig, ax = pyplot.subplots()
    ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
    ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
    ax.legend()
    pyplot.ylabel('Log Loss')
    pyplot.title('XGBoost Log Loss')
    pyplot.show()
    # plot classification error
    fig, ax = pyplot.subplots()
    ax.plot(x_axis, results['validation_0']['error'], label='Train')
    ax.plot(x_axis, results['validation_1']['error'], label='Test')
    ax.legend()
    pyplot.ylabel('Classification Error')
    pyplot.title('XGBoost Classification Error')
    pyplot.show()

def run_all(df):
    scoring = ['precision_macro', 'recall_macro']
    test_results = []
    random_state = None
    
    
    df = df.replace({'grade':{1:0}})
    df = df.replace({'grade':{3:1}})

    best_accuracy, best_model = 0, None
    for exp_id in range(0, NUM_RUNS):
        print(f'\n==========================')
        print(f'========= Run {exp_id} ==========')
        print(f'==========================')

        # split into train and test
        if random_state:
            temp_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
            train_df, val_df = train_test_split(temp_df, test_size=0.1, random_state=42)
        else:
            temp_df, test_df = train_test_split(df, test_size=0.1)
            train_df, val_df = train_test_split(temp_df, test_size=0.1)

        train_df_size_1 = train_df[train_df['grade']==0].shape[0] 
        train_df_size_3 = train_df[train_df['grade']==1].shape[0] 
        print(f'train_df size: {len(train_df)}, val_df size: {len(val_df)}, test_df size: {len(test_df)}')
        print(f'grade-1 : {train_df_size_1}, grade-3: {train_df_size_3}')
        y = train_df['grade'].copy()
        train_df.drop(columns=['grade'], inplace=True, errors='ignore')

        y_val = val_df['grade'].copy()
        val_df.drop(columns=['grade'], inplace=True, errors='ignore')

        y_test = test_df['grade']
        test_df.drop(columns=['grade'], inplace=True, errors='ignore')

        # outlier detection
        if OUTLIER_DETECTION == 'YES':
            print(f'Shape of train_df before OD: {train_df.shape}, val_df: {val_df.shape}')            
            y_train_pred, y_val_pred, od_model = perform_outlier_detection(train_df, val_df)

            X_mask = y_train_pred == 1
            train_df = train_df[~X_mask]
            y = y[~X_mask]

            X_val_mask = y_val_pred == 1
            val_df = val_df[~X_val_mask]
            y_val = y_val[~X_val_mask]
            print(f'Shape of train_df after OD: {train_df.shape}, val_df: {val_df.shape}')

        # balancing
        X_balanced, y_balanced = balance_df(train_df, y, random_state)
        train_df = X_balanced
        new_features = []  

        classifier = 'XGBoost'
        dic = {'classifier': classifier}

        eval_set = [(train_df, y_balanced), (val_df, y_val)]
        eval_metric = ["error", "logloss"]

        model = XGBClassifier(colsample_bytree=0.8, gamma=0.5, max_depth=5, min_child_weight=1, subsample=0.6)
        model.fit(train_df, y_balanced, eval_metric=eval_metric, eval_set=eval_set, early_stopping_rounds=10, verbose=False)

        plot_learning_curve(model)

        y_pred = model.predict(train_df)
        train_result = calculate_metrics(y_balanced, y_pred)
        print(f'\ntrain_result: {train_result}')

        y_test_pred = model.predict(test_df)        
        test_result = calculate_metrics(y_test, y_test_pred)
        print(f'\ntest_result: {test_result}')

        test_result['selected_features'] = new_features
        test_result['model'] = model

        top_features = model.get_booster().get_score(importance_type='weight')
        test_result['top_features_weight'] = top_features
        top_features = model.get_booster().get_score(importance_type='gain')
        test_result['top_features_gain'] = top_features
        top_features = model.get_booster().get_score(importance_type='cover')
        test_result['top_features_cover'] = top_features

        test_results.append(test_result)
        
    

        if test_result['accuracy'] > best_accuracy:
            best_model = model
        
    return test_df, test_results, best_model, best_accuracy


test_df, test_results, best_model, best_accuracy = run_all(df_developmental)

df_test_results = pd.DataFrame(test_results)

df_test_results['f1_score'].mean()


# ## Feature importance

all_top_features_weight = collections.defaultdict(int)
for run in df_test_results['top_features_weight']:
    for gene, metric in run.items():
        all_top_features_weight[gene] += metric
sorted_features_lst_weight = sorted(all_top_features_weight.items(), key=lambda kv: kv[1], reverse=True)

df_weight = pd.DataFrame(sorted_features_lst_weight)
df_weight.columns = ['gene', 'weight']
df_weight.set_index('gene', inplace=True)

all_top_features_gain = collections.defaultdict(int)
for run in df_test_results['top_features_gain']:
    for gene, metric in run.items():
        all_top_features_gain[gene] += metric
sorted_features_lst_gain = sorted(all_top_features_gain.items(), key=lambda kv: kv[1], reverse=True)

df_gain = pd.DataFrame(sorted_features_lst_gain)
df_gain.columns = ['gene', 'gain']
df_gain.set_index('gene', inplace=True)

all_top_features_cover = collections.defaultdict(int)
for run in df_test_results['top_features_cover']:
    for gene, metric in run.items():
        all_top_features_cover[gene] += metric      
sorted_features_lst_cover = sorted(all_top_features_cover.items(), key=lambda kv: kv[1], reverse=True)

df_cover = pd.DataFrame(sorted_features_lst_gain)
df_cover.columns = ['gene', 'cover']
df_cover.set_index('gene', inplace=True)


df_total = pd.merge(df_weight, df_gain, left_index=True, right_index=True)
df_total = pd.merge(df_total, df_cover, left_index=True, right_index=True)
df_total.head()

# shap is for grade2 and unkown howeve we do it here
shap.initjs()

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(test_df)
shap.summary_plot(shap_values, test_df, max_display=74, show=False)

