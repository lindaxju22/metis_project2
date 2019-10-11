#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 16:58:37 2019

@author: lindaxju
"""
#%%
import pickle
import csv
from datetime import datetime
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.linear_model import lars_path
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import scipy.stats as stats
from scipy.stats import boxcox
#%%
def remove_outliers(dataframe, col_name):
    Q1 = dataframe[col_name].quantile(0.25)
    Q3 = dataframe[col_name].quantile(0.75)
    IQR = Q3 - Q1
    
    dataframe_clean = dataframe[~((dataframe[col_name] < (Q1 - 1.5 * IQR)) |(dataframe[col_name] > (Q3 + 1.5 * IQR)))]
    
    return dataframe_clean
#%%
def get_hist(dataframe, col_name, y_name):
    plt.hist(dataframe[col_name], bins=30)
    plt.title('histogram of ' + col_name)
    plt.xlabel(col_name)
    plt.ylabel('counts')
    plt.grid(axis='y', alpha=0.75)
#%%
def get_boxplot(dataframe, col_name):
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=dataframe.fan_score)
#%%
def get_transformation(df,features_orig,features_boxcox,features_log):
    df_hist_transformed = df.copy()
    features = features_orig.copy()
    if len(features_boxcox)>0:
        for col in features_boxcox:
            df_hist_transformed[col] = boxcox(df[col]+.01)[0]
            df_hist_transformed.rename(columns={col: col+'_bc'}, inplace=True)
            features.remove(col)
            features.append(col+'_bc')
    if len(features_log)>0:
        for col in features_log:
            df_hist_transformed[col] = np.log(df[col]+.01)
            df_hist_transformed.rename(columns={col: col+'_log'}, inplace=True)
            features.remove(col)
            features.append(col+'_log')
    return df_hist_transformed, features
#%%
# Find the MAE and R^2 on the test set using this model
def mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))
#%%
## Note: lars_path takes numpy matrices, not pandas dataframes
def get_lars_path(X_trval,y_train_val,method):
    alphas, _, coefs = lars_path(X_trval, y_train_val.values, method=method)
    # plotting the LARS path
    xx = np.sum(np.abs(coefs.T), axis=1)
    xx /= xx[-1]
    
    plt.figure(figsize=(10,10))
    plt.plot(xx, coefs.T)
    ymin, ymax = plt.ylim()
    plt.vlines(xx, ymin, ymax, linestyle='dashed')
    plt.xlabel('|coef| / max|coef|')
    plt.ylabel('Coefficients')
    plt.title(method + ' path')
    plt.axis('tight')
    plt.legend(X_train_val.columns)
    plt.show()
#%%
# Run the cross validation, find the best alpha, refit the model on all the data 
# with that alpha
def lm_reg(method,alphavec,X_trval,y_train_val):
    if method == 'lasso':
        model = LassoCV(alphas = alphavec, cv=5)
    elif method == 'ridge':
        model = RidgeCV(alphas = alphavec, cv=5)
    model.fit(X_trval, y_train_val)
    return model
#%%
def plot_actual_vs_predicted(y_actual,y_predict):
    plt.scatter(y_actual,y_predict,color='black',alpha=0.1)
    plt.plot([0,70],[0,70],color='blue')
    plt.title('predicted vs. actual fantasy points')
    plt.xlabel('actual')
    plt.ylabel('predicted')
    plt.show();
#%%
def diagnostic_plot(col, x, y):
    plt.figure(figsize=(18,3))
    
    rgr = LinearRegression()
    rgr.fit(x,y)
    pred = rgr.predict(x)

    plt.subplot(1, 4, 1)
    plt.hist(x, bins=30)
    plt.title('histogram of ' + str(col))
    plt.xlabel(col)
    plt.ylabel('counts')
    plt.grid(axis='y', alpha=0.75)

    plt.subplot(1, 4, 2)
    plt.scatter(x,y)
    plt.plot(x, pred, color='blue',linewidth=1)
    plt.title("Regression fit")
    plt.xlabel(col)
    plt.ylabel("fantasy pts")
    
    plt.subplot(1, 4, 3)
    res = y - pred
    plt.scatter(pred, res)
    plt.title("Residual plot")
    plt.xlabel("prediction")
    plt.ylabel("residuals")
    
    plt.subplot(1, 4, 4)
    #Generates a probability plot of sample data against the quantiles of a 
    # specified theoretical distribution 
    stats.probplot(res, dist="norm", plot=plt)
    plt.title("Normal Q-Q plot")
#%%
#%reset
#df_all_big.to_csv(r'df_all_big.csv')
with open('data/df_all_2019-10-10-11-14-32.pickle','rb') as read_file:
    df_all = pickle.load(read_file)
#df_all = pd.read_csv('data/df_all.csv')
#df_all.drop([df_all.columns[0]],axis='columns',inplace=True)
#with open('df_all_big.pickle', 'wb') as to_write:
#    pickle.dump(df_all_big, to_write)
#%%
df_all.shape
#%%
df_all.head()
#%%
list(df_all.columns)
#%%
df_all.info()
#%%
#df_all.dropna(inplace=True)
#df_all.info()
#%%
features_orig = ['home','min_SMA','pts_SMA','fgm_SMA','fga_SMA','fgpct_SMA','3pm_SMA',
                 '3pa_SMA','3ppct_SMA','ftm_SMA','fta_SMA','ftpct_SMA','oreb_SMA',
                 'dreb_SMA','reb_SMA','ast_SMA','stl_SMA','blk_SMA','tov_SMA','pf_SMA',
                 'pm_SMA','fan_score_SMA','offrtg_SMA','defrtg_SMA','netrtg_SMA',
                 'usgpct_SMA','pace_SMA','pie_SMA','offrtg_SMA_opp','defrtg_SMA_opp',
                 'netrtg_SMA_opp','pace_SMA_opp','pie_SMA_opp','days_rest']
len(features_orig) # removes 4 columns not needed for regression and 1 target column
#%%
# Check out distributions
df_all.iloc[:,5:].hist(bins=30,figsize=(20,20));
#%%
############################### Transformations ###############################
features_boxcox = ['3pa_SMA','3pm_SMA','3ppct_SMA','ast_SMA','dreb_SMA',
                   'fta_SMA','ftm_SMA','ftpct_SMA','oreb_SMA','pts_SMA',
                   'reb_SMA','stl_SMA','tov_SMA']
features_log = ['days_rest']
#%%
df_hist_transformed, features = get_transformation(df_all,features_orig,features_boxcox,features_log)
#%%
# Check out distributions
df_hist_transformed.iloc[:,5:].hist(bins=30,figsize=(20,20));
#%%
############################# Outliers in Target #############################
# Check out distribution of target
get_hist(df_all,'fan_score','fan_score')
#%%
# Check out outliers
get_boxplot(df_all,'fan_score')
#%%
df_all_clean = remove_outliers(df_all,'fan_score')
#%%
# Check out distribution of fantasy points
get_hist(df_all_clean,'fan_score','fantasy pts')
#%%
# Check out outliers
get_boxplot(df_all_clean,'fan_score')
#%%
############################### Decide Final df ###############################
#df_final = df_hist_transformed.copy()
#df_final = df_all_clean.copy()

df_final = df_all.copy()
features = features_orig.copy()

#df_final.dropna(inplace = True)
#%%
############################### Model Final df ###############################
X = df_final.loc[:,features]
y = df_final['fan_score']
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
## Scale the data
std = StandardScaler()
std.fit(X_train_val.values)
## Scale the Predictors on both the train and test set
X_trval = std.transform(X_train_val.values)
X_te = std.transform(X_test.values)
#%%
lasso_alphavec = 10**np.linspace(-2,2,1000)
ridge_alphavec = 10**np.linspace(1,2.5,200)
lasso_model = lm_reg('lasso',lasso_alphavec,X_trval,y_train_val)
ridge_model = lm_reg('ridge',ridge_alphavec,X_trval,y_train_val)
#%%
# Best alpha
print(lasso_model.alpha_)
print(np.log10(lasso_model.alpha_))
print(ridge_model.alpha_)
print(np.log10(ridge_model.alpha_))
#%%
# These are the (standardized) coefficients found
# when it refit using that best alpha
list(zip(features, lasso_model.coef_))
list(zip(features, ridge_model.coef_))
#%%
# Make predictions on the test set using the new model
test_set_pred_lasso = lasso_model.predict(X_te)
test_set_pred_ridge = ridge_model.predict(X_te)
# Find the MAE and R^2 on the test set using this model
print('lasso MAE: ' + str(mae(y_test, test_set_pred_lasso)))
print('lasso R2: ' + str(r2_score(y_test, test_set_pred_lasso)))
print('ridge MAE: ' + str(mae(y_test, test_set_pred_ridge)))
print('ridge R2: ' + str(r2_score(y_test, test_set_pred_ridge)))
#%%
plot_actual_vs_predicted(y_test,test_set_pred_lasso)
#%%
#plt.scatter(y_test,test_set_pred_lasso,color='black',alpha=0.1)
#plt.plot([0,70],[0,70],color='blue')
#plt.title('predicted vs. actual fantasy points')
#plt.xlabel('actual')
#plt.ylabel('predicted')
#plt.savefig("figures/actual_predict_lasso.svg", format="svg")
#%%
plot_actual_vs_predicted(y_test,test_set_pred_ridge)
#%%
get_lars_path(X_trval,y_train_val,'lasso')
#%%
get_lars_path(X_trval,y_train_val,'ridge')
#%%
########################### Final Linear Regression ###########################
#%%
coef_final = []
features_final = []
for i in list(zip(features, lasso_model.coef_)):
    if abs(i[1]) > 0.4:
        features_final.append(i[0])
        coef_final.append(i)
print(coef_final)
print(features_final)
len(features_final)
#%%
coef_final = [(item[0],round(item[1],2)) for item in coef_final]
#%%
sorted(coef_final, key=lambda tup: tup[1], reverse = True)
#%%
sorted(coef_final, key=lambda tup: abs(tup[1]), reverse = True)
#%%
X_final = X_train_val.loc[:,features_final]
X_test_final = X_test.loc[:,features_final]
lm_final = LinearRegression()
lm_final.fit(X_final, y_train_val)
print(f'Linear Regression train R^2: {lm_final.score(X_final, y_train_val):.2f}')
fit_final = lm_final.fit(X_final, y_train_val)
yhat_final = fit_final.predict(X_test_final)
#%%
plot_actual_vs_predicted(y_test,yhat_final)
#%%
rnd = 2
#%%
print('Train R^2: ' + str(round(lm_final.score(X_final, y_train_val),rnd)))
print('Test R^2: ' + str(round(lm_final.score(X_test_final, y_test),rnd)))
#%%
for col in features_final[0:]:
    x = X_test_final.loc[:,col].to_numpy()
    diagnostic_plot(col,x.reshape(len(x),1),y_test)
#%%
print('Train Adj  R^2: ' + str(round(1-(1-lm_final.score(X_final,y_train_val))*(len(y_train_val)-1)/(len(y_train_val)-X_final.shape[1]-1),rnd)))
print('Test Adj R^2: ' + str(round(1-(1-lm_final.score(X_test_final,y_test))*(len(y_test)-1)/(len(y_test)-X_test_final.shape[1]-1),rnd)))
#%%
#%%
#%%
#%%
timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
filename = 'df_final_'+timestamp
df_all.to_csv(r'data/'+filename+'.csv')
with open('data/'+filename+'.pickle', 'wb') as to_write:
    pickle.dump(df_all, to_write)
#%%
#%%
#%%
#%%
#%%
