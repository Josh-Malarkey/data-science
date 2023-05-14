from IPython.display import Image
from IPython.core.display import HTML 
from collections import defaultdict
import numpy as np
import random as py_random
import numpy.random as np_random
import time
import seaborn as sns
import matplotlib.pyplot as plt
from mycolorpy import colorlist as mcp
import pandas as pd
import pandasql as sqldf
import patsy
import random
import scipy.stats as stats
import sklearn.model_selection as model_selection
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
import sklearn.linear_model as linear
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SequentialFeatureSelector as sfs
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tabulate import tabulate
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate


def plot_log_odds_vs_numeric(df, target_variable, numeric_fields, multiple='dodge', figsize=None, 
                                          palette=None, kde=True):
    plots = len(numeric_fields)
    if plots < 3:
        cols = plots
        rows = 1
    else:
        cols = 3
        rows = int(np.ceil(plots / cols))
    if figsize is None:
        x = cols*5
        y = rows*3.5
        figsize=(x, y)
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=figsize, layout='tight')
    if palette is None:
        palette = ['cornflowerblue', 'maroon']
    for ax, field in zip(axs.flat, numeric_fields):
        if field is not None:
            new_df = pd.DataFrame(df)
            X = new_df[field]
            y = new_df[target_variable]
            logit_results = sm.GLM(y, X, family=sm.families.Binomial()).fit()
            predicted = logit_results.predict(X)
            log_odds = np.log(predicted / (1 - predicted))
            a, b = np.polyfit(df[field], log_odds, 1)
            ax.scatter(x=df[field], y=log_odds, color=palette[0], label='observations')
            ax.plot(df[field], a*df[field]+b, color=palette[1], label='line of best fit')     
            ax.set_title(f'\n{field} variable')
            ax.set_ylabel('Log Odds')
            ax.legend()
        ax.grid(False)
    plt.show()
    

def categorical_vs_numeric_mult_histogram(data, categorical_field, numeric_fields, 
                                          data_labels, multiple='dodge', figsize=None, 
                                          palette=None, kde=True):
    plots = len(numeric_fields)
    if plots < 3:
        cols = plots
        rows = 1
    else:
        cols = 3
        rows = int(np.ceil(plots / 3))
    if figsize is None:
        x = cols*5
        y = rows*3.5
        figsize=(x, y)
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=figsize, layout='constrained')
    if palette is None:
        palette = ['maroon', '#2ca02c']
    i = 0
    for ax, field in zip(axs.flat, numeric_fields):
        if field is not None:
            sns.histplot(data=data, x=field, hue=categorical_field, 
                         kde=kde, palette=palette, multiple=multiple, bins='auto', alpha=0.5, ax=ax)
            ax.legend(data_labels)
            ax.set_title(f'\n{field} variable')
            stats = data[field].describe(exclude=['name']).round(2)
            # fig.text(i/x + i/rows, i/y + i/cols, f'{stats}')
            # print(i/rows, i/cols)
            # i+=1
        ax.grid(False)
    plt.show()


def descriptive_numeric(dataframe, col_name, precision=3):
    ds = pd.DataFrame(dataframe[col_name].describe())
    parametric_stats = {'Range': ds.loc[['max']].values[0] - ds.loc[['min']].values[0],
                        'IQR': + ds.loc[['75%']].values[0] - ds.loc[['25%']].values[0],
                        'COV': + ds.loc[['std']].values[0] / ds.loc[['mean']].values[0]
                       }
    all_stats = pd.concat([ds, pd.DataFrame(parametric_stats).T], axis=1)
    index = {0:'Count', 1:'Mean', 2:'Std', 3:'Min', 4:'Q1', 5:'Med', 6:'Q3', 7:'Max', 8:'Range', 9:'IQR', 10:'COV'}
    return pd.DataFrame(all_stats.stack()).reset_index(drop=True).rename(columns={0:col_name}, index=index)


def descriptive_categorical(dataframe, col_name, cmap=None, alpha_sort=False):
    if cmap is None:
        cmap='gist_earth_r'
    data = pd.DataFrame(dataframe[col_name].value_counts(normalize=True).sort_values(ascending=False)*100).reset_index()
    data.rename(columns={'index': col_name, col_name: '% observations'}, inplace=True)
    if alpha_sort:
        data.sort_values(by=col_name, inplace=True)
    data['Cumulative %'] = data['% observations'].cumsum()
    data_styled = data.style.background_gradient(cmap=cmap, subset=['% observations','Cumulative %'])\
        .format(formatter={'% observations': "{:.2f}%", 'Cumulative %':"{:.2f}%"})\
        .hide_index()
    return data_styled

    
def measure_correlation(x, y):
    print("Correlation Statistics")
    print("----------------------")
    print( "r   =", "{:.4f}".format(stats.pearsonr(x, y)[0]))
    print( "rho =", "{:.4f}".format(stats.spearmanr(x, y)[0]))
    
    
def describe_categorical_numeric(data, numeric, categorical, transpose=False):
    grouped = data.groupby(categorical)
    grouped_y = grouped[numeric].describe()
    if transpose:
        return pd.DataFrame(grouped_y.transpose())
    else:
        return pd.DataFrame(grouped_y)
    
    
def get_outliers(model, conditions=None):
    """
    model = statsmodels.api model
    std_threshold = # of standard deviations for outlier threshold determination
    """
    
    summary_df = model.get_influence().summary_frame()
    results_df = summary_df[['cooks_d']]
    results_df['cooks_threshold'] = (4 / model.nobs)
    results_df['std_residual'] = stats.zscore(model.resid_pearson)
    results_df['std_residual'] = np.abs(results_df['std_residual'])
    results_df['std_threshold'] = 3
    choices = ['outlier']
    if conditions is None:
        conditions = [(results_df['cooks_d'] > (4 / model.nobs)) & (results_df['std_residual'] > 3)]
    results_df['outlier'] = np.select(conditions, choices, default='not outlier')
    return results_df


def plot_residuals(model, title, stat='count', conditions=None, palette=['#1E88E5', '#D81B60']):
    """
    model = statsmodels.api model
    std_threshold = # of standard deviations for outlier threshold determination
    """
    results_df = get_outliers(model, conditions)
    
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(12,5))
    sns.histplot(data=results_df, x='std_residual', hue='outlier', stat=stat, palette=palette, bins='auto', ax=ax[0], kde=True)
    ax[0].set_xlabel('Residual Standard Deviation')
    ax[0].grid(False)
    sns.histplot(data=results_df, x='cooks_d', hue='outlier', stat=stat, palette=palette, bins='auto', ax=ax[1], kde=True)
    ax[1].set_xlabel('Residual Cooks Distance')
    ax[1].grid(False)
    fig.suptitle(title)
    plt.show()
    
    
def get_model_vif(X_variables_as_df):
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_variables_as_df.columns
    vif_data["VIF"] = [variance_inflation_factor(X_variables_as_df.values, i) for i in range(len(X_variables_as_df.columns))]
    return vif_data.sort_values(by='VIF', ascending=False)


def plot_correlation(X_variables_as_df, cmap=None, annot=True, size=None):
    if cmap is None:
        cmap = 'Blues'
    if size is None:
        size = (10,6)
    fig, ax = plt.subplots(figsize=size)
    sns.heatmap(X_variables_as_df.corr(), annot=annot, cmap=cmap, ax=ax)
    ax.set_title('Correlation Matrix')
    plt.show()
    plt.close()
    
    
def cross_validation(model, _X, _y, _cv=5):
    """
    Source: https://www.section.io/engineering-education/how-to-implement-k-fold-cross-validation
    Parameters to perform 5 Folds Cross-Validation
     ----------
    model: Python Class, default=None
            This is the machine learning algorithm to be used for training.
    _X: array
         This is the matrix of features.
    _y: array
         This is the target variable.
    _cv: int, default=5
        Determines the number of folds for cross-validation.
     Returns
     -------
     The function returns a dictionary containing the metrics 'accuracy', 'precision',
     'recall', 'f1' for both training set and validation set.
    """
    _scoring = ['accuracy', 'precision', 'recall', 'f1']
    results = cross_validate(estimator=model, X=_X, y=_y, cv=_cv, scoring=_scoring, return_train_score=True)

    return {"Training Accuracy scores": results['train_accuracy'],
            "Mean Training Accuracy": results['train_accuracy'].mean() * 100,
            "Training Precision scores": results['train_precision'],
            "Mean Training Precision": results['train_precision'].mean(),
            "Training Recall scores": results['train_recall'],
            "Mean Training Recall": results['train_recall'].mean(),
            "Training F1 scores": results['train_f1'],
            "Mean Training F1 Score": results['train_f1'].mean(),
            "Validation Accuracy scores": results['test_accuracy'],
            "Mean Validation Accuracy": results['test_accuracy'].mean() * 100,
            "Validation Precision scores": results['test_precision'],
            "Mean Validation Precision": results['test_precision'].mean(),
            "Validation Recall scores": results['test_recall'],
            "Mean Validation Recall": results['test_recall'].mean(),
            "Validation F1 scores": results['test_f1'],
            "Mean Validation F1 Score": results['test_f1'].mean()
            }


# Grouped Bar Chart for both training and validation data
def plot_result(x_label, y_label, plot_title, train_data, val_data, palette=None):
    """
    Source: https://www.section.io/engineering-education/how-to-implement-k-fold-cross-validation
    Parameters to plot a grouped bar chart showing the training and validation
    results of the ML model in each fold after applying K-fold cross-validation.
     ----------
     x_label: str,
        Name of the algorithm used for training e.g 'Decision Tree'

     y_label: str,
        Name of metric being visualized e.g 'Accuracy'
     plot_title: str,
        This is the title of the plot e.g 'Accuracy Plot'

     train_result: list, array
        This is the list containing either training precision, accuracy, or f1 score.

     val_result: list, array
        This is the list containing either validation precision, accuracy, or f1 score.
     palette: array of colors
        defaults to blue, red
     Returns
     -------
     The function returns a Grouped Barchart showing the training and validation result
     in each fold.
    """
    if palette is None:
        palette = ['#004968', '#929292']
    # Set size of plot
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = labels = ['Fold ' + str(i+1) for i in range(len(train_data))]
    
    X_axis = np.arange(len(labels))
    ax.set_ylim(0.25, 1)
    for i, v in enumerate(train_data):
        ax.text(X_axis[i] - 0.3, v + 0.02, str(np.round(v, 2)), color=palette[0], fontweight='bold')
    for i, v in enumerate(val_data):
        ax.text(X_axis[i] + 0.1, v + 0.02, str(np.round(v, 2)), color=palette[1], fontweight='bold')
    ax.bar(X_axis - 0.2, train_data, 0.4, color=palette[0], label='Training')
    ax.bar(X_axis + 0.2, val_data, 0.4, color=palette[1], label='Validation')
    ax.set_title(plot_title, fontsize=22)
    ax.set_xticks(X_axis, labels)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.legend()
    ax.grid(False)
    plt.show()
    
    
def plot_confusion_matrix(X, y, labels, model_name, algo=None, test_size=0.2, cmap=None):
    if algo is None:
        algo = linear_model.LogisticRegression(fit_intercept=True)
    if cmap is None:
        cmap = plt.cm.get_cmap('cividis').reversed()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    algo.fit(X_train, y_train)
    pred = algo.predict(X_test)
    score = np.round(accuracy_score(y_test, pred), 2)
    cm = confusion_matrix(y_test, algo.predict(X_test))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap=cmap)
    plt.title(model_name + ' Accuracy: ' + str(score*100) + '%')
    plt.xlabel('Predicted Outcome')
    plt.ylabel('Actual Outcome')
    plt.show()
    
    
def plot_k_fold_validation(algo, results_df, model_name, model_type, X, y, palette=None, folds=5):
    if palette is None:
        palette= ['#1f77b4', '#aec7e8']
    plot_title = model_name + ' ' + model_type + ': ' + str(folds) + '-Fold Validation'
    validation = cross_validation(algo.fit(X, y), X, y, folds)
    train_accuracy = np.round(np.mean(validation['Training Accuracy scores'])*100,2)
    test_accuracy = np.round(np.mean(validation['Validation Accuracy scores'])*100,2)
    print('Mean Training Accuracy score: ', str(train_accuracy))
    print('Mean Validation Accuracy score: ', str(test_accuracy))
    plot_result(model_name + ' ' + model_type , 'Accuracy', plot_title, validation['Training Accuracy scores'],
                                 validation['Validation Accuracy scores'], palette=palette)
    results_df.loc[len(results_df.index)] = [model_type, model_name, train_accuracy, test_accuracy]
    

def stepwise_feature_selection(model, X, y, scoring='precision', n_features='auto'):
    """
    model: sklearn model
    """
    selector = sfs(model, n_features_to_select=n_features, scoring=scoring)
    results = selector.fit(X, y)

    return {"Feature Count": results.n_features_to_select_,
            "Feature Names": selector.get_feature_names_out(),
            "Reduced Features": selector.transform(X)
            }