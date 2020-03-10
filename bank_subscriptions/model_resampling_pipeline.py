import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.model_selection import train_test_split
import pandas as pd

from sklearn import metrics 
from collections import Counter

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks, EditedNearestNeighbours


# -------------------------RESAMPLING PIPELINE ---------------------------------------------------------
def model_resampling_pipeline(X_train, X_test, y_train, y_test, model, b= 0.5):
    results = {'ordinary': {},
               'class_weight': {},
               'oversample': {},
               'undersample': {}}
    
    # ------ No balancing ------
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, predictions)
    precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_test, predictions,
                                                                                pos_label=1,
                                                                                average='binary')
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, predictions).ravel()
    fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    tnr = tn / (tn + fp)
    tpr_score = tp / (tn + tp)
    gmean = (tnr * recall)**0.5
    weighted_accuracy = (b*recall) + ((1 - b)*tnr)
    
    results['ordinary'] = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 
                                            'fscore': fscore, 'n_occurences': support,
                                            'predictions_count': Counter(predictions),
                                            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
                                            'auc': auc,
                                            'tnr': tnr, 'gmean': gmean,
                                            'weighted_accuracy': weighted_accuracy}
    
    
    # ------ Class weight ------
    if 'class_weight' in model.get_params().keys():
        model.set_params(class_weight='balanced')
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, predictions)
        precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_test, predictions, 
                                                                                     pos_label=1,
                                                                                     average='binary')
        tn, fp, fn, tp = metrics.confusion_matrix(y_test, predictions).ravel()
        fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        tnr = tn / (tn + fp)
        tpr_score = tp / (tn + tp)
        gmean = (tnr * recall)**0.5
        weighted_accuracy = (b*recall) + ((1 - b)*tnr)
    
        results['class_weight'] = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 
                                            'fscore': fscore, 'n_occurences': support,
                                            'predictions_count': Counter(predictions),
                                            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
                                            'auc': auc,
                                            'tnr': tnr, 'gmean': gmean,
                                            'weighted_accuracy': weighted_accuracy}

    
    # ------------ OVERSAMPLING TECHNIQUES ------------
    techniques = [RandomOverSampler(),
                  SMOTE(),
                  ADASYN()]
    
    for sampler in techniques:
        technique = sampler.__class__.__name__
        X_resampled, y_resampled = sampler.fit_sample(X_train, y_train)
        
        X_resampled = pd.DataFrame(X_resampled)
        X_resampled.columns = X_train.columns

        
        model.fit(X_resampled, y_resampled)
        predictions = model.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, predictions)
        precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_test, predictions,
                                                                                     pos_label=1,
                                                                                     average='binary')
        tn, fp, fn, tp = metrics.confusion_matrix(y_test, predictions).ravel()
        fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        tnr = tn / (tn + fp)
        tpr_score = tp / (tn + tp)
        gmean = (tnr * recall)**0.5
        weighted_accuracy = (b*recall) + ((1 - b)*tnr)
    
        results['oversample'][technique] = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 
                                            'fscore': fscore, 'n_occurences': support,
                                            'predictions_count': Counter(predictions),
                                            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
                                            'auc': auc,
                                            'tnr': tnr, 'gmean': gmean,
                                            'weighted_accuracy': weighted_accuracy}

    
    # ------------ UNDERSAMPLING TECHNIQUES ------------
    techniques = [RandomUnderSampler(),
                  NearMiss(version=1),
                  NearMiss(version=2),
                  TomekLinks(),
                  EditedNearestNeighbours()]
    
    for sampler in techniques:
        technique = sampler.__class__.__name__
        if technique == 'NearMiss': technique+=str(sampler.version)
        X_resampled, y_resampled = sampler.fit_sample(X_train, y_train)
        
        X_resampled = pd.DataFrame(X_resampled)
        X_resampled.columns = X_train.columns

        model.fit(X_resampled, y_resampled)
        predictions = model.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, predictions)
        precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_test, predictions,
                                                                                     pos_label=1,
                                                                                     average='binary')
        tn, fp, fn, tp = metrics.confusion_matrix(y_test, predictions).ravel()
        fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        tnr = tn / (tn + fp)
        tpr_score = tp / (tn + tp)
        gmean = (tnr * recall)**0.5
        weighted_accuracy = (b*recall) + ((1 - b)*tnr)
    
        results['undersample'][technique] = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 
                                            'fscore': fscore, 'n_occurences': support,
                                            'predictions_count': Counter(predictions),
                                            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
                                            'auc': auc,
                                            'tnr': tnr, 'gmean': gmean,
                                            'weighted_accuracy': weighted_accuracy}

        

    return results


# -------------------TOOL TO VISUAALIZE RESULTS-------------------------------------------------
def evaluate_method(results, method, metrics = ['precision', 'recall', 'fscore', 'tnr']):
    plt.style.use('seaborn-white')
    
    fig, ax = plt.subplots(nrows=1, ncols=5, sharey=True, figsize=(11, 5))
    for i, metric in enumerate(metrics):
        ax[i].axhline(results['ordinary'][metric], label='No Resampling')
        
        if results['class_weight']:
            ax[i].bar(0, results['class_weight'][metric], label='Adjust Class Weight')
            
        #ax[0].legend(loc='upper center', bbox_to_anchor=(9, 1.01),
        #             ncol=1, fancybox=True, shadow=True)
        
        for j, (technique, result) in enumerate(results[method].items()):
                ax[i].bar(j+1, result[metric], label=technique)

        ax[i].set_title(f'Subscribed: \n{metric}')
    
    # AUC vis
    ax[4].set_title(f'Area under curve')
    ax[4].axhline(results['ordinary']['auc'], label='No Resampling')
    if results['class_weight']:
        ax[4].bar(0, results['class_weight']['auc'], label='Adjust Class Weight')
    #techniques = []
    for j, (technique, result) in enumerate(results[method].items()):
        ax[4].bar(j+1, result[metric], label=technique)
        #techniques.append(technique)
    
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), #ncol=1, 
               fancybox=True, shadow=True, prop={'size': 12})
    
    
#-----------ADD RESULTS TO SUMMARY TABLE----------------------------------------------------
def add_results(df, model, results, technique, method=None):
    
    if method:
        recall = results[technique][method]['recall']
        precision = results[technique][method]['precision']
        tnr = results[technique][method]['tnr']
        auc = results[technique][method]['auc']
        f1 = results[technique][method]['fscore']
        df.loc[len(df)]=[method, model, recall, precision, tnr, auc, f1] 
    
    else:
        recall = results[technique]['recall']
        precision = results[technique]['precision']
        tnr = results[technique]['tnr']
        auc = results[technique]['auc']
        f1 = results[technique]['fscore']
        df.loc[len(df)]=[technique, model, recall, precision, tnr, auc, f1]   

