# Visuals
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import numpy as np
import itertools
sns.set_style('white')


def stacked_bar(df, cat, groups, normalize=False, legend_label=[None], 
                title=None, xlabel=None, ylabel=None, legend_title=None):
    plt.style.use('seaborn-white')
    data = df.groupby(cat)[groups].value_counts(normalize=normalize).unstack(groups)
    
    data.plot(kind='barh', stacked=True, colormap=ListedColormap(sns.color_palette("GnBu", 2)), 
              figsize=(6,4), edgecolor = "black")
    
    ytick_labels = sorted(list(df[cat].unique()))
    legend_labels = legend_label
    
    plt.gca().invert_yaxis()
    plt.ylabel(ylabel)
    plt.yticks(np.arange(len(ytick_labels)), labels=ytick_labels, fontsize=12)
    plt.xlabel(xlabel, fontsize=12)
    plt.legend(legend_label, fontsize=12, loc='center left', 
               bbox_to_anchor=(1, 0.5), title=legend_title)
    plt.title(title, fontsize=16);
    
def corr_matrix(data):
    # Set the style of the visualization
    sns.set(style="white")

    # Create a covariance matrix
    corr = data.corr()

    # Generate a mask the size of our covariance matrix
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(14,11))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220,10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.8, center=0, square=True, linewidths=.4, annot=True, cbar_kws={'shrink':0.6})
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    #Add Normalization Option
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap, )
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def nested_bar(data, x, hue, title=None, legend_title=None, 
               legend_labels=None, order=None):
    sns.set(style="ticks", palette="pastel")
    fig = plt.figure(figsize=(14, 5))

    g = sns.countplot(x=x, hue=hue, palette=["m", "g"],
            data=data, order=order)

    sns.despine(offset=12, trim=True)
    plt.legend(legend_labels, title=legend_title)
    plt.title(title);