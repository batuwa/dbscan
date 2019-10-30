import os 

import matplotlib.pyplot as plt


def cluster_plot(data, labels, title='Cluster plot', circle=False, radius=0.25, savefig=False):
    """Plots the data points with their corresponding cluster and also identifies
    any outliers.
    
    Arguments:
        data {List[List]} -- Feature vector of the input data points.
        labels {List} -- Label with the cluster numberd for each of the data points.
    
    Keyword Arguments:
        title {str} -- The title of the plot. (default: {'Cluster plot'})
        circle {bool} -- Flag indicating whether or not to show the neighbourhood circle of each cluster. (default: {False})
        radius {float} -- If circle is True, the radius of the neighbourhood circle. (default: {0.25})
        savefig {bool} -- Flag indicating whether or not to save the plot in png format with title name. (default: {False})
    """
    fig, ax = plt.subplots(figsize=(15, 10))

    plt.scatter(data[:, 0], data[:,1], c=labels)
    
    if circle:
        circle = plt.Circle((data[0,0], data[0,1]),radius=radius, fill=None)
        ax.set_aspect(1)
        ax.add_artist(circle)
        
    for i in range(data.shape[0]):
        if labels[i] == -1:
            plt.text(data[i, 0]+0.05, data[i,1]-0.05, 'outlier', fontsize=8)
   
    plt.title(title)
    
    if savefig:
        plot_directory = '../plots'
        if not os.path.exists(plot_directory):
            os.makedirs(plot_directory)
        plt.savefig(plot_directory + title + '.png')
        
    plt.show()