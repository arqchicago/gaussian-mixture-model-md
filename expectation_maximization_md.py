seed = 5954
import numpy as np
from scipy.stats import norm, multivariate_normal
import random
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from matplotlib import style
colors = ["red", "green", "blue", "yellow", "pink", "black", "orange", "purple", "beige", "brown", "gray", "cyan", "magenta"]
style.use('fivethirtyeight')

np.random.seed(seed)
random.seed(seed)


def gen_data(n_samples):
    # generating some random sample data
    # a mixture of ellipsoidal and appx circular shape

    # data 1
    data1 = np.random.randn(n_samples, 2) + np.array([8, 10])
    
    # data 2
    C = np.array([[0., 0.7], [1.5, 0.9]])
    data2 = np.dot(np.random.randn(n_samples, 2), C) + np.array([4, 5])

    # data 3
    C = np.array([[1., -0.3], [0.5, 1.4]])
    data3 = np.dot(np.random.randn(n_samples, 2) + np.array([0, 7]), C)

    # concatenate datasets into the final data set
    data = [data1, data2, data3]
    cols = ['r','b','g']
    X = np.vstack([data1, data2, data3])
    X = np.vstack(data)

    return X, data, cols


def plot_data(X, data, cols, means=None, covs=None, filename='output/data_plot_labels.png'):
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'gray', 'olive', 'cyan', 'black']
    cols = random.sample(colors, means.shape[0])
    plt.style.use('seaborn')
    fig, ax = plt.subplots()

    ax.scatter(X[:, 0], X[:, 1], 2.5) 
    fig.savefig('output/data_plot.png')

    fig2, ax2 = plt.subplots()    
    for i in range(len(data)):
        ax2.scatter(data[i][:, 0], data[i][:, 1], 1, facecolor=cols[i]) 
        
    if means is not None:
        for i in range(means.shape[0]):
            ax2.scatter(means[i][0], means[i][1], 50, facecolor='b', edgecolors='r', linewidths=2) 

    x = np.arange(-1.0, 11.0, 0.01)
    y = np.arange(1.0, 14.0, 0.01)
    X, Y = np.meshgrid(x, y)
    coords = np.array([X.ravel(), Y.ravel()]).T
    cols = random.sample(colors, means.shape[0])

    for i in range(means.shape[0]):
        mean, cov = means[i], covs[i]
        Z = multivariate_normal(mean, cov).pdf(coords)
        Z = Z.reshape(X.shape)
        plt.contour(X, Y, Z, colors = cols[i])
        
    ax.grid(True)
    fig2.savefig(filename)
   
    return 1


class GaussianMixtureModelmd():
    """
    A Gaussian mixture model for 2d data
    """

    def __init__(self, x, num_components):
        """
        Initialize a Gaussian mixture model.

        params:
        x = 1d numpy array data
        num_components = int
        """
        self.x = x
        self.shape = x.shape
        self.num_components = num_components
        self.means = np.array(x[np.random.choice(x.shape[0], num_components, replace=False)])      
        self.covs = np.full((num_components, x.shape[1], x.shape[1]), np.cov(x, rowvar = False))
        self.prior_mixture = np.ones(num_components)/num_components
        self.responsibilities = np.zeros([len(x), num_components])
        self._log_likelihood = 0
        self._log_lh_diff = 0

    def train(self, iters, tol):
        converged = False
        for j in range(iters):
            self.e_step()
            self.m_step()
            
            if(abs(self._log_lh_diff) <=tol):
                converged = True
                print(f'gmm converged after {j} iterations!')
                break
        
        if converged==False:
            print(f'gmm did not converge after {j} iterations.')
            print(f'consider more iterations or reducing the tolerance')
        
    
    def e_step(self):
        """
        expectation step: computing responsibilities of each data cluster 
        """
        log_likelihood_prev = self._log_likelihood
        for i in range(self.num_components):
            self.responsibilities[:,i] = np.multiply(self.prior_mixture[i], multivariate_normal(self.means[i], self.covs[i]).pdf(self.x))
            
        self._log_likelihood = np.sum(np.log(np.sum(self.responsibilities, axis = 1)))
        self._log_lh_diff = self._log_likelihood - log_likelihood_prev
        self.responsibilities = self.responsibilities/self.responsibilities.sum(axis=1)[:,None]
        
        
    def m_step(self):
        """
        maximization step: updating means and covariances
        """

        for i in range(self.num_components):
            self.prior_mixture[i] = np.divide(np.sum(self.responsibilities[:,i], axis=0), len(self.x))
            self.means[i] = np.divide(np.dot(self.responsibilities[:,i], self.x), np.sum(self.responsibilities[:,i], axis=0))
            weighted_sum = np.dot(self.responsibilities[:, i] * (self.x - self.means[i]).T, (self.x - self.means[i]))
            self.covs[i] = np.divide(weighted_sum, np.sum(self.responsibilities[:,i], axis=0))

    def get_means(self):
        return self.means
        
    def get_covs(self):
        return self.covs
        
    def get_responsibilities(self):
        return self.responsibilities
        
        
if __name__ == '__main__':

    # generate random data clusters
    n_samples = 500
    X, data, cols = gen_data(n_samples)

    # running the gmm
    gmm_md = GaussianMixtureModelmd(X, 3)
    gmm_md.train(iters=50, tol=1e-7)
    means = gmm_md.get_means()
    print(f'means\n {means}')
    covs = gmm_md.get_covs()
    print(f'covariances\n {covs}')
    plot_data(X, data, cols, means, covs, 'output/data_plot_labels.png')

    