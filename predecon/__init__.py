import logging
import os

import numpy as np

from joblib import delayed, Memory, Parallel
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.neighbors import NearestNeighbors

def _neighborhood_variances(X, eps, n_jobs=None):
    """
    Calculate variance of each neighborhood for each point an attribute.

    References
    ----------
    Definition 1 (variance along an attribute) [Boehm,2004]
    """
    n = NearestNeighbors(radius=eps, metric='l2', n_jobs=n_jobs)
    n.fit(X)
    neighborhoods = n.radius_neighbors(return_distance=False)
    nh_size = np.array([nh.size for nh in neighborhoods])

    nh_variance_ = np.full(X.shape, np.inf)
    for p_idx in np.where(nh_size > 0)[0]:
        nh_variance_[p_idx] = np.square(X[neighborhoods[p_idx]] - X[p_idx]).sum(axis=0) / nh_size[p_idx]

    return nh_variance_

def _subspace_preferences(X, nh_var, var_thres, kappa):
    """
    Calculate subspace preferences.

    References
    ----------
    Definition 2 (subspace preference dimensionality) [Boehm,2004]
    """
    subs_pref = np.full(X.shape, 1.)
    subs_pref[nh_var <= var_thres] = kappa
    return subs_pref

def _weighted_pairwise_distances(X, subs_pref, subs_pref_params, n_jobs=None):
    """
    Calculate general preference weighted pairwise similarity between all points.

    References
    ----------
    Definition 4 (general preference weighted similarity) [Boehm,2004]
    """
    def _weighted_similarity(X, W, i, squared=False):
        """
        Calculate preference weighted similarities between point at index i and all other points.

        References
        ----------
        Definition 3 (preference weighted similarity measure) [Boehm,2004]
        """
        sim = np.sum(W * np.square(X[i] - X), axis=1) 
        return sim if squared else np.sqrt(sim)
    
    p = Parallel(
        n_jobs=n_jobs, backend='threading'
    )

    dist_p = np.array(p(
        delayed(_weighted_similarity)(X, subs_pref, i, True) \
            for i in np.arange(X.shape[0])
    ))

    # General preference weighted similarity (Section 3, Sefinition 4)
    return np.sqrt(np.maximum(dist_p, dist_p.T))

class PreDeCon(BaseEstimator, ClusterMixin, TransformerMixin):
    """
    PreDeCon - Subspace Preference Weighted Density Connected Points

    Parameters
    ----------
    eps : float
        (epsilon)

    min_samples : int
        (mu)

    max_dimensions : int
        (lambda)

    variance_thres : float
        (delta)

    kappa : int
        (kappa)

    n_jobs : int or None, optional (default=None)
        The number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    cachedir : str or None, optional (default=None)
        cachedir used by :obj:`joblib.Memory`.
        **Note**: Hyperparameters are used to identify repeated calls only. Thus, ``cachedir``
        cannot be used across different datasets.

    Attributes
    ----------
    core_samples_ : array, shape = [n_samples]
        Boolean array, indicating if corresponding sample is considered a core sample.

    core_sample_indices_ : array, shape = [n_core_samples]
        Indices of core samples.

    dist_p_ : array, shape = [n_samples, n_samples]
        General preference weighted pairwise similarities.

    labels_ : array, shape = [n_samples]
        Cluster labels for each point in the dataset given to fit().
        Noisy samples are given the label -1.

    nh_variance_ : array, shape = [n_samples, n_features]
        Variance of the neighborhood for each sample and feature.

    subs_pref_ : array, shape = [n_samples, n_features]
        Subspace preference vectors.

    weighted_neighborhoods_ : [n_samples]
        Indices for pref weighted neighborhood.


    References
    ----------
    Böhm, C. et al., "Density Connected Clustering with Local Subspace Preferences".
    In: Proceedings of the 4th IEEE Internation Conference on Data Mining (ICDM),
    Brighton, UK, 2004.
    """
    def __init__(self, eps=0.5, min_samples=5, max_dimensions=1, variance_thres=5, kappa=100,
                 n_jobs=None, cachedir=None):
        self.eps = eps
        self.min_samples = min_samples
        self.max_dimensions = max_dimensions
        self.variance_thres = variance_thres
        self.kappa = kappa

        self.n_jobs = n_jobs
        self.cachedir = cachedir
        self.memory = Memory(self.cachedir)

        self.log = logging.getLogger(__name__)

    def _neighborhood_variances(self, X):
        self.log.debug('Compute variances of neighborhood for each point')
        return self.memory.cache(_neighborhood_variances, ignore=['X', 'n_jobs'])(
            X.values, self.eps, n_jobs=self.n_jobs
        )

    def _subspace_preferences(self, X):
        self.log.debug('Compute subspace preference vectors based on variance')
        return self.memory.cache(_subspace_preferences, ignore=['X', 'nh_var'])(
            X.values, self.nh_variance_, self.variance_thres, self.kappa
        )
    
    def _weighted_pairwise_distances(self, X):
        self.log.debug('Compute pairwise distances with preference weighted similarities')
        return self.memory.cache(_weighted_pairwise_distances, ignore=['X', 'subs_pref', 'n_jobs'])(
            X.values, self.subs_pref_, (self.eps, self.kappa, self.variance_thres), n_jobs=self.n_jobs
        )

    def _validate_params(self):
        if not self.eps > 0:
            raise ValueError('eps must be positive.')
        if not self.min_samples > 0:
            raise ValueError('min_samples must be positive.')
        if not self.max_dimensions > 0:
            raise ValueError('max_dimensions must be positive.')
        if not self.variance_thres > 0:
            raise ValueError('variance_thres must be positive.')
        if not self.kappa > 0:
            raise ValueError('kappa must be positive.')

    def fit(self, X):
        self._validate_params()

        # Variance along an attribute (Section 3, Definition 1)
        self.nh_variance_ = self._neighborhood_variances(X)
        
        # General preference weighted similarity measure (Section 3, Definition 4)
        self.subs_pref_ = self._subspace_preferences(X)

        ### Compute pairwise preference weighted similarities
        self.dist_p_ = self._weighted_pairwise_distances(X)

        ### Compute preference weighted core points
        self.log.debug('Compute preference weighted core-points')
        # Preference weighted eps-neighborhood (Section 3, Definition 5)
        n = NearestNeighbors(radius=self.eps, metric='precomputed', n_jobs=self.n_jobs)
        n.fit(self.dist_p_)
        self.weighted_neighborhoods_ = n.radius_neighbors(return_distance=False)

        nh_size = np.array([nh.size for nh in self.weighted_neighborhoods_])
        p_dim = np.count_nonzero(self.nh_variance_ <= self.variance_thres, axis=1)

        # Preference weighted core point (Section 3, Definition 6)
        self.core_samples_ = (p_dim <= self.max_dimensions) & (nh_size >= self.min_samples)
        self.core_sample_indices_ = np.where(self.core_samples_)[0]

        stack = set()
        self.labels_ = np.full(X.shape[0], -1)
        cur_label = 0

        ### Perform DFS (similar to DBSCAN)
        self.log.debug('Begin sequential scan')
        for o in self.core_sample_indices_:
            if self.labels_[o] != -1:
                continue

            while True:
                if self.labels_[o] == -1:
                    self.labels_[o] = cur_label
                    if self.core_samples_[o]:
                        # Direct preference weighted reachability (Section 3, Defintion 7)
                        dir_reach = (self.labels_[self.weighted_neighborhoods_[o]] == -1) \
                                     & (p_dim[self.weighted_neighborhoods_[o]] <= self.max_dimensions)
                        stack.update(
                            self.weighted_neighborhoods_[o][dir_reach]
                        )
                try:
                    o = stack.pop()
                except KeyError:
                    break
            
            cur_label += 1