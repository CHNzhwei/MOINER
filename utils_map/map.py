#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 20:29:36 2019

@author: wanxiang.shen@u.nus.edu

main molmap code

"""

from utils_map.logtools import print_error
from utils_map.matrixopt import Scatter2Grid, Scatter2Array 
from utils_map.Extraction import Extraction
from utils_map import vismap
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE
from sklearn.utils import shuffle
from joblib import Parallel, delayed, load, dump
from umap import UMAP
from tqdm import tqdm
import pandas as pd
import numpy as np
import os



class Base:
    
    def __init__(self):
        pass
        
    def _save(self, filename):
        return dump(self, filename)
        
    def _load(self, filename):
        return load(filename)
 

class omicsMap(Base):
    
    def __init__(self, 
                 fmap_type, 
                 fmap_shape,
                 similarity_df,
                 split_channels,
                 omics_colormaps,
                 metric):
        super().__init__()
        assert fmap_type in ['scatter', 'grid'], 'no such feature map type supported!'

        self.ftype = "multi-omics"
        self.metric = metric
        self.method = None
        self.isfit = False

        self.dist_matrix = similarity_df
        self.fmap_type = fmap_type
        self.flist = list(self.dist_matrix.columns)
        
        if fmap_type == 'grid':
            S = Scatter2Grid()
        else:
            if fmap_shape == None:
                N = len(self.flist)
                l = np.int(np.sqrt(N))*2
                fmap_shape = (l, l)                
            S = Scatter2Array(fmap_shape)
        
        self._S = S
        self.split_channels = split_channels
        self.omics_colormaps = omics_colormaps
        self.extract = None
        
    def _fit_embedding(self, 
                        method = 'tsne',  
                        n_components = 2,
                        random_state = 1,  
                        verbose = 0,
                        n_neighbors = 30,
                        min_dist = 0.1,
                        **kwargs):
        
        """
        parameters
        -----------------
        method: {'tsne', 'umap', 'mds'}, algorithm to embedd high-D to 2D
        kwargs: the extra parameters for the conresponding algorithm
        """
        dist_matrix = self.dist_matrix

        if method == 'tsne':
            embedded = TSNE(n_components=n_components, 
                            random_state=random_state,
                            verbose = verbose,
                            **kwargs)
        elif method == 'umap':
            embedded = UMAP(n_components = n_components, 
                            n_neighbors = n_neighbors,
                            min_dist = min_dist,
                            verbose = verbose,
                            random_state=random_state, 
                            **kwargs)
        
        embedded = embedded.fit(dist_matrix)

        
        df = pd.DataFrame(embedded.embedding_, index = self.flist,columns=['x', 'y'])
        self.extract = Extraction(omics_colormaps = self.omics_colormaps) 
        typemap = self.extract.bitsinfo.set_index('IDs')

        df = pd.concat((df,typemap),axis=1)
        df['Channels'] = df['Subtypes']
        df.to_csv("./results_map/5.Feature_similarity_coordinate_info.csv")
        self.df_embedding = df
        self.embedded = embedded
        

    def fit(self, 
            method = 'umap', min_dist = 0.1, n_neighbors = 30,
            verbose = 0, random_state = 1, **kwargs): 

        if 'n_components' in kwargs.keys():
            kwargs.pop('n_components')
            
        ## embedding  into a 2d 
        assert method in ['tsne', 'umap'], 'no support such method!'
        
        self.method = method
        
        ## 2d embedding first
        self._fit_embedding(method = method,
                            n_neighbors = n_neighbors,
                            random_state = random_state,
                            min_dist = min_dist, 
                            verbose = verbose,
                            n_components = 2, **kwargs)

        
        if self.fmap_type == 'scatter':
            ## naive scatter algorithm
            print('   [MoInter] --Info: Applying naive scatter feature map...')
            self._S.fit(self.df_embedding, self.split_channels, channel_col = 'Channels')
            print('   [MoInter] --Info: Finished')
            
        else:
            ## linear assignment algorithm 
            print('   [MoInter] --Info: Applying grid feature map(assignment), this may take several minutes(1~30 min)')
            self._S.fit(self.df_embedding, self.split_channels, channel_col = 'Channels')
            print('   [MoInter] --Info: Finished')
        
        ## fit flag
        self.isfit = True
        self.fmap_shape = self._S.fmap_shape
    

    
    def transform(self, 
                  sigle_npy, 
                ):
    
    
        """
        parameters
        --------------------
        smiles:smiles string of compound
        scale: bool, if True, we will apply MinMax scaling by the precomputed values
        scale_method: {'minmax', 'standard'}
        """
        
        if not self.isfit:
            print_error('please fit first!')
            return

        arr = sigle_npy
        df = pd.DataFrame(arr).T
        df.columns = self.extract.bitsinfo.IDs
        vector_1d = df.values[0] #shape = (N, )
        fmap = self._S.transform(vector_1d)    
        return np.nan_to_num(fmap)
        

        
    def batch_transform(self, 
                        npy, 
                        n_jobs=4):
    

        
        P = Parallel(n_jobs=n_jobs)
        res = P(delayed(self.transform)(sigle_npy) for sigle_npy in tqdm(npy, ascii=True)) 
        X = np.stack(res) 
        X = X.transpose(0,3,1,2)

        return X

    
    def rearrangement(self, orignal_X, target_mp):

        """
        Re-Arragement feature maps X from orignal_mp's to target_mp's style, in case that feature already extracted but the position need to be refit and rearrangement.

        parameters
        -------------------
        orignal_X: the feature values transformed from orignal_mp(object self)
        target_mp: the target feature map object

        return
        -------------
        target_X, shape is (N, W, H, C)
        """
        assert self.flist == target_mp.flist, print_error('Input features list is different, can not re-arrangement, check your flist by mp.flist method' )
        assert len(orignal_X.shape) == 4, print_error('Input X has error shape, please reshape to (samples, w, h, channels)')
        
        idx = self._S.df.sort_values('indices').idx.tolist()
        idx = np.argsort(idx)

        N = len(orignal_X) #number of sample
        M = len(self.flist) # number of features
        res = []
        for i in tqdm(range(N), ascii=True):
            x = orignal_X[i].sum(axis=-1)
            vector_1d_ordered = x.reshape(-1,)
            vector_1d_ordered = vector_1d_ordered[:M]
            vector_1d = vector_1d_ordered[idx]
            fmap = target_mp._S.transform(vector_1d)
            res.append(fmap)
        return np.stack(res)

    
    
    def plot_scatter(self, htmlpath='./results_map', htmlname=None, radius = 3):
        """radius: the size of the scatter, must be int"""
        df_scatter, H_scatter = vismap.plot_scatter(self,  
                                htmlpath=htmlpath, 
                                htmlname=htmlname,
                                radius = radius)
        
        self.df_scatter = df_scatter
        return H_scatter   
        
        
    def plot_grid(self, htmlpath='./results_map', htmlname=None):
        
        if self.fmap_type != 'grid':
            return
        
        df_grid, H_grid = vismap.plot_grid(self,  
                                htmlpath=htmlpath, 
                                htmlname=htmlname)
        
        self.df_grid = df_grid
        return H_grid       
        
        
    def load(self, filename):
        return self._load(filename)
    
    
    def save(self, filename):
        return self._save(filename)
    