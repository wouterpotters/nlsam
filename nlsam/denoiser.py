from __future__ import division, print_function

import numpy as np
import warnings
from time import time

from itertools import repeat, product, izip
# from functools import partial
from multiprocessing import Pool
from time import time

from dipy.core.ndindex import ndindex

from nlsam.utils import im2col_nd, col2im_nd, padding #sparse_dot_to_array
from scipy.sparse import lil_matrix, csc_matrix, issparse

#from glmnet import ElasticNet, CVGlmNet
# from sklearn.linear_model import lasso_path, LassoLarsIC, lars_path
from sklearn.linear_model import lasso_path as sk_lasso_path
#from nlsam.coordinate_descent import lasso_cd
from nlsam.enet import lasso_path, select_best_path

from sklearn.feature_extraction.image import extract_patches
from scipy.optimize import nnls
from skimage.util.shape import view_as_windows, view_as_blocks
# from sklearn.linear_model import Lasso

warnings.simplefilter("ignore", category=FutureWarning)

try:
    import spams
except ImportError:
    raise ValueError("Couldn't find spams library, did you properly install the package?")


def greedy_set_finder(sets):
    """Returns a list of subsets that spans the input sets with a greedy algorithm
    http://en.wikipedia.org/wiki/Set_cover_problem#Greedy_algorithm"""

    sets = [set(s) for s in sets]
    universe = set()

    for s in sets:
        universe = universe.union(s)

    output = []

    while len(universe) != 0:

        max_intersect = 0

        for i, s in enumerate(sets):

            n_intersect = len(s.intersection(universe))

            if n_intersect > max_intersect:
                max_intersect = n_intersect
                element = i

        output.append(tuple(sets[element]))
        universe = universe.difference(sets[element])

    return output


def processer(arglist):
    data, mask, variance, block_size, overlap, param_alpha, param_D, dtype, n_iter = arglist
    return _processer(data, mask, variance, block_size, overlap, param_alpha, param_D, dtype=dtype, n_iter=n_iter)


def _processer(data, mask, variance, block_size, overlap, param_alpha, param_D, dtype=np.float64, n_iter=10, gamma=3., tau=1.):
    # return data
    # mask=np.ones_like(mask)
    # data = data.astype(np.float64)
    orig_shape = data.shape
    extraction_step = np.array([1, 1, 1, block_size[-1]])

    no_overlap = False

    if no_overlap:
        overlap = (0, 0, 0, 0)
        extraction_step = list(block_size)[:-1] + [1]

    # print(mask.shape, block_size[:-1],  extraction_step[:-1])
    mask_array = extract_patches(mask, block_size[:-1], extraction_step[:-1]).reshape((-1, np.prod(block_size[:-1]))).T
    # mask_array = im2col_nd(mask, block_size[:3], overlap[:3])

    train_idx = np.sum(mask_array, axis=0) > mask_array.shape[0] / 2

    # mask_array = extract_patches(mask, patch_shape=block_size[:3], extraction_step=extraction_step[:3])
    # # mask_array = mask_array.reshape(mask_array.shape[0], -1).T
    # # print(np.sum(np.abs(mask_array - b)))
    # train_idx = np.sum(mask_array.reshape([-1] + list(block_size[:-1])), axis=(1, 2, 3)) > mask_array.shape[0] / 2

    # If mask is empty, return a bunch of zeros as blocks
    if not np.any(train_idx):
        return np.zeros_like(data)

    # X = im2col_nd(data, block_size, overlap)
    # a1 = time()
    D = param_alpha['D']

    X = extract_patches(data, patch_shape=block_size, extraction_step=extraction_step)
    X_out = np.zeros((D.shape[0], train_idx.shape[0]), dtype=np.float32)

    # var_mat = np.median(im2col_nd(variance[..., 0:orig_shape[-1]], block_size, overlap), axis=0).astype(np.float32)
    var_mat = None
    # X_full_shape = X.shape
    # X = X[:, train_idx].astype(np.float32)
    # var_mat = np.ones_like(var_mat)
    alpha = np.zeros((D.shape[1], X_out.shape[1]), dtype=np.float32)

    def mystandardize(D):
        S = np.std(D, axis=0, ddof=1)
        M = np.mean(D, axis=0)
        D_norm = (D - M) / S
        return D_norm, M, S

    # lambdas = np.logspace(np.log10(lambda_max * eps), np.log10(lambda_max), num=nlam)[::-1]
    nlam = 100  #len(lambdas)
    Xhat = np.zeros((D.shape[0], nlam), dtype=np.float64)
    alphas = np.zeros((D.shape[1], nlam), dtype=np.float64)
    best = np.zeros(X_out.shape[1])
    weights = np.ones(X_out.shape[0])

    # Assume b/b0 for weights
    # iidx = np.prod(block_size[:-1])

    for i, idx in enumerate(ndindex(X.shape[:X.ndim // 2])):

        if not train_idx[i]:
            continue
        # print(X.shape, X[idx].shape)
        # np.log(X[idx][..., 1:] / X[idx][..., 0:1])
        # weights[iidx:] = (-1/1000.) * np.log(X[idx][..., 1:] / X[idx][..., 0:1]).ravel()
        # adc = 800 * 1e-6
        # print((X[idx][..., 0:1] * np.exp(-1000 * adc)).ravel().shape, weights[iidx:].shape, weights.shape, np.prod(block_size[:-1]))
        # weights[iidx:] = (X[idx][..., 0:1] * np.exp(-1000 * adc)).ravel().repeat(block_size[-1] - 1)
        # print(X[idx][..., 0:1].shape, X.shape, X[idx].shape)
        # weights = 1/(X[idx] / X[idx][..., 0:1]).ravel()
        # weights = (X[idx] / X[idx][..., 0:1]).ravel()
        # iso = nnls(D, X[idx].ravel())[0]
        # isoc = np.dot(D, iso)
        # print(isoc.reshape(np.prod(block_size)).shape, isoc.shape, X[idx].shape)
        # X[idx] -= isoc.reshape(block_size)
        weigths = None
        # weights = (1. / X[idx]).ravel()#.repeat(block_size[-1])
        # weights = (1. / X[idx][..., 0:1]).ravel().repeat(block_size[-1])
        # weights[np.logical_not(np.isfinite(weights))] = 0

        Xhat[:], alphas[:] = lasso_path(D, X[idx], nlam=nlam, fit_intercept=True, pos=True, standardize=True, ols=False, weights=weights) #, lambdas=lambdas, maxit=10000)
        X_out[:, i], alpha[:, i], best[i] = select_best_path(D, X[idx], alphas, Xhat, var_mat, criterion='bic')
        # X_out[:, i] += isoc
    weigths = np.ones(X_out.shape[1], dtype=dtype, order='F')
    # weigths[train_idx] = 1. / ((alpha != 0).sum(axis=0) + 1.)

    if no_overlap:
        out = np.zeros_like(data)
        shaper = np.array(out.shape[:-1]) - np.array(block_size[:-1]) + 1
        shaper = shaper[:-1]

        # shaper = (range(0, (out.shape[0]//block_size[0] + 1) * block_size[0], block_size[0]),
        #           range(0, (out.shape[1]//block_size[1] + 1) * block_size[1], block_size[1]),
        #           range(0, (out.shape[2]//block_size[2] + 1) * block_size[2], block_size[2]))
        shaper = (range(0, out.shape[0], block_size[0]),
                  range(0, out.shape[1], block_size[1]),
                  range(0, out.shape[2], block_size[2]))
        idx = 0
        for i in shaper[0]:
            for j in shaper[1]:
                for k in shaper[2]:

                    out[i:i + block_size[0], j:j + block_size[1]] += X_out[:, idx].reshape(block_size)
                    idx += 1

        return out

    return col2im_nd(X_out, block_size, orig_shape, overlap, weigths)


def denoise(data, block_size, overlap, param_alpha, param_D, variance, n_iter=10,
            mask=None, dtype=np.float64):


    # print(padding(data, block_size, overlap).shape, data.shape)

    # a=time()
    # extraction_step = [1, 1 ,1, data.shape[-1]] #np.array([1, 1, 1, block_size[-1]])
    # X2 = extract_patches(data, patch_shape=block_size, extraction_step=extraction_step)
    # a1=time()
    # X2=X2.reshape([-1] + list(block_size))
    # # X3 = view_as_windows(data, window_shape=block_size, step=extraction_step).reshape([-1] + list(block_size))
    # # print(X2.shape, X3.shape, np.sum(np.abs(X2-X3)))
    # c=time()
    # X2 = X2.reshape(X2.shape[0], -1).T
    # b=time()
    # data2 = np.zeros((129,129,78,6))
    # data2[:128,:128] = data[:]
    # data=data2
    # no overlapping blocks for training
    # a=time()

    # no_over = (0, 0, 0, 0)
    # X = im2col_nd(data, block_size, no_over)
    # print(data.shape, block_size, X.shape, time()-a)

    # a=time()
    patch_shape = block_size
    extraction_step = list(block_size)[:-1] + [1]
    # extraction_step=1
    X = extract_patches(data, patch_shape, extraction_step).reshape((-1, np.prod(patch_shape))).T
    # X = X.reshape(X.shape, )
    # print(data.shape, block_size, X2.shape, time()-a, X2.flags)
    # print(np.sum(np.abs(X-X2)))

    # 1/0
    # t=time()
    # print(t-b, t-a, b-c,c-a1,a1-a)#, np.sum(np.abs(X-X2)))
    # # 1/0
    # X=X2
    # # Solving for D
    param_alpha['pos'] = True
    param_alpha['mode'] = 2
    param_alpha['lambda1'] = 1.2 / np.sqrt(np.prod(block_size))

    param_D['verbose'] = False
    param_D['posAlpha'] = True
    param_D['posD'] = True
    # param_D['mode'] = 5
    param_D['mode'] = 2
    param_D['lambda1'] = 1.2 / np.sqrt(np.prod(block_size))
    param_D['K'] = int(2 * np.prod(block_size))
    param_D['iter'] = 150
    param_D['batchsize'] = 500

    if 'D' in param_alpha:
        param_D['D'] = param_alpha['D']

    # a=time()
    # mask_col = im2col_nd(mask, block_size[:3], no_over[:3])
    # mask_col = im2col_nd(np.broadcast_to(mask[..., None], data.shape), block_size, no_over)
    mask_col = extract_patches(mask, patch_shape[:-1], extraction_step[:-1]).reshape((-1, np.prod(patch_shape[:-1]))).T
    train_idx = np.sum(mask_col, axis=0) > mask_col.shape[0]/2
    # print(train_idx.shape)
    # mask_array = extract_patches(mask, patch_shape=block_size, extraction_step=block_size)
    # # mask_array = mask_array.reshape(mask_array.shape[0], -1).T
    # # print(np.sum(np.abs(mask_array - b)))
    # train_idx = np.sum(mask_array.reshape([-1] + list(block_size[:2])), axis=(1, 2)) > mask_array.shape[0] / 2
    # print(train_idx.shape)
    # 1/0
    # print(train_idx.shape)
    train_data = X[:, train_idx]
    # print(train_idx.shape, train_data.shape)
    # print(np.sum(train_idx), np.sum( np.any(train_data != 0, axis=0)))
    # print(np.sum(train_data), np.sum(train_data[:, np.any(train_data != 0, axis=0)]))
    # 1/0
    # print(train_idx.shape, X.shape, train_data.shape)
    # print(np.sum(np.abs(train_data)))
    # 1/0
    train_data = np.asfortranarray(train_data[:, np.any(train_data != 0, axis=0)], dtype=np.float64)
    # print(np.sum(np.abs(train_data)))
    train_data /= np.sqrt(np.sum(train_data**2, axis=0, keepdims=True), dtype=np.float64)
    # print(train_idx.shape, X.shape, train_data.shape)
    # print(time()-a)
    # 1/0
    # a=time()
    param_alpha['D'] = spams.trainDL(train_data, **param_D)
    param_alpha['D'] /= np.sqrt(np.sum(param_alpha['D']**2, axis=0, keepdims=True, dtype=np.float64))
    param_D['D'] = param_alpha['D']

    del train_data, X
    # print(time() - a)
    # 1/0
    n_cores = param_alpha['numThreads']
    param_alpha['numThreads'] = 1
    param_D['numThreads'] = 1

    time_multi = time()
    pool = Pool(processes=n_cores)
    # a=time()

    overlap = True

    if overlap:
        ranger = range(data.shape[2] - block_size[2] + 1)
    else:
        ranger = range(0, data.shape[2], block_size[2])

    arglist = [(data[:, :, k:k+block_size[2]], mask[:, :, k:k+block_size[2]], variance[:, :, k:k+block_size[2]], block_size_subset, overlap_subset, param_alpha_subset, param_D_subset, dtype_subset, n_iter_subset)
               for k, block_size_subset, overlap_subset, param_alpha_subset, param_D_subset, dtype_subset, n_iter_subset
               in zip(ranger,
                      repeat(block_size),
                      repeat(overlap),
                      repeat(param_alpha),
                      repeat(param_D),
                      repeat(dtype),
                      repeat(n_iter))]
    # arglist = [(data, mask, variance, block_size, overlap, param_alpha, param_D, dtype, n_iter)]
    # print(time()-a, memory_usage_resource()-mem, mem)
    # 1/0
    data_denoised = pool.map(processer, arglist)
    pool.close()
    pool.join()

    param_alpha['numThreads'] = n_cores
    param_D['numThreads'] = n_cores

    print('Multiprocessing done in {0:.2f} mins.'.format((time() - time_multi) / 60.))

    # Put together the multiprocessed results
    data_subset = np.zeros_like(data)
    divider = np.zeros_like(data, dtype=np.int16)
    ones = np.ones_like(data_denoised[0], dtype=np.int16)
    # print(len(data_denoised), data_subset.shape, divider.shape, ones.shape)
    # for k in range(len(data_denoised)):
    for i, k in enumerate(ranger):
        data_subset[:, :, k:k+block_size[2]] += data_denoised[i]
        divider[:, :, k:k+block_size[2]] += ones

    data_subset /= divider
    return data_subset
