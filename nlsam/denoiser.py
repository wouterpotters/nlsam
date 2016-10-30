from __future__ import division, print_function

import numpy as np
import warnings
import logging

from itertools import repeat, product, starmap
from time import time
from multiprocessing import Pool, get_context

from dipy.core.ndindex import ndindex

from nlsam.utils import im2col_nd, col2im_nd, padding
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
# from nlsam.utils import im2col_nd, col2im_nd
from nlsam.angular_tools import angular_neighbors

# from scipy.sparse import lil_matrix

warnings.simplefilter("ignore", category=FutureWarning)

try:
    import spams
except ImportError:
    raise ImportError("Couldn't find spams library, is the package correctly installed?")

logger = logging.getLogger('nlsam')


def nlsam_denoise(data, sigma, bvals, bvecs, block_size,
                  mask=None, is_symmetric=False, n_cores=None,
                  subsample=True, n_iter=10, b0_threshold=10, verbose=False):
    """Main nlsam denoising function which sets up everything nicely for the local
    block denoising.

    Input
    -----------
    data : ndarray
        Input volume to denoise.
    sigma : ndarray
        Noise standard deviation estimation at each voxel.
        Converted to variance internally.
    bvals : 1D array
        the N b-values associated to each of the N diffusion volume.
    bvecs : N x 3 2D array
        the N 3D vectors for each acquired diffusion gradients.
    block_size : tuple, length = data.ndim
        Patch size + number of angular neighbors to process at once as similar data.

    Optional parameters
    -------------------
    mask : ndarray, default None
        Restrict computations to voxels inside the mask to reduce runtime.
    is_symmetric : bool, default False
        If True, assumes that for each coordinate (x, y, z) in bvecs,
        (-x, -y, -z) was also acquired.
    n_cores : int, default None
        Number of processes to use for the denoising. Default is to use
        all available cores.
    subsample : bool, default True
        If True, find the smallest subset of indices required to process each
        dwi at least once.
    n_iter : int, default 10
        Maximum number of iterations for the reweighted l1 solver.
    b0_threshold : int, default 10
        A b-value below b0_threshold will be considered as a b0 image.

    Output
    -----------
    data_denoised : ndarray
        The denoised dataset
    """

    if verbose:
        logger.setLevel(logging.INFO)

    if mask is None:
        mask = np.ones(data.shape[:-1], dtype=np.bool)

    if data.shape[:-1] != mask.shape:
        raise ValueError('data shape is {}, but mask shape {} is different!'.format(data.shape, mask.shape))

    if data.shape[:-1] != sigma.shape:
        raise ValueError('data shape is {}, but sigma shape {} is different!'.format(data.shape, sigma.shape))

    if len(block_size) != len(data.shape):
        raise ValueError('Block shape {} and data shape {} are not of the same '
                         'length'.format(data.shape, block_size.shape))

    b0_loc = tuple(np.where(bvals <= b0_threshold)[0])
    num_b0s = len(b0_loc)
    variance = sigma**2
    orig_shape = data.shape

    logger.info("Found {} b0s at position {}".format(str(num_b0s), str(b0_loc)))

    # Average multiple b0s, and just use the average for the rest of the script
    # patching them in at the end
    if num_b0s > 1:
        mean_b0 = np.mean(data[..., b0_loc], axis=-1)
        dwis = tuple(np.where(bvals > b0_threshold)[0])
        data = data[..., dwis]
        bvals = np.take(bvals, dwis, axis=0)
        bvecs = np.take(bvecs, dwis, axis=0)

        rest_of_b0s = b0_loc[1:]
        b0_loc = b0_loc[0]

        data = np.insert(data, b0_loc, mean_b0, axis=-1)
        bvals = np.insert(bvals, b0_loc, [0.], axis=0)
        bvecs = np.insert(bvecs, b0_loc, [0., 0., 0.], axis=0)
        b0_loc = tuple([b0_loc])
        num_b0s = 1

    else:
        rest_of_b0s = None

    # Double bvecs to find neighbors with assumed symmetry if needed
    if is_symmetric:
        logger.info('Data is assumed to be already symmetrized.')
        sym_bvecs = np.delete(bvecs, b0_loc, axis=0)
    else:
        sym_bvecs = np.vstack((np.delete(bvecs, b0_loc, axis=0), np.delete(-bvecs, b0_loc, axis=0)))

    neighbors = (angular_neighbors(sym_bvecs, block_size[-1] - num_b0s) % (data.shape[-1] - num_b0s))[:data.shape[-1] - num_b0s]

    # Full overlap for dictionary learning
    overlap = np.array(block_size, dtype=np.int16) - 1
    b0 = np.squeeze(data[..., b0_loc])
    data = np.delete(data, b0_loc, axis=-1)

    indexes = []
    for i in range(len(neighbors)):
        indexes += [(i,) + tuple(neighbors[i])]

    if subsample:
        indexes = greedy_set_finder(indexes)

    b0_block_size = tuple(block_size[:-1]) + ((block_size[-1] + num_b0s,))

    denoised_shape = data.shape[:-1] + (data.shape[-1] + num_b0s,)
    data_denoised = np.zeros(denoised_shape, np.float32)

    # Put all idx + b0 in this array in each iteration
    to_denoise = np.empty(data.shape[:-1] + (block_size[-1] + 1,), dtype=np.float64)

    for i, idx in enumerate(indexes):
        dwi_idx = tuple(np.where(idx <= b0_loc, idx, np.array(idx) + num_b0s))
        logger.info('Now denoising volumes {} / block {} out of {}.'.format(idx, i + 1, len(indexes)))

        to_denoise[..., 0] = np.copy(b0)
        to_denoise[..., 1:] = data[..., idx]

        data_denoised[..., b0_loc + dwi_idx] += local_denoise(to_denoise,
                                                              b0_block_size,
                                                              overlap,
                                                              variance,
                                                              n_iter=n_iter,
                                                              mask=mask,
                                                              dtype=np.float64,
                                                              n_cores=n_cores,
                                                              verbose=verbose)

    divider = np.bincount(np.array(indexes, dtype=np.int16).ravel())
    divider = np.insert(divider, b0_loc, len(indexes))

    data_denoised = data_denoised[:orig_shape[0],
                                  :orig_shape[1],
                                  :orig_shape[2],
                                  :orig_shape[3]] / divider

    # Put back the original number of b0s
    if rest_of_b0s is not None:

        b0_denoised = np.squeeze(data_denoised[..., b0_loc])
        data_denoised_insert = np.empty(orig_shape, dtype=np.float32)
        n = 0

        for i in range(orig_shape[-1]):
            if i in rest_of_b0s:
                data_denoised_insert[..., i] = b0_denoised
                n += 1
            else:
                data_denoised_insert[..., i] = data_denoised[..., i - n]

        data_denoised = data_denoised_insert

    return data_denoised


def local_denoise(data, block_size, overlap, variance, n_iter=10, mask=None,
                  dtype=np.float64, n_cores=None, verbose=False):
    if verbose:
        logger.setLevel(logging.INFO)

    if mask is None:
        mask = np.ones(data.shape[:-1], dtype=np.bool)

    patch_shape = block_size
    extraction_step = list(block_size)[:-1] + [1]
    # extraction_step=1
    X = extract_patches(data, patch_shape, extraction_step).reshape((-1, np.prod(patch_shape))).T

    # Solving for D
    param_D = {}
    param_D['verbose'] = False
    param_D['posAlpha'] = True
    param_D['posD'] = True
    # param_D['mode'] = 5
    param_D['mode'] = 2
    param_D['lambda1'] = 1.2 / np.sqrt(np.prod(block_size))
    param_D['K'] = int(2 * np.prod(block_size))
    param_D['iter'] = 150
    param_D['batchsize'] = 500

    # if 'D' in param_alpha:
    #     print('D in alpha')
    #     param_D['D'] = param_alpha['D']

    mask_col = extract_patches(mask, patch_shape[:-1], extraction_step[:-1]).reshape((-1, np.prod(patch_shape[:-1]))).T
    train_idx = np.sum(mask_col, axis=0) > mask_col.shape[0] / 2

    train_data = X[:, train_idx]
    train_data = np.asfortranarray(train_data[:, np.any(train_data != 0, axis=0)], dtype=np.float64)
    train_data /= np.sqrt(np.sum(train_data**2, axis=0, keepdims=True), dtype=np.float64)

    D = spams.trainDL(train_data, **param_D)
    # D /= np.sqrt(np.sum(D**2, axis=0, keepdims=True, dtype=dtype))
    # param_D['D'] = param_alpha['D']

    del train_data, X

    # param_alpha['numThreads'] = 1
    # param_D['numThreads'] = 1

    time_multi = time()

    try:
        pool = get_context(method='forkserver').Pool(processes=n_cores)
    except ValueError:
        pool = Pool(processes=n_cores)

    # overlap = True
    ranger = range(data.shape[2] - block_size[2] + 1)
    # if overlap:
    #     ranger = range(data.shape[2] - block_size[2] + 1)
    # else:
    #     ranger = range(0, data.shape[2], block_size[2])

    arglist = [(data[:, :, k:k + block_size[2]],
                mask[:, :, k:k + block_size[2]],
                variance[:, :, k:k + block_size[2]],
                block_size_subset,
                overlap_subset,
                D_subset,
                dtype_subset,
                n_iter_subset)
               for k, block_size_subset, overlap_subset, D_subset, dtype_subset, n_iter_subset
               in zip(ranger,
                      repeat(block_size),
                      repeat(overlap),
                      repeat(D),
                      repeat(dtype),
                      repeat(n_iter))]

    data_denoised = pool.map(processer, arglist)
    pool.close()
    pool.join()

    # param_alpha['numThreads'] = n_cores
    # param_D['numThreads'] = n_cores

    logger.info('Multiprocessing done in {0:.2f} mins.'.format((time() - time_multi) / 60.))

    # Put together the multiprocessed results
    data_subset = np.zeros_like(data)
    divider = np.zeros_like(data, dtype=np.int16)
    ones = np.ones_like(data_denoised[0], dtype=np.int16)

    for i, k in enumerate(ranger):
        data_subset[:, :, k:k + block_size[2]] += data_denoised[i]
        divider[:, :, k:k + block_size[2]] += ones

    return data_subset / divider


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
    data, mask, variance, block_size, overlap, D, dtype, n_iter = arglist
    return _processer(data, mask, variance, block_size, overlap, D, dtype=dtype, n_iter=n_iter)


def _processer(data, mask, variance, block_size, overlap, D,
               dtype=np.float64, n_iter=10, gamma=3., tau=1., tolerance=1e-5):

    orig_shape = data.shape
    extraction_step = np.array([1, 1, 1, block_size[-1]])

    # # no_overlap = False
    # mask_array = im2col_nd(mask, block_size[:-1], overlap[:-1])
    # train_idx = np.sum(mask_array, axis=0) > (mask_array.shape[0] / 2.)

    # # If mask is empty, return a bunch of zeros as blocks
    # if not np.any(train_idx):
    #     return np.zeros_like(data)

    # X = im2col_nd(data, block_size, overlap)
    # var_mat = np.median(im2col_nd(variance, block_size[:-1], overlap[:-1])[:, train_idx], axis=0)
    # # X_full_shape = X.shape
    # X = X[:, train_idx]

    # if no_overlap:
    #     overlap = (0, 0, 0, 0)
    #     extraction_step = list(block_size)[:-1] + [1]

    # print(mask.shape, block_size[:-1],  extraction_step[:-1])
    mask_array = extract_patches(mask, block_size[:-1], extraction_step[:-1]).reshape((-1, np.prod(block_size[:-1]))).T
    # mask_array = im2col_nd(mask, block_size[:3], overlap[:3])
    train_idx = np.sum(mask_array, axis=0) > mask_array.shape[0] / 2

    # If mask is empty, return a bunch of zeros as blocks
    if not np.any(train_idx):
        return np.zeros_like(data)

    X = extract_patches(data, patch_shape=block_size, extraction_step=extraction_step)
    X_out = np.zeros((D.shape[0], train_idx.shape[0]), dtype=np.float32)
    # weigths = np.ones(X_out.shape[1], dtype=dtype, order='F')
    # print(X_out.shape, X.shape)
    # print(X.reshape(-1, np.prod(block_size)).T.shape)

    # return col2im_nd(X.reshape(-1, np.prod(block_size)).T, block_size, orig_shape, overlap, weigths)
    # var_mat = np.median(im2col_nd(variance[..., 0:orig_shape[-1]], block_size, overlap), axis=0).astype(np.float32)
    var_mat = None
    # X_full_shape = X.shape
    # X = X[:, train_idx].astype(np.float32)
    # var_mat = np.ones_like(var_mat)
    alpha = np.zeros((D.shape[1], X_out.shape[1]), dtype=np.float32)

    # def mystandardize(D):
    #     S = np.std(D, axis=0, ddof=1)
    #     M = np.mean(D, axis=0)
    #     D_norm = (D - M) / S
    #     return D_norm, M, S

    # lambdas = np.logspace(np.log10(lambda_max * eps), np.log10(lambda_max), num=nlam)[::-1]
    from glmnet import glmnet
    # from glmnetCoef import glmnetCoef
    # from glmnetPredict import glmnetPredict

    nlam = 100
    Xhat = np.zeros((D.shape[0], nlam), dtype=np.float64)
    alphas = np.zeros((D.shape[1], nlam), dtype=np.float64)
    best = np.zeros(X_out.shape[1])
    # weights = np.ones(X_out.shape[0])
    # print(X.shape, mask_array.shape, train_idx.shape, np.sum(train_idx), X.reshape((-1, np.prod(block_size))).T.shape, X_out.shape)
    # 1/0
    # Assume b/b0 for weights
    # iidx = np.prod(block_size[:-1])

    for i, idx in enumerate(ndindex(X.shape[:X.ndim // 2])):

        if not train_idx[i]:
            continue

        fit = glmnet(x=D.copy(), y=X[idx].flatten(), family='gaussian', alpha=1., nlambda=nlam)
        # print(np.sum(D.copy()), np.sum(X[idx].flatten()), 'sum1')
        predict = np.dot(D, fit['beta']) + fit['a0']
        # print(fit['a0'], 'a0glmnet')
        alphas[:, :fit['beta'].shape[1]] = fit['beta']
        alphas[:, fit['beta'].shape[1]:] = 0.

        Xhat[:, :predict.shape[1]] = predict
        Xhat[:, predict.shape[1]:] = 0.

        X_out[:, i], alpha[:, i], best[i] = select_best_path(D, X[idx], alphas, Xhat, var_mat, criterion='bic')

        # a, b = lasso_path(D, X[idx], nlam=nlam, fit_intercept=True, pos=True, standardize=True)
        # print(np.sum(D.copy()), np.sum(X[idx].flatten()), 'sum1')
        # print(np.sum(fit['beta']), np.sum(b), 'sums stuff')

        # print(np.sum(np.abs(a-Xhat)), np.sum(np.abs(b-alphas)))
        # print(a[0], a[0].shape, 'a0')
        # print(Xhat[0], Xhat[0].shape, 'xhat')
        # print(a.shape, b.shape, Xhat.shape, alphas.shape)
        # print(np.sum(b),np.sum(alphas),np.sum(fit['beta']), 'sums')
        # print(b[0], b[0].shape, 'b0')
        # print(alphas[0], alphas[0].shape, 'alphas')

        # 1/0
        # # weigths = None
        # Xhat[:], alphas[:] = lasso_path(D, X[idx], nlam=nlam, fit_intercept=True, pos=True, standardize=True)
        # X_out[:, i], alpha[:, i], best[i] = select_best_path(D, X[idx], alphas, Xhat, var_mat, criterion='bic')
        # # print(np.mean(X_out[:, i]), np.mean(X[idx]))
        # # print('mean residuals', np.mean((X_out[:, i] - X[idx].ravel())**2), best[i])
    weigths = np.ones(X_out.shape[1], dtype=dtype, order='F')

    # if no_overlap:
    #     out = np.zeros_like(data)
    #     shaper = np.array(out.shape[:-1]) - np.array(block_size[:-1]) + 1
    #     shaper = shaper[:-1]

    #     shaper = (range(0, out.shape[0], block_size[0]),
    #               range(0, out.shape[1], block_size[1]),
    #               range(0, out.shape[2], block_size[2]))
    #     idx = 0
    #     for i in shaper[0]:
    #         for j in shaper[1]:
    #             for k in shaper[2]:

    #                 out[i:i + block_size[0], j:j + block_size[1]] += X_out[:, idx].reshape(block_size)
    #                 idx += 1

    #     return out

    return col2im_nd(X_out, block_size, orig_shape, overlap, weigths)
