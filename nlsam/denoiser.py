from __future__ import division, print_function

import numpy as np
import warnings
from time import time

from itertools import repeat
# from functools import partial
from multiprocessing import Pool

from nlsam.utils import im2col_nd, col2im_nd, sparse_dot_to_array
from scipy.sparse import lil_matrix, csc_matrix, issparse

from glmnet import ElasticNet, CVGlmNet
from sklearn.linear_model import lasso_path, LassoLarsIC, lars_path
from nlsam.coordinate_descent import enet_coordinate_descent_gram as lasso_cd

warnings.simplefilter("ignore", category=FutureWarning)

try:
    import spams
except ImportError:
    raise ValueError("Couldn't find spams library, did you properly install the package?")

# def universal_worker(input_pair):
#     """http://stackoverflow.com/a/24446525"""
#     function, args = input_pair
#     return function(*args)


# def pool_args(function, *args):
#     return izip(repeat(function), *izip(*args))



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


# def apply_weights(alpha, W):
#     cx = alpha.tocoo()
#     for i, j in izip(cx.row, cx.col):
#         cx[i, j] /= W[i, j]

#     return cx


def compute_weights(alpha_old, alpha, W, tau, eps):
    cx = alpha_old.tocoo()
    cy = alpha.tocoo()

    # Reset W values to eps
    idx = cx.nonzero()
    W[idx] = 1. / eps[idx[1]]

    # Assign new weights
    idx = cy.nonzero()
    W[idx] = 1. / ((cy.data**tau) + eps[idx[1]])

    return


def check_conv(alpha_old, alpha, eps=1e-5):
    x = alpha.tocoo()
    y = alpha_old.tocoo()

    # eps >= is for efficiency reason, and matrices are always 2D so we remove the useless dimension
    return (eps >= np.abs(x - y).max(axis=0)).toarray().squeeze()

# def _processer(arglist):
#     return processer(*arglist)
#
def processer(arglist):
    data, mask, variance, block_size, overlap, param_alpha, param_D, dtype, n_iter = arglist
    return _processer(data, mask, variance, block_size, overlap, param_alpha, param_D, dtype=dtype, n_iter=n_iter)

# def processer(arglist):
def _processer(data, mask, variance, block_size, overlap, param_alpha, param_D, dtype=np.float64, n_iter=10, gamma=3., tau=1.):
    # data, mask, variance, block_size, overlap, param_alpha, param_D, dtype, n_iter = arglist
    # gamma = 3.
    # tau = 1.

    orig_shape = data.shape
    mask_array = im2col_nd(mask, block_size[:3], overlap[:3])
    train_idx = np.sum(mask_array, axis=0) > mask_array.shape[0] / 2

    # If mask is empty, return a bunch of zeros as blocks
    if not np.any(train_idx):
        return np.zeros_like(data)

    X = im2col_nd(data, block_size, overlap)
    var_mat = np.median(im2col_nd(variance[..., 0:orig_shape[-1]], block_size, overlap)[:, train_idx], axis=0).astype(dtype)
    X_full_shape = X.shape
    X = X[:, train_idx]

    # param_alpha['L'] = int(0.5 * X.shape[0])
    # X_mean = X.mean(axis=1, keepdims=True)
    # X -= X_mean
    D = param_alpha['D']
    # del param_alpha['D']

    alpha = np.empty((D.shape[1], X.shape[1]))
    # alpha_prev = csc_matrix((D.shape[1], 1))
    # W = np.ones(alpha.shape, dtype=dtype, order='F')

    # DtD = np.dot(D.T, D)
    # DtX = np.dot(D.T, X)
    # DtXW = np.empty_like(DtX, order='F')

    # alpha_old = np.ones(alpha.shape, dtype=dtype)
    # has_converged = np.zeros(alpha.shape[1], dtype=np.bool)

    # xi = np.random.randn(X.shape[0], X.shape[1]) * var_mat
    # eps = np.max(np.abs(np.dot(D.T, xi)), axis=0)
    param_alpha['mode'] = 2
    param_alpha['pos'] = True
    # del param_alpha['pos']

    # for _ in range(n_iter):
    # not_converged = np.equal(has_converged, False)
    # DtXW[:, not_converged] = DtX[:, not_converged] / W[:, not_converged]

    # for i in range(alpha.shape[1]):
        # if not has_converged[i]:

    def obj_df(y, X, beta, variance, criterion='aic'):
        # print(y.shape, X.shape, beta.shape)
        # print(beta.sum())
        # Check if array is all zeros
        if issparse(beta):
            if beta.nnz == 0:
                return 1e300
        elif np.sum(beta) == 0:
            return 1e300

        n = y.shape[0]

        if criterion == 'aic':
            w = 2
        elif criterion == 'bic':
            w = np.log(n)
        else:
            raise ValueError('Criterion {} is not supported!'.format(criterion))

        # XtX_inv = np.linalg.inv(np.dot(X.T, X))
        # beta = XtX_inv.dot(np.dot(X.T, y) - 0.5*lbda * np.sign(beta))
        mu = np.empty(((y.shape[0],) + beta.shape[1:]), dtype=np.float32)
        # print(mu.shape)
        for i in range(beta.shape[-1]):
            mu[..., i] = np.dot(X, beta[..., i])
        # print(mu.shape)
        # mu = np.matmul(X, beta)
        # print(mu.shape)
        # mu = X.dot(beta)
        # df_mu = np.trace(X * np.linalg.inv(X.T, X) * X.T)
        square_res = np.sum((y[..., None] - mu)**2, axis=0, dtype=np.float32)
        # if issparse(beta):
        #     df_mu = beta.nnz
        # else:
        #     df_mu = np.sum(beta != 0).sum()
        df_mu = np.sum(beta != 0, axis=0, dtype=np.int16)
        # print(df_mu.shape, mu.shape, y.shape, square_res.shape, variance.shape)
        return (square_res / (variance[..., None] * n)) + (w * df_mu / n)

    def obj_func(X, D, alpha, lbda):
        # alpha = alpha.toarray()
        # print(X.shape, D.shape, alpha.shape, lbda, np.dot(D, alpha).shape, type(X - np.dot(D, alpha)))
        # print(type(X), type(D), type(alpha), type(lbda))
        # print(type(X - D.dot(alpha)))
        # np.asarray((X - np.dot(D, alpha)))**2
        # print(((X - np.dot(D, alpha))**2).shape)
        # np.sum((X - np.dot(D, alpha))**2)
        # lbda * alpha.abs().sum()
        residuals = np.sum((X - sparse_dot_to_array(D, alpha))**2).squeeze()
        # print(np.sum((X - np.dot(D, alpha))**2).shape, )
        return 0.5 * residuals + lbda * np.abs(alpha.data).sum()

    prev_obj = 1e300
    n_alphas = 10
    eps = 0.001
    custom_path = True
    sklearn_path = True
    admm = False

    if admm:

        from sporco.admm import bpdn

        alphas = np.empty((D.shape[1], X.shape[1], n_alphas), dtype=np.float32)

        lambda_max = np.abs(np.dot(D.T, X)).max(axis=0)
        lbda_max = lambda_max.max()
        lgrid = np.logspace(np.log10(lbda_max * eps + 1e-10), np.log10(lbda_max), num=n_alphas)

        for i, lbda in enumerate(lgrid):
            # print(D.shape, X.shape, alphas.shape, alpha.shape)
            solver = bpdn.ElasticNet(D, X, lbda, mu=0)
            alphas[..., i] = solver.solve()

        obj = obj_df(X, D, alphas, var_mat)
        best_idx = np.argmin(obj, axis=1)

        for i in range(obj.shape[0]):
            alpha[:, i] = alphas[:, i, best_idx[i]]

        X = np.dot(D, alpha)

    else:
        if sklearn_path:

            Q = np.asfortranarray(np.dot(D.T, D))
            q = np.asfortranarray(np.dot(D.T, X))

            max_path_length = 100
            path_length = min(D.shape[1], max_path_length) + 1
            alphas = np.zeros((D.shape[1], X.shape[1], path_length), dtype=np.float32)
            # npass = 1

            # alphas = np.empty((D.shape[1], X.shape[1], path_length), dtype=np.float64, order='F')
            # lambda_max = np.abs(q).max(axis=0)
            # lbda_max = np.median(lambda_max)  #.max()
            coef = None
            for idx in range(X.shape[1]):
                # print(alphas.shape, alphas[:, idx, :path_length].shape, D.shape, X[:, idx].shape, q[:, idx].shape, Q.shape)
                # stuff = lars_path(D, X[:, idx], Xy=q[:, idx], Gram=Q, positive=True, method='lasso')
                # print(stuff[0].shape, len(stuff[1]), stuff[2].shape)
                _, _, coef = lars_path(D, X[:, idx], Xy=q[:, idx], Gram=Q, copy_X=False, positive=True, max_iter=max_path_length, method='lasso')
                # _, coef, _ = lasso_path(D, X[:, idx], Xy=q[:, idx], Gram=Q, copy_X=False, positive=True, coef_init=coef) #, max_iter=max_path_length, method='lasso')
                # print(coef.shape)
                alphas[:, idx, :coef.shape[1]] = coef
                # print(coef.shape, idx, X.shape[1])

            obj = obj_df(X, D, alphas, var_mat, criterion='aic')
            best_idx = np.argmin(obj, axis=1)

            for i in range(obj.shape[0]):
                alpha[:, i] = alphas[:, i, best_idx[i]]

            X = np.dot(D, alpha)
        else:
        # print(lambda_max.min(), lambda_max.max())
            if custom_path:

                Q = np.asfortranarray(np.dot(D.T, D))
                q = np.asfortranarray(np.dot(D.T, X))
                # npass = 1

                alphas = np.empty((D.shape[1], X.shape[1], n_alphas), dtype=np.float64, order='F')
                lambda_max = np.abs(q).max(axis=0)
                lbda_max = np.median(lambda_max)  #.max()

                lgrid = np.logspace(np.log10(lbda_max * eps + 1e-10), np.log10(lbda_max), num=n_alphas)

                for i, lbda in enumerate(lgrid):
                    param_alpha['lambda1'] = lbda
                    spams.lasso(X, Q=Q, q=q, cholesky=True, **param_alpha).toarray(out=alphas[..., i])

                obj = obj_df(X, D, alphas, var_mat, criterion='aic')
                best_idx = np.argmin(obj, axis=1)

                for i in range(obj.shape[0]):
                    alpha[:, i] = alphas[:, i, best_idx[i]]

                X = np.dot(D, alpha)
                # for _ in range(npass):

                #     lgrid = np.logspace(np.log10(lbda_max * eps + 1e-10), np.log10(lbda_max), num=n_alphas)

                #     for i, lbda in enumerate(lgrid):
                #         # print(lbda, lambda_max)
                #         param_alpha['lambda1'] = lbda
                #         # alphas[..., i] = spams.lasso(X[:, idx:idx + 1], Q=Q, q=q[:, idx:idx + 1], **param_alpha).toarray().squeeze()
                #         # alphas[..., i] = spams.lasso(X, Q=Q, q=q, **param_alpha).toarray()#.squeeze()
                #         spams.lasso(X, Q=Q, q=q, **param_alpha).toarray(out=alphas[..., i])#.squeeze()
                #         # alpha_prev = alphas

                #     obj = obj_df(X, D, alphas, var_mat)
                #     # print(obj.shape)
                #     best_idx = np.argmin(obj, axis=1)

                #     lbda_max = lambda_max[max(best_idx)]

                # for i in range(obj.shape[0]):
                #     alpha[:, i] = alphas[:, i, best_idx[i]]

                # X = np.dot(D, alpha)
                # print(alpha.shape, alpha)

                # param_lassocd = {}
                # param_lassocd['beta'] = 0.
                # param_lassocd['max_iter'] = 500
                # param_lassocd['tol'] = 1e-4
                # param_lassocd['positive'] = True

                # Q = np.dot(D.T, D)
                # q = np.empty(D.shape[1])
                # # q = np.dot(D.T, X)
                # alpha_prev = np.zeros(D.shape[1])
                # alphas = np.empty((D.shape[1], X.shape[1], n_alphas), dtype=np.float64)

                # for idx in range(X.shape[1]):

                #     np.dot(D.T, X[:, idx], out=q)
                #     lambda_max = np.abs(q).max(axis=0)
                #     lbda_max = lambda_max.max()
                #     lgrid = np.logspace(np.log10(lbda_max * eps + 1e-10), np.log10(lbda_max), num=n_alphas)

                #     y = np.ascontiguousarray(X[:, idx])

                #     for i, lbda in enumerate(lgrid):
                #         # print(q.shape, np.dot(D.T, X[:, idx]).shape)

                #         # print(i, lbda, lbda_max)
                #         # print(q[:, idx].flags, y.flags, Q.flags)
                #         param_lassocd['alpha'] = lbda
                #         alpha_prev = lasso_cd(alpha_prev, Q=Q, q=q, y=y, **param_lassocd)
                #         alphas[:, idx, i] = alpha_prev

                # obj = obj_df(X, D, alphas, var_mat)
                # # print(obj, alphas.shape, alpha_prev.shape, X.shape, D.shape)
                # best_idx = np.argmin(obj, axis=1)

                # for i in range(obj.shape[0]):
                #     alpha[:, i] = alphas[:, i, best_idx[i]]

                # X = np.dot(D, alpha)

            else:

                # lasso = LassoLarsIC(positive=True, criterion='bic', normalize=False)
                # # print(D.shape, X.shape)
                # lasso.fit(D.T, X.T)
                # alpha = lasso.coef__
                # X = lasso.predict(D)

                alpha = np.empty((D.shape[1], X.shape[1]))
                n_lambdas = 10
                # mu = np.empty((X.shape[0], X.shape[1], n_alphas))
                # criterion = 'aic'
                # alpha_init = None
                constraints = np.ones((2, D.shape[1])) * 1e300  # Max value for alphas
                constraints[0] = 0  # Min value for alphas

                # n = X.shape[0]

                # if criterion == 'aic':
                #     w = 2
                # elif criterion == 'bic':
                #     w = np.log(n)
                # else:
                #     raise ValueError('Criterion {} is not supported!'.format(criterion))
                alphas = np.empty((D.shape[1], X.shape[1], n_alphas), dtype=np.float64)
                # lambda_max = np.abs(q).max(axis=0)
                # lbda_max = lambda_max.max()
                # lgrid = np.logspace(np.log10(lbda_max * eps + 1e-10), np.log10(lbda_max), num=n_alphas)

                for idx in range(X.shape[1]):

                    lasso = ElasticNet(alpha=1., standardize=False, n_lambdas=n_lambdas)
                    lasso.fit(D, X[:, idx], box_constraints=constraints)
                    # print(alphas.shape, lasso.out_lambdas.shape, D.shape, X[:, idx].shape)
                    alphas[:, idx] = lasso.out_lambdas
                # print(X.shape, X[:, idx].shape, D.shape, alphas.shape, var_mat.shape)
                obj = obj_df(X, D, alphas, var_mat)
                # print(obj.shape)
                best_idx = np.argmin(obj, axis=1)
                for i in range(obj.shape[0]):
                    alpha[:, i] = alphas[:, i, best_idx[i]]

                # print(X[:, idx].shape, lasso.predict(D)[:, best_idx].shape, lasso.predict(D).shape, obj.shape, best_idx.shape)
                # print(X.shape, D.shape, alphas.shape, var_mat.shape)
                # alpha[:, idx] = lasso.base_estimator.get_coefficients_from_lambda_idx(lasso.best_lambda_idx)
                for idx in range(X.shape[1]):
                    X[:, idx] = lasso.predict(D)[:, best_idx[idx]]



                    # X = np.dot(D, alpha)

                #     prev_obj = 1e300

                #     # lambda_grid = np.logspace(np.log10(lambda_grid * eps), np.log10(lambda_grid), num=n_lambdas)
                #     lambdas, alphas, _ = lasso_path(D, X[:, idx], positive=True, coef_init=alpha_init, normalize=False, n_alphas=n_alphas)
                #     mu[:, idx] = np.dot(D, alphas)
                #     alpha_init = alphas[:,0]
                #     # print(D.shape, X.shape, X[:, idx].shape, alpha_init.shape)

                # df_mu = np.sum(mu != 0, axis=0, keepdims=True)
                # obj = (np.sum((X - mu)**2) / (var_mat[:, None] * n)) + (w * df_mu / n)
                # # print(lambdas.shape, X.shape, alphas.shape, mu.shape, obj.shape, df_mu.shape, obj.shape)
                # # print(obj)
                # # print(np.max(mu), np.max(alphas))
                # best_idx = np.argmin(obj, axis=0)
                # # alpha_init = alphas[:, best_idx]
                # alpha[:, idx] = alphas[:, best_idx]
                    # print(lambdas[best_idx], np.sum(np.abs(alphas[:,best_idx])))
                    # X[:, idx] = np.dot(D, alpha[:, idx])

                    # lasso = LassoLarsIC(positive=True,
                    #                     criterion='aic',
                    #                     normalize=False,
                    #                     fit_intercept=False,
                    #                     max_iter=100)

                    # # # print(D.shape, X.shape)
                    # lasso.fit(D, X[:, idx])
                    # # # print(dir(lasso))
                    # # alpha_init = lasso.coef_
                    # alpha[:, idx] = lasso.coef_
                    # X[:, idx] = lasso.predict(D)
                    # print(D.flags)
                    # print(X.flags)
                    # print(X[:, idx].flags)

                    # lasso = ElasticNet(alpha=1., standardize=False, n_lambdas=100) #, normalize=False) #, include_intercept=False)
                    # lassocv = CVGlmNet(lasso, n_folds=3, n_jobs=1, verbose=False, include_full=True)
                    # lassocv.fit(D, X[:, idx], box_constraints=constraints)

                    # alpha[:, idx] = lassocv.base_estimator.get_coefficients_from_lambda_idx(lassocv.best_lambda_idx)
                    # X[:, idx] = lassocv.predict(D)

                    # lambdas, alphas, _ = lasso_path(D, X, positive=True)


                    # mu = lasso.predict(D)

                    # alphas = np.empty((D.shape[1], mu.shape[1]))

                    # for i in range(alphas.shape[1]):
                    #     # print(i, alphas.shape,  lasso.get_coefficients_from_lambda_idx(i).shape)
                    #     alphas[:, i] = lasso.get_coefficients_from_lambda_idx(i)

                    # df_mu = np.sum(alphas != 0, axis=0, keepdims=True)
                    # obj = (np.sum((X[:, idx:idx+1] - mu)**2) / (var_mat[idx, None] * n)) + (w * df_mu / n)

                    # # for i in range(mu.shape[1]):

                    # #     df_mu = np.sum(lasso.get_coefficients_from_lambda_idx(i) != 0)

                    # #     # print(X.shape, D.shape, mu.shape, lasso.get_coefficients_from_lambda_idx(1).shape, df_mu.shape)
                    # #     obj = (np.sum((X[:, idx] - mu[:, i])**2) / (var_mat[idx] * n)) + (w * df_mu / n)
                    # #     # print(obj.shape, var_mat.shape)

                    # #     if obj < prev_obj:
                    # #         best_idx = i
                    # #         prev_obj = obj

                    # best_idx = np.argmin(obj)
                    # alpha[:, idx] = lasso.get_coefficients_from_lambda_idx(best_idx)
                    # # print(alpha.shape)
                    # # X[:, idx] = np.dot(D, alpha)
                    # X[:, idx] = mu[:, best_idx]


                    # lbda =
                    # alpha_zero =

                    # for i in range(path.shape[-1]):

                    #     obj = obj_df(X[:, idx], D, path[:, i:i+1], alpha_zero, var_mat[idx])

                    #     if obj < prev_obj:
                    #         best_alpha = path[:, i:i+1]
                    #         prev_obj = obj

                    # alpha[:, idx:idx+1] = best_alpha

                # # param_alpha['lambda1'] = None
                # param_alpha['return_reg_path'] = True
                # # print(X.shape, lambda_max.shape)
                # # 1/0

                # best_alpha = np.zeros((D.shape[1], 1))
                # del DtX

                # for idx in range(X.shape[1]):

                #     param_alpha['lambda1'] = eps * lambda_max[idx]
                #     _, path = spams.lasso(X[:, idx:idx + 1], Q=Q, q=q[:, idx:idx + 1], **param_alpha)

                #     for i in range(path.shape[-1]):

                #         obj = obj_df(X[:, idx:idx + 1], D, path[:, i:i+1], param_alpha['lambda1'], var_mat[idx])
                #         # print(obj)
                #         if obj < prev_obj:
                #             best_alpha = path[:, i:i+1]
                #             prev_obj = obj

                #     alpha[:, idx:idx+1] = best_alpha

            # arr = alpha.toarray()
            # nonzero_ind = arr != 0
            # arr[nonzero_ind] /= W[nonzero_ind]
            # has_converged = np.max(np.abs(alpha_old - arr), axis=0) < 1e-5

            # if np.all(has_converged):
            #     break

            # alpha_old = arr
            # W[:] = 1. / (np.abs(alpha_old**tau) + eps)

                # compute_weights(alpha_old, alpha, W, tau, eps)

    # alpha = arr
    # X = D.dot(alpha)
    # X = sparse_dot(D,alpha)
    # param_alpha['D'] = D
    # X = np.dot(D, alpha) #+ X_mean
 ###   # X = sparse_dot_to_array(D, alpha) #+ X_mean
    # X = np.dot(D, alpha) #+ X_mean
    # print(type(X), X.shape, D.shape, alpha.shape)
    weigths = np.ones(X_full_shape[1], dtype=dtype, order='F')
    weigths[train_idx] = 1. / ((alpha != 0).sum(axis=0) + 1.)
    # print(type(X), X.shape)
    X2 = np.zeros(X_full_shape, dtype=dtype, order='F')
    # print(X2.shape, X.shape, D.shape, alpha.shape)
    X2[:, train_idx] = X

    return col2im_nd(X2, block_size, orig_shape, overlap, weigths)


def denoise(data, block_size, overlap, param_alpha, param_D, variance, n_iter=10,
            mask=None, dtype=np.float64):


    # no overlapping blocks for training
    no_over = (0, 0, 0, 0)
    X = im2col_nd(data, block_size, no_over)

    # Solving for D
    param_alpha['pos'] = True
    param_alpha['mode'] = 2
    param_alpha['lambda1'] = 1.2 / np.sqrt(np.prod(block_size))

    param_D['verbose'] = False
    param_D['posAlpha'] = True
    param_D['posD'] = True
    param_D['mode'] = 2
    param_D['lambda1'] = 1.2 / np.sqrt(np.prod(block_size))
    param_D['K'] = int(2*np.prod(block_size))
    param_D['iter'] = 150
    param_D['batchsize'] = 500

    if 'D' in param_alpha:
        param_D['D'] = param_alpha['D']

    # mask_col = im2col_nd(mask, block_size[:3], no_over[:3])
    mask_col = im2col_nd(np.broadcast_to(mask[..., None], data.shape), block_size, no_over)
    train_idx = np.sum(mask_col, axis=0) > mask_col.shape[0]/2

    train_data = X[:, train_idx]
    train_data = np.asfortranarray(train_data[:, np.any(train_data != 0, axis=0)], dtype=dtype)
    train_data /= np.sqrt(np.sum(train_data**2, axis=0, keepdims=True), dtype=dtype)
    param_alpha['D'] = spams.trainDL(train_data, **param_D)
    param_alpha['D'] /= np.sqrt(np.sum(param_alpha['D']**2, axis=0, keepdims=True, dtype=dtype))
    param_D['D'] = param_alpha['D']

    del train_data

    n_cores = param_alpha['numThreads']
    param_alpha['numThreads'] = 1
    param_D['numThreads'] = 1

    time_multi = time()
    pool = Pool(processes=n_cores)

    arglist = [(data[:, :, k:k+block_size[2]], mask[:, :, k:k+block_size[2]], variance[:, :, k:k+block_size[2]], block_size_subset, overlap_subset, param_alpha_subset, param_D_subset, dtype_subset, n_iter_subset)
               for k, block_size_subset, overlap_subset, param_alpha_subset, param_D_subset, dtype_subset, n_iter_subset
               in zip(range(data.shape[2] - block_size[2] + 1),
                      repeat(block_size),
                      repeat(overlap),
                      repeat(param_alpha),
                      repeat(param_D),
                      repeat(dtype),
                      repeat(n_iter))]
    # arglist = [(data, mask, variance, block_size, overlap, param_alpha, param_D, dtype, n_iter)]

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

    for k in range(len(data_denoised)):
        data_subset[:, :, k:k+block_size[2]] += data_denoised[k]
        divider[:, :, k:k+block_size[2]] += ones

    data_subset /= divider
    return data_subset
