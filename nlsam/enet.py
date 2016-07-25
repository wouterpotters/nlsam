import numpy as np
from _glmnet import elnet, modval, uncomp, solns
import _glmnet
from scipy.sparse import csr_matrix
# class ElasticNet(object):
#     """ElasticNet based on GLMNET"""
#     def __init__(self, alpha, rho=0.2):
#         super(ElasticNet, self).__init__()
#         self.alpha = alpha
#         self.rho = rho
#         self.coef_ = None
#         self.rsquared_ = None

#     def fit(self, X, y):
#         n_lambdas, intercept_, coef_, _, _, rsquared_, lambdas, _, jerr \
#         = elastic_net(X, y, self.rho, lambdas=[self.alpha])
#         # elastic_net will fire exception instead
#         # assert jerr == 0
#         self.coef_ = coef_
#         self.intercept_ = intercept_
#         self.rsquared_ = rsquared_
#         return self

#     def predict(self, X):
#         return np.dot(X, self.coef_) + self.intercept_


def elastic_net_path(X, y, rho, **kwargs):
    """return full path for ElasticNet"""

    n_lambdas, intercepts, coefs_, indices, nin, _, lambdas, _, jerr \
        = elastic_net(X, y, rho, **kwargs)

    # Ordering from fortran starts at 1, so fix it to 0 for python
    # indices -= 1
    # print(n_lambdas)
    nlam = coefs_.shape[1]
    reordered_coefs = np.zeros((X.shape[1], nlam), dtype=np.float32)
    # predict = np.zeros((X.shape[0], nlam), dtype=np.float32)

    for i in range(nlam):
        nval = nin[i]
        ind = indices[:nval] - 1
        reordered_coefs[ind, i] = coefs_[:nval, i]

    predict = np.dot(X, reordered_coefs) + intercepts

    return predict, reordered_coefs


def select_best_path(X, y, beta, mu, variance, criterion='aic'):
    '''See https://arxiv.org/pdf/0712.0881.pdf p. 9 eq. 2.15 and 2.16'''

    # y.shape(y.shape[0])
    y = y.copy().ravel()
    n = y.shape[0]

    if criterion == 'aic':
        w = 2
    elif criterion == 'bic':
        w = np.log(n)
    else:
        raise ValueError('Criterion {} is not supported!'.format(criterion))

    # mu = np.empty((X.shape[0],) + intercepts.shape, dtype=np.float32)
    # print(mu.shape, X.shape, y.shape, beta.shape, intercepts.shape, variance.shape)
    # print(X.shape, beta.shape, intercepts.shape)
    # print(np.dot(X, beta[0]).shape, intercepts[0].shape, mu[0].shape, (np.dot(X, beta[0]) + intercepts[0]).shape)
    # for i in range(beta.shape[0]):
    #     # print(i)
    #     mu[:, i] = np.dot(X, beta[i]) + intercepts[i]

    # print(X.shape, y.shape, beta.shape, mu.shape, variance.shape)
    residuals = np.sum((y[..., None] - mu)**2, axis=0, dtype=np.float32)
    df_mu = np.sum(beta != 0, axis=0, dtype=np.int16)
    # print(mu.shape, df_mu.shape, beta.shape, variance[...,None].shape, residuals.shape, y.shape)
    criterion = (residuals / (n * variance)) + (w * df_mu / n)
    # print(criterion.shape, df_mu.shape, residuals.shape)
    # We don't want empty models
    criterion[df_mu == 0] = 1e300

    best_idx = np.argmin(criterion, axis=0)

    # print(mu.shape, best_idx.shape, beta.shape)
    # print(best_idx)
    # print(criterion.shape, df_mu.shape, mu.shape)
    # # print(criterion)
    # print(mu.shape, beta.shape)
    # print(mu[:, best_idx].shape, beta[:, best_idx].shape)
    # for i in range(100):
    #     print(mu[i])
    # print(best_idx, criterion[best_idx-1], criterion[best_idx])
    # print(criterion)
    return mu[:, best_idx], beta[:, best_idx]
    # best_betas = np.empty(beta.shape[:-1], dtype=np.float32)
    # best_mu = np.empty(y.shape, dtype=np.float32)
    # print(criterion.shape, best_idx.shape, best_betas.shape, beta.shape, best_mu.shape, mu.shape)
    # for i in range(criterion.shape[0]):
    #     # print(best_idx[i], )
    #     best_betas[:, i] = beta[:, i, best_idx[i]]
    #     best_mu[:, i] = mu[:, i, best_idx[i]]
    # # print(criterion[10])
    # return best_mu, best_betas

# def Lasso(alpha):
#     """Lasso based on GLMNET"""
#     return ElasticNet(alpha, rho=1.0)


def lasso_path(X, y, **kwargs):
    """return full path for Lasso"""

    # from glmnet import ElasticNet
    # enet = ElasticNet(alpha=1.)
    # enet.fit(X, y, normalize=False, include_intercept=False, **kwargs)

    # nlam = enet.n_lambdas
    # yhat = np.zeros((y.shape[0], nlam))
    # alpha = np.zeros((X.shape[1], nlam))

    # for i in range(nlam):
    #     yhat[:, i], alpha[:, i] = enet.predict(X).squeeze(), enet.get_coefficients_from_lambda_idx(i)
    # return yhat, alpha
    # # return enet.get_coefficients_from_lambda_idx(0)
    return elastic_net_path(X, y, rho=1.0, **kwargs)


# def __elastic_net(X, y, rho, **kwargs):

#     # out = [None] * y.shape[1]
#     # for i in range(y.shape[1]):
#     #     out[i] =
#     # print(len(out))

#     n_lambdas, intercepts, coefs_, indices, nin, _, lambdas, _, jerr = _elastic_net(X, y, rho, **kwargs)

#     return n_lambdas, intercepts[None, :], coefs_[None, :], indices[None, :], nin[None, :], _, lambdas[None, :], _, jerr


# def _elastic_net(predictors, target, balance, memlimit=None,
#                  largest=None, **kwargs):
#     """
#     Raw-output wrapper for elastic net linear regression.
#     """
#     _DEFAULT_THRESH = 1.0e-4
#     _DEFAULT_FLMIN = 0.001
#     _DEFAULT_NLAM = 100
#     # Mandatory parameters
#     predictors = np.asanyarray(predictors)
#     target = np.asanyarray(target)

#     # Decide on largest allowable models for memory/convergence.
#     memlimit = predictors.shape[1] if memlimit is None else memlimit

#     # If largest isn't specified use memlimit.
#     largest = memlimit if largest is None else largest

#     if memlimit < largest:
#         raise ValueError('Need largest <= memlimit')

#     # Flags determining overwrite behavior
#     overwrite_pred_ok = False
#     overwrite_targ_ok = False

#     thr = _DEFAULT_THRESH   # Minimum change in largest coefficient
#     weights = None          # Relative weighting per observation case
#     vp = None               # Relative penalties per predictor (0 = no penalty)
#     isd = True              # Standardize input variables before proceeding?
#     jd = np.zeros(1)        # Predictors to exclude altogether from fitting
#     ulam = None             # User-specified lambda values
#     flmin = _DEFAULT_FLMIN  # Fraction of largest lambda at which to stop
#     nlam = _DEFAULT_NLAM    # The (maximum) number of lambdas to try.

#     for keyword in kwargs:
#         if keyword == 'overwrite_pred_ok':
#             overwrite_pred_ok = kwargs[keyword]
#         elif keyword == 'overwrite_targ_ok':
#             overwrite_targ_ok = kwargs[keyword]
#         elif keyword == 'threshold':
#             thr = kwargs[keyword]
#         elif keyword == 'weights':
#             weights = np.asarray(kwargs[keyword]).copy()
#         elif keyword == 'penalties':
#             vp = kwargs[keyword].copy()
#         elif keyword == 'standardize':
#             isd = bool(kwargs[keyword])
#         elif keyword == 'exclude':
#             # Add one since Fortran indices start at 1
#             exclude = (np.asarray(kwargs[keyword]) + 1).tolist()
#             jd = np.array([len(exclude)] + exclude)
#         elif keyword == 'lambdas':
#             if 'flmin' in kwargs:
#                 raise ValueError("Can't specify both lambdas & flmin keywords")
#             ulam = np.asarray(kwargs[keyword])
#             flmin = 2. # Pass flmin > 1.0 indicating to use the user-supplied.
#             nlam = len(ulam)
#         elif keyword == 'flmin':
#             flmin = kwargs[keyword]
#             ulam = None
#         elif keyword == 'nlam':
#             if 'lambdas' in kwargs:
#                 raise ValueError("Can't specify both lambdas & nlam keywords")
#             nlam = kwargs[keyword]
#         else:
#             raise ValueError("Unknown keyword argument '%s'" % keyword)

#     box_constraints = np.zeros((2, predictors.shape[1]), order='F')
#     box_constraints[1] = 1e300
#     # box_constraints[0] = -1e300

#     # If predictors is a Fortran contiguous array, it will be overwritten.
#     # Decide whether we want this. If it's not Fortran contiguous it will
#     # be copied into that form anyway so there's no chance of overwriting.
#     if np.isfortran(predictors):
#         if not overwrite_pred_ok:
#             # Might as well make it F-ordered to avoid ANOTHER copy.
#             predictors = predictors.copy(order='F')

#     # target being a 1-dimensional array will usually be overwritten
#     # with the standardized version unless we take steps to copy it.
#     if not overwrite_targ_ok:
#         target = target.copy()

#     # Uniform weighting if no weights are specified.
#     if weights is None:
#         weights = np.ones(predictors.shape[0])

#     # Uniform penalties if none were specified.
#     if vp is None:
#         vp = np.ones(predictors.shape[1])

#     # Call the Fortran wrapper.
#     lmu, a0, ca, ia, nin, rsq, alm, nlp, jerr =  \
#             _glmnet.elnet(balance, predictors, target, weights, jd, vp, box_constraints,
#                           memlimit, flmin, ulam, thr, nlam=nlam)

#     # Check for errors, documented in glmnet.f.
#     if jerr != 0:
#         if jerr == 10000:
#             raise ValueError('cannot have max(vp) < 0.0')
#         elif jerr == 7777:
#             raise ValueError('all used predictors have 0 variance')
#         elif jerr < 7777:
#             raise MemoryError('elnet() returned error code %d' % jerr)
#         else:
#             raise Exception('unknown error: %d' % jerr)

#     return lmu, a0, ca, ia, nin, rsq, alm, nlp, jerr


def elastic_net(X, y, rho, pos=True, thr=1.0e-4, weights=None, vp=None,
                isd=False, nlam=100, maxit=1000, intr=False, **kwargs):
    """
    Raw-output wrapper for elastic net linear regression.
    """
    # print(thr)
    # X/y is overwritten in the fortran function at every loop, so we must copy it each time
    X = np.array(X, copy=True, dtype=np.float64, order='F')
    y = np.array(y.ravel(), copy=True, dtype=np.float64, order='F')

    # if y.ndim != 2:
    #     y.shape = (y.shape + (1,))
    # print(X.shape)
    # nx = X.shape[1]

    # # Flags determining overwrite behavior
    # overwrite_pred_ok = False
    # overwrite_targ_ok = False

    # thr = 1.0e-4   # Minimum change in largest coefficient
    # weights = None          # Relative weighting per observation case
    # vp = None               # Relative penalties per predictor (0 = no penalty)
    # isd = True              # Standardize input variables before proceeding?
    jd = np.zeros(1)        # X to exclude altogether from fitting
    ulam = None             # User-specified lambda values
    flmin = 0.001  # Fraction of largest lambda at which to stop
    # nlam = 100    # The (maximum) number of lambdas to try.
    # maxit = 1000

    box_constraints = np.zeros((2, X.shape[1]), order='F')
    box_constraints[1] = 1e300

    if not pos:
        box_constraints[0] = -1e300

    for keyword in kwargs:
        # if keyword == 'overwrite_pred_ok':
        #     overwrite_pred_ok = kwargs[keyword]
        # elif keyword == 'overwrite_targ_ok':
        #     overwrite_targ_ok = kwargs[keyword]
        # if keyword == 'threshold':
        #     thr = kwargs[keyword]
        # elif keyword == 'weights':
        #     weights = np.asarray(kwargs[keyword]).copy()
        # elif keyword == 'penalties':
        #     vp = kwargs[keyword].copy()
        # elif keyword == 'standardize':
        #     isd = bool(kwargs[keyword])
        if keyword == 'exclude':
            # Add one since Fortran indices start at 1
            exclude = (np.asarray(kwargs[keyword]) + 1).tolist()
            jd = np.array([len(exclude)] + exclude)
        elif keyword == 'lambdas':
            if 'flmin' in kwargs:
                raise ValueError("Can't specify both lambdas & flmin keywords")
            ulam = np.asarray(kwargs[keyword])
            flmin = 2.  # Pass flmin > 1.0 indicating to use the user-supplied.
            nlam = len(ulam)
        elif keyword == 'flmin':
            flmin = kwargs[keyword]
            ulam = None
        elif keyword == 'nlam':
            if 'lambdas' in kwargs:
                raise ValueError("Can't specify both lambdas & nlam keywords")
            nlam = kwargs[keyword]
        else:
            raise ValueError("Unknown keyword argument '%s'" % keyword)

    # # If X is a Fortran contiguous array, it will be overwritten.
    # # Decide whether we want this. If it's not Fortran contiguous it will
    # # be copied into that form anyway so there's no chance of overwriting.
    # if np.isfortran(X):
    #     if not overwrite_pred_ok:
    #         # Might as well make it F-ordered to avoid ANOTHER copy.
    #         X = X.copy(order='F')

    # y being a 1-dimensional array will usually be overwritten
    # with the standardized version unless we take steps to copy it.
    # if not overwrite_targ_ok:
    #     y = y.copy()

    # Uniform weighting if no weights are specified.
    if weights is None:
        weights = np.ones(X.shape[0], order='F')
    else:
        weights = np.asarray(weights).copy()
    # Uniform penalties if none were specified.
    if vp is None:
        vp = np.ones(X.shape[1], order='F')
    else:
        vp = vp.copy()

    # Call the Fortran wrapper.

    nx = X.shape[1]
    # ny = y.shape[1]

    # lmu = np.zeros(( dtype=np.int32)
    # a0 = np.zeros(nlam, dtype=np.float32)
    # ca = np.zeros((nx, nlam), dtype=np.float32)
    # # ca_ = np.zeros((nx, nlam), dtype=np.float32)
    # ia = np.zeros(nx, dtype=np.int32)
    # nin = np.zeros(nlam, dtype=np.int32)
    # rsq = np.zeros(nlam, dtype=np.float32)
    # alm = np.zeros(nlam, dtype=np.float32)
    # print(a0.shape, ca.shape, X.shape, y.shape)
    # for idx in range(y.shape[1]):

    lmu, a0, ca, ia, nin, rsq, alm, nlp, jerr = \
        elnet(rho, X, y, weights, jd, vp, box_constraints, nx, flmin, ulam, thr,
              nlam=nlam, isd=isd, maxit=maxit, intr=intr)

    return lmu, a0, ca, ia, nin, rsq, alm, nlp, jerr


# class GlmnetLinearModel(object):
#     """Class representing a linear model trained by Glmnet."""
#     def __init__(self, a0, ca, ia, nin, rsq, alm, npred):
#         self._intercept = a0
#         self._coefficients = ca[:nin]
#         self._indices = ia[:nin] - 1
#         self._rsquared = rsq
#         self._lambda = alm
#         self._npred = npred


#     def predict(self, X):
#         X = np.atleast_2d(np.asarray(X))
#         return self._intercept + \
#                 np.dot(X[:, self._indices], self._coefficients)

#     @property
#     def coefficients(self):
#         coeffs = np.zeros(self._npred)
#         coeffs[self._indices] = self._coefficients
#         return coeffs


# class GlmnetLinearResults(object):
#     def __init__(self, lmu, a0, ca, ia, nin, rsq, alm, nlp, npred, parm):
#         self._lmu = lmu
#         self._a0 = a0
#         self._ca = ca
#         self._ia = ia
#         self._nin = nin
#         self._rsq = rsq
#         self._alm = alm
#         self._nlp = nlp
#         self._npred = npred
#         self._model_objects = {}
#         self._parm = parm

#     def __len__(self):
#         return self._lmu

#     def __getitem__(self, item):
#         item = (item + self._lmu) if item < 0 else item
#         if item >= self._lmu or item < 0:
#             raise IndexError("model index out of bounds")

#         if item not in self._model_objects:
#             model =  GlmnetLinearModel(
#                         self._a0[item],
#                         self._ca[:,item],
#                         self._ia,
#                         self._nin[item],
#                         self._rsq[item],
#                         self._alm[item],
#                         self._npred
#                     )
#             self._model_objects[item] = model

#         else:
#             model = self._model_objects[item]

#         return model

#     @property
#     def nummodels(self):
#         return self._lmu

#     @property
#     def coefficients(self):
#         return self._ca[:np.max(self._nin), :self._lmu]

#     @property
#     def indices(self):
#         return self._ia

#     @property
#     def lambdas(self):
#         return self._alm[:self._lmu]

#     @property
#     def rho(self):
#         return self._parm


# This part stolen from
# https://github.com/ceholden/glmnet-python/blob/master/glmnet/utils.py


def IC_path(X, y, coefs, intercepts, criterion='aic'):
    """ Return AIC, BIC, or AICc for sets of estimated coefficients

    Args:
        X (np.ndarray): 2D (n_obs x n_features) design matrix
        y (np.ndarray): 1D dependent variable
        coefs (np.ndarray): 1 or 2D array of coefficients estimated from
            GLMNET using one or more ``lambdas`` (n_coef x n_lambdas)
        intercepts (np.ndarray): 1 or 2D array of intercepts from
            GLMNET using one or more ``lambdas`` (n_lambdas)
        criterion (str): AIC (Akaike Information Criterion), BIC (Bayesian
            Information Criterion), or AICc (Akaike Information Criterion
            corrected for finite sample sizes)

    Returns:
        np.ndarray: information criterion as 1D array (n_lambdas)

    Note: AIC and BIC calculations taken from scikit-learn's LarsCV

    """
    # coefs = np.atleast_2d(coefs)

    n_samples = y.shape[0]

    criterion = criterion.lower()
    if criterion == 'aic' or criterion == 'aicc':
        K = 2
    elif criterion == 'bic':
        K = np.log(n_samples)
    else:
        raise ValueError('Criterion must be either AIC, BIC, or AICc')

    residuals = y[:, np.newaxis] - (np.dot(X, coefs) + intercepts)
    mse = np.mean(residuals**2, axis=0)
    # df = np.zeros(coefs.shape[1], dtype=np.int16)
    df = np.sum(coefs != 0, axis=-1)

    # for k, coef in enumerate(coefs.T):
    #     mask = np.abs(coef) > np.finfo(coef.dtype).eps
    #     if not np.any(mask):
    #         continue
    #     df[k] = np.sum(mask)

    with np.errstate(divide='ignore'):
        criterion_ = n_samples * np.log(mse) + K * df
        if criterion == 'aicc':
            criterion_ += (2 * df * (df + 1)) / (n_samples - df - 1)

    return criterion_
