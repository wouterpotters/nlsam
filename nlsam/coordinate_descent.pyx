#cython: wraparound=False, cdivision=True, boundscheck=False

# License: BSD 3 clause

# Stolen from sklearn.linearmodel.cd_fast.pyx
# Stripped out + heavily modifed version of
# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/linear_model/cd_fast.pyx

from itertools import product, repeat

from libc.math cimport fabs
cimport numpy as np
import numpy as np

cimport cython
# from libcpp cimport bint
from cython.parallel import prange

ctypedef np.float64_t DOUBLE
ctypedef np.uint32_t UINT32_t

np.import_array()

# The following two functions are shamelessly copied from the tree code.

# cdef enum:
#     # Max value for our rand_r replacement (near the bottom).
#     # We don't use RAND_MAX because it's different across platforms and
#     # particularly tiny on Windows/MSVC.
#     RAND_R_MAX = 0x7FFFFFFF


# cdef inline UINT32_t our_rand_r(UINT32_t* seed) nogil:
#     seed[0] ^= <UINT32_t>(seed[0] << 13)
#     seed[0] ^= <UINT32_t>(seed[0] >> 17)
#     seed[0] ^= <UINT32_t>(seed[0] << 5)

#     return seed[0] % (<UINT32_t>RAND_R_MAX + 1)


# cdef inline UINT32_t rand_int(UINT32_t end, UINT32_t* random_state) nogil:
#     """Generate a random integer in [0; end)."""
#     return our_rand_r(random_state) % end


cdef inline double fmax(double x, double y) nogil:
    if x > y:
        return x
    return y


cdef inline double fsign(double f) nogil:
    if f == 0:
        return 0
    elif f > 0:
        return 1.0
    else:
        return -1.0


cdef double abs_max(int n, double* a) nogil:
    """np.max(np.abs(a))"""
    cdef int i
    cdef double m = fabs(a[0])
    cdef double d
    for i in range(1, n):
        d = fabs(a[i])
        if d > m:
            m = d
    return m


cdef double max(int n, double* a) nogil:
    """np.max(a)"""
    cdef int i
    cdef double m = a[0]
    cdef double d
    for i in range(1, n):
        d = a[i]
        if d > m:
            m = d
    return m


# cdef double diff_abs_max(int n, double* a, double* b) nogil:
#     """np.max(np.abs(a - b))"""
#     cdef int i
#     cdef double m = fabs(a[0] - b[0])
#     cdef double d
#     for i in range(1, n):
#         d = fabs(a[i] - b[i])
#         if d > m:
#             m = d
#     return m


cdef extern from "cblas.h":
    enum CBLAS_ORDER:
        CblasRowMajor=101
        CblasColMajor=102
    enum CBLAS_TRANSPOSE:
        CblasNoTrans=111
        CblasTrans=112
        CblasConjTrans=113
        AtlasConj=114

    void daxpy "cblas_daxpy"(int N, double alpha, double *X, int incX,
                             double *Y, int incY) nogil
    double ddot "cblas_ddot"(int N, double *X, int incX, double *Y, int incY
                             ) nogil
    double dasum "cblas_dasum"(int N, double *X, int incX) nogil
    void dger "cblas_dger"(CBLAS_ORDER Order, int M, int N, double alpha,
                double *X, int incX, double *Y, int incY, double *A, int lda) nogil
    void dgemv "cblas_dgemv"(CBLAS_ORDER Order,
                      CBLAS_TRANSPOSE TransA, int M, int N,
                      double alpha, double *A, int lda,
                      double *X, int incX, double beta,
                      double *Y, int incY) nogil
    double dnrm2 "cblas_dnrm2"(int N, double *X, int incX) nogil
    void dcopy "cblas_dcopy"(int N, double *X, int incX, double *Y, int incY) nogil
    void dscal "cblas_dscal"(int N, double alpha, double *X, int incX) nogil


cdef double[:] enet_coordinate_descent_gram(double[:] w, double alpha, double beta,
                                 double[:, :] Q,
                                 double[:] q,
                                 double[:] y,
                                 int max_iter=500, double tol=1e-4, bint positive=1):
    """Cython version of the coordinate descent algorithm
        for Elastic-Net regression

        We minimize

        (1/2) * w^T Q w - q^T w + alpha norm(w, 1) + (beta/2) * norm(w, 2)^2

        which amount to the Elastic-Net problem when:
        Q = X^T X (Gram matrix)
        q = X^T y
    """

    # get the data information into easy vars
    cdef unsigned int n_samples = y.shape[0]
    cdef unsigned int n_features = Q.shape[0]

    # initial value "Q w" which will be kept of up to date in the iterations
    cdef double[:] H = np.dot(Q, w)

    cdef double[:] XtA = np.zeros(n_features)
    cdef double tmp
    cdef double w_ii
    cdef double d_w_max
    cdef double w_max
    cdef double d_w_ii
    cdef double gap = tol + 1.0
    cdef double d_w_tol = tol
    cdef double dual_norm_XtA
    cdef unsigned int ii
    cdef unsigned int n_iter = 0
    cdef unsigned int f_iter
    # cdef UINT32_t rand_r_state_seed = rng.randint(0, RAND_R_MAX)
    # cdef UINT32_t* rand_r_state = &rand_r_state_seed

    cdef double y_norm2 = np.dot(y, y)
    cdef double* w_ptr = <double*>&w[0]
    cdef double* Q_ptr = &Q[0, 0]
    cdef double* q_ptr = <double*>&q[0]
    cdef double* H_ptr = &H[0]
    cdef double* XtA_ptr = &XtA[0]

    tol = tol * y_norm2

    with nogil:
        for n_iter in range(max_iter):
            w_max = 0.0
            d_w_max = 0.0
            for ii in range(n_features):  # Loop over coordinates
            # for f_iter in range(n_features):  # Loop over coordinates
                # if random:
                #     ii = rand_int(n_features, rand_r_state)
                # else:
                #     ii = f_iter
                # ii = f_iter
                if Q[ii, ii] == 0.0:
                    continue

                w_ii = w[ii]  # Store previous value

                if w_ii != 0.0:
                    # H -= w_ii * Q[ii]
                    daxpy(n_features, -w_ii, Q_ptr + ii * n_features, 1,
                          H_ptr, 1)

                tmp = q[ii] - H[ii]

                if positive and tmp < 0:
                    w[ii] = 0.0
                else:
                    w[ii] = fsign(tmp) * fmax(fabs(tmp) - alpha, 0) \
                        / (Q[ii, ii] + beta)

                if w[ii] != 0.0:
                    # H +=  w[ii] * Q[ii] # Update H = X.T X w
                    daxpy(n_features, w[ii], Q_ptr + ii * n_features, 1,
                          H_ptr, 1)

                # update the maximum absolute coefficient update
                d_w_ii = fabs(w[ii] - w_ii)
                if d_w_ii > d_w_max:
                    d_w_max = d_w_ii

                if fabs(w[ii]) > w_max:
                    w_max = fabs(w[ii])

            if w_max == 0.0 or d_w_max / w_max < d_w_tol or n_iter == max_iter - 1:
                # the biggest coordinate update of this iteration was smaller than
                # the tolerance: check the duality gap as ultimate stopping
                # criterion

                # q_dot_w = np.dot(w, q)
                q_dot_w = ddot(n_features, w_ptr, 1, q_ptr, 1)

                for ii in range(n_features):
                    XtA[ii] = q[ii] - H[ii] - beta * w[ii]
                if positive:
                    dual_norm_XtA = max(n_features, XtA_ptr)
                else:
                    dual_norm_XtA = abs_max(n_features, XtA_ptr)

                # temp = np.sum(w * H)
                tmp = 0.0
                for ii in range(n_features):
                    tmp += w[ii] * H[ii]
                R_norm2 = y_norm2 + tmp - 2.0 * q_dot_w

                # w_norm2 = np.dot(w, w)
                w_norm2 = ddot(n_features, &w[0], 1, &w[0], 1)

                if (dual_norm_XtA > alpha):
                    const = alpha / dual_norm_XtA
                    A_norm2 = R_norm2 * (const ** 2)
                    gap = 0.5 * (R_norm2 + A_norm2)
                else:
                    const = 1.0
                    gap = R_norm2

                # The call to dasum is equivalent to the L1 norm of w
                gap += (alpha * dasum(n_features, &w[0], 1) -
                        const * y_norm2 +  const * q_dot_w +
                        0.5 * beta * (1 + const ** 2) * w_norm2)

                if gap < tol:
                    # return if we reached desired tolerance
                    break

    # return np.asarray(w)
        return w


def lasso_cd(double[:, :] X, double[:, :] D, bint positive=True, int n_lambdas=100, int max_iter=500, double tol=1e-4):
    cdef:
        int n = X.shape[1]
        int n_samples = X.shape[0]
        int i, j

        double eps=1e-3
        double l1_reg = 1.
        double l2_reg = 0.

        double[:, :] q = np.dot(D.T, X)
        double[:, :] Q = np.dot(D.T, D)
        double[:, :, :] coefs = np.zeros((D.shape[1], X.shape[1], n_lambdas), dtype=np.float64)
        double[:, :] prev_coef = np.zeros((D.shape[1], n_lambdas), dtype=np.float64)

        double lambda_max = np.max(np.abs(q)) / n_samples
        double[:] grid = np.logspace(np.log10(lambda_max * eps), np.log10(lambda_max), num=n_lambdas)[::-1]

        double lbda = 0.15
        double l1_ratio = 1.

    for j, i in product(range(n_lambdas), range(n)):
    # for i, j in product(range(n), [0]):
        # lbda = grid[j]
        # print(i,j,lbda,coefs.shape, coefs.ndim)
        l1_reg = lbda * l1_ratio * n_samples
        l2_reg = lbda * (1.0 - l1_ratio) * n_samples
        coefs[:, i, j] = enet_coordinate_descent_gram(prev_coef[:, j], l1_reg, l2_reg, Q, q[:, i], X[:, i], max_iter, tol, positive)
        prev_coef[:, j] = coefs[:, i, j]

    return np.asarray(coefs)

import _glmnet
# from _glmnet cimport elnet

def wrap_glmnet(double rho,
                double[:, ::1] X,
                double[:, ::1] y,
                double[:] weights,
                double[:] jd,
                double[:] vp,
                double[:, :] box_constraints,
                double flmin=0.001 ,
                double ulam=2,
                double thr=1.0e-4,
                int nlam=100,
                bint isd=0,
                int maxit=500):
    """
    Raw-output wrapper for elastic net linear regression.
    """
    cdef:
        # Flags determining overwrite behavior
        bint overwrite_pred_ok = False
        bint overwrite_targ_ok = False

        int nx = X.shape[1]
        int ny = y.shape[1]
        int memlimit = X.shape[1]
        int idx

        # int nlam=100
        # int isd=False
        # int maxit=500
        int jerr, nlp
        double lmu, rsq
        np.ndarray[double, ndim=1] a0_ = np.zeros(nlam, dtype=np.float64)
        # double [:,:] a0_ = np.zeros(nx, dtype=np.float64)
        np.ndarray[double, ndim=2] ca_ = np.zeros((nx, nlam), dtype=np.float64)
        np.ndarray[int, ndim=1] ia = np.zeros(nx, dtype=np.int32)
        np.ndarray[int, ndim=1] nin_ = np.zeros(nlam, dtype=np.int32)
        np.ndarray[double, ndim=1] alm_ = np.zeros(nlam, dtype=np.float64)
        # double[:] lmu = np.zeros(ny)
        # double[:] arr = np.zeros(ny)
        np.ndarray[double, ndim=2] intercepts = np.zeros((nlam, ny))
        np.ndarray[double, ndim=3] alphas = np.zeros((nx, nlam, ny))
        np.ndarray[double, ndim=2] lambdas = np.zeros((nlam, ny))

        np.ndarray[int, ndim=2] indices = np.zeros((nx, ny), dtype=np.int32)
        np.ndarray[int, ndim=2] df = np.zeros((nlam, ny), dtype=np.int32)
    # rsq = np.zeros((nlam, ny))


    # out=0

    # Call the Fortran wrapper.
    for idx in range(ny):
        # lmu, a0_, ca_, ia, nin_, rsq, alm_, nlp, jerr =  \
        out =  \
                _glmnet.elnet(rho, X, y[:, idx], weights, jd, vp, box_constraints,
                              memlimit, flmin, ulam, thr, nlam=nlam, isd=isd, maxit=maxit)

        # print(type(a0_), type(ca_), type(alm_))
        # print(a0_.shape, ca_.shape, alm_.shape)
        # print(a0_.shape, ca_.shape, nin_.shape, alm_.shape, indices.shape, ia.shape, X.shape, y.shape)
        # print(idx, indices.shape, ia.shape)
        intercepts[:,idx] = a0_
        df[:,idx] = nin_
        lambdas[:,idx] = alm_
        indices[:,idx] = ia - 1
        alphas[:,:,idx] = ca_[indices[:, idx]]


        # out =  \
        #         _glmnet.elnet(rho, X, y[:, idx], weights, jd, vp, box_constraints,
        #                       memlimit, flmin, ulam, thr, nlam=nlam, isd=isd, maxit=maxit)

    # return lmu, intercepts, alphas, ia, nin, rsq, lambdas, nlp, jerr
    return intercepts, alphas, indices, lambdas, df
