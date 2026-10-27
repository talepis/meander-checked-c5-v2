import numba as nb

@nb.njit(cache=True, nogil=True)
def scatter_cols_serial(T,maps,V):
    nr=V.shape[0]; nc=V.shape[1]
    for i in range(nr):
        for j in range(nc):
            T[i,maps[j]] += V[i,j]

@nb.njit(cache=True, nogil=True)
def scatter_rows_serial(T,maps,V):
    nr=V.shape[0]; nc=V.shape[1]
    for j in range(nc):
        for i in range(nr):
            T[maps[i],j] += V[i,j]

@nb.njit(cache=True, nogil=True, parallel=True)
def scatter_cols_parallel(T,maps,V):
    nr=V.shape[0]; nc=V.shape[1]
    for i in nb.prange(nr):
        for j in range(nc):
            T[i,maps[j]] += V[i,j]

@nb.njit(cache=True, nogil=True, parallel=True)
def scatter_rows_parallel(T,maps,V):
    nr=V.shape[0]; nc=V.shape[1]
    for j in nb.prange(nc):
        for i in range(nr):
            T[maps[i],j] += V[i,j]
