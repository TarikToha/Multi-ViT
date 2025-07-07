# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False

cimport numpy as np
import numpy as np

def project(np.ndarray[np.float32_t, ndim=2] point_cloud):
    cdef int i, x, z, N = point_cloud.shape[0]
    cdef np.ndarray[np.float32_t, ndim=3] rgbd_frame = np.ones((80, 80, 3), dtype=np.float32)
    cdef float r, g, b

    for i in range(N):
        x = <int> point_cloud[i, 0]
        z = <int> point_cloud[i, 2]

        if abs(x) < 40 and z < 80:
            x += 40
            r = point_cloud[i, 3] / 255.0
            g = point_cloud[i, 4] / 255.0
            b = point_cloud[i, 5] / 255.0
            rgbd_frame[z, x, 0] = r
            rgbd_frame[z, x, 1] = g
            rgbd_frame[z, x, 2] = b

    return rgbd_frame
