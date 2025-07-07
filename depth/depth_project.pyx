# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False

cimport numpy as np
import numpy as np

def depth_to_camera(np.ndarray[np.uint16_t, ndim=2] depth_frame,
                    float depth_scale,
                    dict depth_config,
                    dict color_config):
    cdef int d_h = depth_config['height'], d_w = depth_config['width']
    cdef float d_ppx = depth_config['ppx'], d_ppy = depth_config['ppy']
    cdef float d_fx = depth_config['fx'], d_fy = depth_config['fy']

    cdef int c_h = color_config['height'], c_w = color_config['width']
    cdef float c_ppx = color_config['ppx'], c_ppy = color_config['ppy']
    cdef float c_fx = color_config['fx'], c_fy = color_config['fy']

    cdef int depth_x, depth_y, depth_pixel_index
    cdef int color_x0, color_y0, color_x1, color_y1
    cdef int color_x, color_y, color_pixel_index

    cdef float depth
    cdef float x_d, y_d, X, Y, Z, x_c, y_c

    cdef np.uint16_t[:] depth_in = depth_frame.flatten()
    cdef np.uint16_t[:] depth_out = np.zeros((c_h * c_w), dtype=np.uint16)

    for depth_y in range(d_h):
        depth_pixel_index = depth_y * d_w
        for depth_x in range(d_w):
            depth = depth_scale * depth_in[depth_pixel_index]
            depth_pixel_index += 1
            if depth <= 0:
                continue

            # Map the top-left corner of the depth pixel onto the color image
            x_d = depth_x - 0.5
            y_d = depth_y - 0.5

            X = depth * (x_d - d_ppx) / d_fx
            Y = depth * (y_d - d_ppy) / d_fy
            Z = depth

            x_c = (X / Z) * c_fx + c_ppx
            y_c = (Y / Z) * c_fy + c_ppy

            color_x0 = int(x_c + 0.5)
            color_y0 = int(y_c + 0.5)

            if color_x0 < 0 or color_y0 < 0:
                continue

            # Map the bottom-right corner of the depth pixel onto the color image
            x_d = depth_x + 0.5
            y_d = depth_y + 0.5

            X = depth * (x_d - d_ppx) / d_fx
            Y = depth * (y_d - d_ppy) / d_fy
            Z = depth

            x_c = (X / Z) * c_fx + c_ppx
            y_c = (Y / Z) * c_fy + c_ppy

            color_x1 = int(x_c + 0.5)
            color_y1 = int(y_c + 0.5)

            if color_x1 >= c_w or color_y1 >= c_h:
                continue

            # Transfer between the depth pixels and the pixels inside the rectangle on the other image
            for color_y in range(color_y0, color_y1 + 1):
                for color_x in range(color_x0, color_x1 + 1):
                    color_pixel_index = color_y * c_w + color_x
                    if depth_out[color_pixel_index] == 0:
                        depth_out[color_pixel_index] = depth_in[depth_pixel_index]
                    else:
                        if depth_out[color_pixel_index] > depth_in[depth_pixel_index]:
                            depth_out[color_pixel_index] = depth_in[depth_pixel_index]

    return np.reshape(depth_out, (c_h, c_w))
