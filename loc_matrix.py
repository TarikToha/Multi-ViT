import numpy as np

weapon_error_matrix = np.array([
    [0, 50, 50, 50, 50, 50, 50, 50, 50],
    [100, 0, 9, 12, 15, 24, 25.63, 48, 48.84],
    [100, 9, 0, 15, 12, 25.63, 24, 48.84, 48],
    [100, 12, 15, 0, 9, 12, 15, 36, 37.11],
    [100, 15, 12, 9, 0, 15, 12, 37.11, 36],
    [100, 24, 25.63, 12, 15, 0, 9, 24, 25.63],
    [100, 25.63, 24, 15, 12, 9, 0, 25.63, 24],
    [100, 48, 48.84, 36, 37.11, 24, 25.63, 0, 9],
    [100, 48.84, 48, 37.11, 36, 25.63, 24, 9, 0],
])

bin_labels = ['anomaly', 'clear']
weapon2idx = {
    'no_weapon': 0, 'left_chest': 1, 'right_chest': 2, 'left_waist': 3, 'right_waist': 4,
    'left_pocket': 5, 'right_pocket': 6, 'left_ankle': 7, 'right_ankle': 8,
}

object_sym = {
    'std_brick_object': 'br',
    'std_t_object': 'nt',
    'old_t_object': 'ot'
}

fall_error_matrix = np.array([
    [0, 50, 50, 50, 50, 50, 50],
    [100, 0, 3, 6, 3.0, 4.2, 6.7],
    [100, 3, 0, 3, 4.2, 3.0, 4.2],
    [100, 6, 3, 0, 6.7, 4.2, 3.0],
    [100, 3.0, 4.2, 6.7, 0, 3.0, 6],
    [100, 4.2, 3.0, 4.2, 3.0, 0, 3],
    [100, 6.7, 4.2, 3.0, 6, 3, 0],
])

pose2idx = {
    'standing': 2, 'sitting': 1, 'blank': 0
}
idx2pose = ['blank', 'sitting', 'standing']
fall_class_names = ['blank', 'fall', 'normal']

loc2idx = {
    'blank': 0, 'long_center': 1, 'long_left': 2, 'long_right': 3,
    'short_center': 4, 'short_left': 5, 'short_right': 6
}
idx2loc = [
    'blank', 'long_center', 'long_left', 'long_right',
    'short_center', 'short_left', 'short_right'
]
loc_class_names = idx2loc
