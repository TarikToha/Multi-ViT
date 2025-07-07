import cv2
import matplotlib.pyplot as plt
import numpy as np

# matplotlib.use('TkAgg')


class_names = np.array(
    ['left_ankle', 'left_chest', 'left_pocket', 'left_waist', 'no_weapon',
     'right_ankle', 'right_chest', 'right_pocket', 'right_waist']
)
class_prob = np.array(
    [0.00001911, 0.00020958, 0.00082624, 0.15692092, 0.99999964,
     0.00008842, 0.00001648, 0.02555639, 0.05330185]
)


def save_visual(class_prob, out_path, save=True):
    class_dict = {}
    for label, prob in zip(class_names, class_prob):
        class_dict[label] = prob

    human_map = np.array([
        [class_dict['left_chest'], class_dict['right_chest']],
        [class_dict['left_waist'], class_dict['right_waist']],
        [class_dict['left_pocket'], class_dict['right_pocket']],
        [class_dict['left_ankle'], class_dict['right_ankle']],
    ])

    # class_idx = np.argmax(class_prob, axis=-1)
    # label = class_names[class_idx]
    human_map = np.pad(human_map, ((1, 0), (1, 1)), mode='constant')
    human_map = cv2.resize(human_map, (300, 800), interpolation=cv2.INTER_LINEAR)
    human_map = (human_map * 255).astype(np.uint8)
    human_map = cv2.applyColorMap(human_map, cv2.COLORMAP_JET)
    human_map = cv2.GaussianBlur(human_map, (15, 15), 5)

    human_shape = cv2.imread('human_shape.jpg')
    human_shape = cv2.resize(human_shape, dsize=(300, 800))
    gray = cv2.cvtColor(human_shape, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contour, key=cv2.contourArea)

    mask = np.zeros_like(human_shape)
    cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
    human_shape = np.where(mask == (255, 255, 255), human_map, human_shape)
    human_shape = cv2.cvtColor(human_shape, cv2.COLOR_BGR2RGB)
    human_shape = human_shape / 255

    fig = plt.figure(figsize=(4, 8))
    ax = fig.add_subplot()
    ax.axis('off')
    im = ax.imshow(human_shape, vmin=0, vmax=1, cmap='jet',
                   interpolation='nearest', aspect='auto')
    plt.colorbar(im)

    fig.tight_layout()
    if save:
        plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    else:
        plt.show()
