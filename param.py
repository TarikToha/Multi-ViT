import numpy as np
from torchvision import transforms

object_sym = {
    'std_brick_object': 'br',
    'std_t_object': 'nt',
    'old_t_object': 'ot'
}

gen_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.ToTensor()
])

cls_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])


def get_prefix(capture_dir):
    prefix = ''
    if capture_dir in object_sym:
        prefix = f'{object_sym[capture_dir]}_'

    return prefix


def get_class_names(dataset_type):
    if dataset_type == 'clothes':
        class_names = np.array(
            ['fleece_jacket', 'leather_jacket', 'normal', 'snow_jacket']
        )

    elif dataset_type == 'environments':
        class_names = np.array(
            ['corridor', 'ladder', 'ladder_whiteboard', 'normal']
        )

    elif dataset_type == 'overall':
        class_names = np.array(
            ['left_ankle', 'left_chest', 'left_pocket', 'left_waist', 'no_weapon',
             'right_ankle', 'right_chest', 'right_pocket', 'right_waist']
        )

    elif dataset_type == 'intrusion':
        class_names = np.array(
            ['blank', 'long_center', 'long_left', 'long_right',
             'short_center', 'short_left', 'short_right']
        )

    elif dataset_type == 'wall':
        class_names = np.array(
            ['brick', 'brick_ladder', 'curtain',
             'furniture', 'furniture_ladder', 'gator', 'gator_ladder']
        )

    elif dataset_type == 'pose':
        class_names = np.array(
            ['blank', 'sitting', 'standing']
        )

    elif dataset_type == 'pose_loc2d':
        class_names = np.array(
            ['blank_blank',
             'sitting_long_center', 'sitting_long_left', 'sitting_long_right',
             'sitting_short_center', 'sitting_short_left', 'sitting_short_right',
             'standing_long_center', 'standing_long_left', 'standing_long_right',
             'standing_short_center', 'standing_short_left', 'standing_short_right']
        )

    elif dataset_type == 'fall':
        class_names = np.array(
            ['fast-fall', 'slow-fall']
        )

    else:
        print(f'{dataset_type}: class names are undefined')
        exit()

    return class_names


def generate_weapon_prompt(clothing_label: str, environment_label: str) -> str:
    clothing_descriptions = {
        "fleece_jacket": (
            "a fleece jacket",
            "creates slightly more diffuse radar reflections in the torso region, but has minimal impact on detection accuracy"
        ),
        "leather_jacket": (
            "a leather jacket",
            "introduces noticeable signal attenuation and scattering, potentially reducing detection reliability for concealed objects"
        ),
        "normal": (
            "normal clothing such as a shirt and pants",
            "produces clean, distinguishable radar reflections with minimal interference, forming the baseline for unarmed detection"
        ),
        "snow_jacket": (
            "a heavily padded snow jacket",
            "significantly attenuates and diffuses mmWave signals, often masking reflections from concealed objects"
        ),
    }

    environment_descriptions = {
        "corridor": (
            "walls introduce multipath reflections, causing ghosting artifacts in the torso and leg regions that may resemble concealed object signatures"
        ),
        "ladder": (
            "a ladder is present on the left side of the scene, introducing moderate environmental reflections that may interfere with clean spectrum interpretation"
        ),
        "ladder_whiteboard": (
            "both a ladder on the left and a whiteboard on the right produce multiple reflection paths, increasing the likelihood of clutter and potential false positives in the radar data"
        ),
        "normal": (
            "the environment is clean and minimally reflective, allowing for clear radar returns with minimal background interference"
        ),
    }

    PROMPT_TEMPLATE = (
        "A person is wearing {clothing_phrase}, which {clothing_effect}. "
        "The person is walking toward the radar in an environment where {environment_effect}. "
        "Generate the expected radar spectrum assuming the person is unarmed and no concealed objects are present."
    )

    clothing_phrase, clothing_effect = clothing_descriptions[clothing_label]
    environment_effect = environment_descriptions[environment_label]

    return PROMPT_TEMPLATE.format(
        clothing_phrase=clothing_phrase,
        clothing_effect=clothing_effect,
        environment_effect=environment_effect
    )
