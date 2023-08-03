import os
import json
import sys

# path = "/media/david/My Passport/out1/bop_data/needle_without_cap_filter_variation/train_pbr"
path = "/media/david/TOSHIBA EXT/needle_with_cap_filter_variation/train_pbr"
#path = '/home/david/render_tracebot/output'
sets = sorted(os.listdir(path))

needles = [8, 9, 10, 31, 32, 33, 34, 81, 82, 83, 84]

for set in sets:
    with open(os.path.join(path, set, 'scene_gt.json')) as f:
        data = json.load(f)
        for key, value in data.items():
            for val in value:
                if val['obj_id'] in needles:
                    val['obj_id'] = 3
    with open(os.path.join(path, set, 'scene_gt.json'), 'w') as out:
        json.dump(data, out, indent=2)

