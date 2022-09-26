import json
import os

import yaml

from hooloovoo.utils.tree import Tree


def load_settings_file(*path: str):
    path = os.path.join(*path)
    with open(path) as fh:
        if path.lower().endswith(".json"):
            d = json.load(fh)
        elif path.lower().endswith(".yaml"):
            d = yaml.load(fh, yaml.Loader)
        else:
            _, ext = os.path.splitext(path)
            raise Exception("Cannot read settings of given file type: " + ext)
    return Tree(d)
