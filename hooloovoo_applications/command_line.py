import argparse
import importlib
import textwrap
from collections import OrderedDict

from hooloovoo.utils.settings import load_settings_file

_apps = OrderedDict([
    ("bnapus", "_180615_brassica_napus"),
])


def build_parser():
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            '''Hooloovoo applications'''
        ),
    )
    parser.add_argument("application", help="which application to run", choices=list(_apps.keys()))
    parser.add_argument("settings", help="a settings file in yaml format")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    module = importlib.import_module("hooloovoo_applications.{}".format(_apps[args.application]))
    settings = load_settings_file(args.settings)
    module.run(settings)


if __name__ == "__main__":
    main()
