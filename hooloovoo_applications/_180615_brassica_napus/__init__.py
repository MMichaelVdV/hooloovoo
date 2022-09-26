from hooloovoo.utils.tree import Tree
from hooloovoo_applications._180615_brassica_napus.infer import Infer
from hooloovoo_applications._180615_brassica_napus.train import Train
from hooloovoo.deeplearning.networks.backbones.densenet import DenseNetBackbone, PublishedModel
from hooloovoo.deeplearning.networks.heads.itpnet import ItpNet
from hooloovoo.deeplearning.networks.network import Network
from hooloovoo.deeplearning.utils import attempt_get_cuda_device, Mode
from hooloovoo.utils.arbitrary import get_module_dir
from hooloovoo.utils.functional import default
from hooloovoo.utils.settings import load_settings_file


def run(settings: Tree):

    model = Network(
        DenseNetBackbone.published_structure(structure=PublishedModel.M121, pretrained=True),
        ItpNet()
    )

    settings.mode = Mode.from_string(settings.mode)
    settings.device = default(settings.device, attempt_get_cuda_device())
    print(settings)

    if settings.mode is Mode.TRAINING:
        Train(settings, model).run()
    if settings.mode is Mode.INFERENCE:
        Infer(settings, model).run()


if __name__ == "__main__":
    run(load_settings_file(get_module_dir("applications.brassica_napus_20172018"), "settings.yaml"))
