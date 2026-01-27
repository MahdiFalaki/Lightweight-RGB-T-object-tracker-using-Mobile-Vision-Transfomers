from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.mmMobileViT_Track.config import cfg, update_config_from_file


def parameters(yaml_name: str, epoch=None):
    params = TrackerParams()

    # update default config from yaml file
    prj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    yaml_file = os.path.join(prj_dir, 'experiments/mmMobileViT_Track/%s.yaml' % yaml_name)
    
    update_config_from_file(yaml_file)
    params.cfg = cfg
    print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    params.checkpoint = os.path.join(prj_dir, "models/mmMobileViT_Track_LasHeR_60ep/checkpoints/train/mmMobileViT_Track/%s/mmMobileViT_Track_ep%04d.pth.tar" %
                                     (yaml_name, cfg.TEST.EPOCH))

    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
