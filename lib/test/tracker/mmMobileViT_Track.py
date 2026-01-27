# /home/mark/Codes/mahdi_codes_folder/mmMobileViT_Track/lib/test/tracker/mmMobileViT_Track.py
import time
import os
import cv2
import torch

from lib.models.mmMobileViT_Track import build_mmMobileViT_Track
from lib.test.tracker.basetracker import BaseTracker
from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box


class mmMobileViT_Track(BaseTracker):
    def __init__(self, params):
        super(mmMobileViT_Track, self).__init__(params)
        network = build_mmMobileViT_Track(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location="cpu")["net"], strict=True)

        self.cfg = params.cfg
        self.device = "cpu" if self.cfg.TEST.DEVICE == "cpu" else "cuda"

        self.network = network.to(self.device)
        self.network.eval()

        self.preprocessor = Preprocessor(device=self.device)

        self.state = None
        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).to(self.device)

        if getattr(params, "debug", None) is None:
            setattr(params, "debug", 0)
        self.use_visdom = False
        self.debug = params.debug
        self.frame_id = 0
        self.save_all_boxes = params.save_all_boxes

        self.z_dict1 = None

        self.time_forward_s = 0.0
        self.time_init_s = 0.0
        self.time_track_s = 0.0
        self.calls = 0

    def reset_timing(self):
        self.time_forward_s = 0.0
        self.time_init_s = 0.0
        self.time_track_s = 0.0
        self.calls = 0

    def get_timing(self):
        return {
            "time_forward_s": self.time_forward_s,
            "time_init_s": self.time_init_s,
            "time_track_s": self.time_track_s,
            "calls": self.calls,
        }

    def initialize(self, image, info: dict):
        z_patch_arr, resize_factor, z_amask_arr = sample_target(
            image, info["init_bbox"], self.params.template_factor, output_sz=self.params.template_size
        )
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)

        z_rgb = template.tensors[:, :3, :, :]
        z_ir = template.tensors[:, 3:, :, :]

        with torch.inference_mode():
            t_fwd0 = time.perf_counter()

            z_v = self.network.backbone.conv_1.forward(z_rgb)
            z_i = self.network.backbone.conv_1.forward(z_ir)

            z_v = self.network.backbone.layer_1.forward(z_v)
            z_i = self.network.backbone.layer_1.forward(z_i)

            z_v = self.network.backbone.layer_2.forward(z_v)
            z_i = self.network.backbone.layer_2.forward(z_i)

            self.z_dict1 = [z_v, z_i]

            dt_init = time.perf_counter() - t_fwd0
            self.time_init_s += dt_init
            self.time_track_s += dt_init

        self.state = info["init_bbox"]
        self.frame_id = 0

        self.calls += 1

        if self.save_all_boxes:
            all_boxes_save = info["init_bbox"] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1

        x_patch_arr, resize_factor, x_amask_arr = sample_target(
            image, self.state, self.params.search_factor, output_sz=self.params.search_size
        )
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        x_rgb = search.tensors[:, :3, :, :]
        x_ir = search.tensors[:, 3:, :, :]

        with torch.inference_mode():
            t_fwd0 = time.perf_counter()

            out_dict = self.network.forward(
                template=self.z_dict1,
                search=[x_rgb, x_ir],
            )

            dt_fwd = time.perf_counter() - t_fwd0
            self.time_forward_s += dt_fwd
            self.time_track_s += dt_fwd

        self.calls += 1

        pred_score_map = out_dict["score_map"]
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict["size_map"], out_dict["offset_map"])
        pred_boxes = pred_boxes.view(-1, 4)
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image, info["gt_bbox"].tolist(), self.state), "Tracking", 1, "Tracking")
                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), "image", 1, "search_region")
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), "image", 1, "template")
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), "heatmap", 1, "score_map")
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), "heatmap", 1, "score_map_hann")

                if "removed_indexes_s" in out_dict and out_dict["removed_indexes_s"]:
                    removed_indexes_s = out_dict["removed_indexes_s"]
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), "image", 1, "masked_search")

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()
            return {"target_bbox": self.state, "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return mmMobileViT_Track
