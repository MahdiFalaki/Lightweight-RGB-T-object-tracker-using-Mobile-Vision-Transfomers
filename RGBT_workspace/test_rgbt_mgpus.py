# /home/mark/Codes/mahdi_codes_folder/mmMobileViT_Track/RGBT_workspace/test_rgbt_mgpus.py
import os
import sys
from os.path import join, isdir, dirname
import numpy as np
import argparse
import cv2
import torch

prj = join(dirname(__file__), "..")
if prj not in sys.path:
    sys.path.append(prj)

from lib.test.tracker.mmMobileViT_Track import mmMobileViT_Track
import lib.test.parameter.mmMobileViT_Track as rgbt_prompt_params
from lib.train.dataset.depth_utils import get_x_frame


def genConfig(seq_path, set_type):
    if set_type == "RGBT234":
        RGB_img_list = sorted(
            [seq_path + "/visible/" + p for p in os.listdir(seq_path + "/visible") if os.path.splitext(p)[1] == ".jpg"]
        )
        T_img_list = sorted(
            [
                seq_path + "/infrared/" + p
                for p in os.listdir(seq_path + "/infrared")
                if os.path.splitext(p)[1] == ".jpg"
            ]
        )
        RGB_gt = np.loadtxt(seq_path + "/visible.txt", delimiter=",")
        T_gt = np.loadtxt(seq_path + "/infrared.txt", delimiter=",")

    elif set_type == "RGBT210":
        RGB_img_list = sorted(
            [seq_path + "/visible/" + p for p in os.listdir(seq_path + "/visible") if os.path.splitext(p)[1] == ".jpg"]
        )
        T_img_list = sorted(
            [
                seq_path + "/infrared/" + p
                for p in os.listdir(seq_path + "/infrared")
                if os.path.splitext(p)[1] == ".jpg"
            ]
        )
        RGB_gt = np.loadtxt(seq_path + "/init.txt", delimiter=",")
        T_gt = np.loadtxt(seq_path + "/init.txt", delimiter=",")

    elif set_type == "GTOT":
        RGB_img_list = sorted(
            [join(seq_path, "v", p) for p in os.listdir(join(seq_path, "v")) if os.path.splitext(p)[1].lower() in [".png", ".bmp"]]
        )
        T_img_list = sorted(
            [join(seq_path, "i", p) for p in os.listdir(join(seq_path, "i")) if os.path.splitext(p)[1].lower() in [".png", ".bmp"]]
        )
        RGB_gt = np.loadtxt(seq_path + "/groundTruth_v.txt", delimiter=" ")
        T_gt = np.loadtxt(seq_path + "/groundTruth_i.txt", delimiter=" ")

        x_min = np.min(RGB_gt[:, [0, 2]], axis=1)[:, None]
        y_min = np.min(RGB_gt[:, [1, 3]], axis=1)[:, None]
        x_max = np.max(RGB_gt[:, [0, 2]], axis=1)[:, None]
        y_max = np.max(RGB_gt[:, [1, 3]], axis=1)[:, None]
        RGB_gt = np.concatenate((x_min, y_min, x_max - x_min, y_max - y_min), axis=1)

        x_min = np.min(T_gt[:, [0, 2]], axis=1)[:, None]
        y_min = np.min(T_gt[:, [1, 3]], axis=1)[:, None]
        x_max = np.max(T_gt[:, [0, 2]], axis=1)[:, None]
        y_max = np.max(T_gt[:, [1, 3]], axis=1)[:, None]
        T_gt = np.concatenate((x_min, y_min, x_max - x_min, y_max - y_min), axis=1)

    elif set_type == "LasHeR":
        RGB_img_list = sorted([seq_path + "/visible/" + p for p in os.listdir(seq_path + "/visible") if p.endswith(".jpg")])
        T_img_list = sorted([seq_path + "/infrared/" + p for p in os.listdir(seq_path + "/infrared") if p.endswith(".jpg")])
        RGB_gt = np.loadtxt(seq_path + "/visible.txt", delimiter=",")
        T_gt = np.loadtxt(seq_path + "/infrared.txt", delimiter=",")

    elif "VTUAV" in set_type:
        RGB_img_list = sorted([seq_path + "/rgb/" + p for p in os.listdir(seq_path + "/rgb") if p.endswith(".jpg")])
        T_img_list = sorted([seq_path + "/ir/" + p for p in os.listdir(seq_path + "/ir") if p.endswith(".jpg")])
        RGB_gt = np.loadtxt(seq_path + "/rgb.txt", delimiter=" ")
        T_gt = np.loadtxt(seq_path + "/ir.txt", delimiter=" ")
    else:
        raise ValueError(f"Unknown dataset type: {set_type}")

    return RGB_img_list, T_img_list, RGB_gt, T_gt


def run_sequence(seq_name, seq_home, dataset_name, tracker, debug=0, save_results=True):
    if "VTUAV" in dataset_name:
        seq_txt = seq_name.split("/")[1]
    else:
        seq_txt = seq_name

    save_name = "..."
    save_folder = f"./RGBT_workspace/results/{dataset_name}/" + save_name
    save_path = save_folder + "/" + seq_txt + ".txt"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if os.path.exists(save_path):
        print(f"-1 {seq_name}")
        return

    seq_path = seq_home + "/" + seq_name
    print("——————————Process sequence: " + seq_name + "——————————————")
    sys.stdout.flush()

    RGB_img_list, T_img_list, RGB_gt, T_gt = genConfig(seq_path, dataset_name)

    if len(RGB_img_list) == len(RGB_gt):
        result = np.zeros_like(RGB_gt)
    else:
        result = np.zeros((len(RGB_img_list), 4), dtype=RGB_gt.dtype)

    result[0] = np.copy(RGB_gt[0])

    t_before = tracker.tracker.time_track_s
    calls_before = tracker.tracker.calls

    for frame_idx, (rgb_path, T_path) in enumerate(zip(RGB_img_list, T_img_list)):
        image = get_x_frame(rgb_path, T_path, dtype=getattr(params.cfg.DATA, "XTYPE", "rgbrgb"))

        if frame_idx == 0:
            tracker.tracker.seq_name = seq_name
            tracker.initialize(image, RGB_gt[0].tolist())
        else:
            region = tracker.track(image, info={"gt_bbox": RGB_gt[frame_idx].tolist()})
            result[frame_idx] = np.array(region)

    if save_results and not debug:
        np.savetxt(save_path, result)

    t_after = tracker.tracker.time_track_s
    calls_after = tracker.tracker.calls

    dt = t_after - t_before
    dframes = calls_after - calls_before

    if dt > 0:
        print(f"{seq_name} , fps:{dframes / dt}")
    else:
        print(f"{seq_name} , fps:inf (dt=0)")

    sys.stdout.flush()


class mmMobileViT_Track_RGBT(object):
    def __init__(self, tracker):
        self.tracker = tracker
        self.tracker.save_dir = ""
        self.seq_name = None

    def initialize(self, image, region):
        self.H, self.W, _ = image.shape
        gt_bbox_np = np.array(region).astype(np.float32)
        init_info = {"init_bbox": list(gt_bbox_np)}
        self.tracker.initialize(image, init_info)

    def track(self, img_RGB, info=None):
        outputs = self.tracker.track(img_RGB, info=info)
        return outputs["target_bbox"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tracker on RGBT dataset.")
    parser.add_argument("--script_name", type=str, default="prompt")
    parser.add_argument("--yaml_name", type=str, default="ViPT_deep_rgbt")
    parser.add_argument("--dataset_name", type=str, default="LasHeR")
    parser.add_argument("--threads", default=4, type=int)
    parser.add_argument("--num_gpus", default=torch.cuda.device_count(), type=int)
    parser.add_argument("--epoch", default=60, type=int)
    parser.add_argument("--mode", default="sequential", type=str)
    parser.add_argument("--debug", default=0, type=int)
    parser.add_argument("--video", default="", type=str)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass

    yaml_name = args.yaml_name
    dataset_name = args.dataset_name

    if args.script_name == "mmMobileViT_Track":
        params = rgbt_prompt_params.parameters(yaml_name)
        mmtrack = mmMobileViT_Track(params)
        tracker_obj = mmMobileViT_Track_RGBT(tracker=mmtrack)
    else:
        raise ValueError(f"Unsupported script_name: {args.script_name}")

    if dataset_name == "GTOT":
        seq_home = "/home/mark/Codes/mahdi_codes_folder/datasets/GTOT"
        seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home, f))]
    elif dataset_name == "RGBT234":
        seq_home = "/home/mark/Codes/mahdi_codes_folder/datasets/RGB-T234"
        seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home, f))]
    elif dataset_name == "RGBT210":
        seq_home = "/home/mark/Codes/mahdi_codes_folder/datasets/RGBT210"
        seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home, f))]
    elif dataset_name == "LasHeR":
        seq_home = "/home/mark/Codes/mahdi_codes_folder/datasets/lasher/testingset"
        seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home, f))]
    elif dataset_name == "VTUAVST":
        seq_home = "/mnt/6196b16a-836e-45a4-b6f2-641dca0991d0/VTUAV/test/short-term"
        with open(join(seq_home, "VTUAV-ST.txt"), "r") as f:
            seq_list = f.read().splitlines()
    elif dataset_name == "VTUAVLT":
        seq_home = "/mnt/6196b16a-836e-45a4-b6f2-641dca0991d0/VTUAV/test/long-term"
        with open(join(seq_home, "VTUAV-LT.txt"), "r") as f:
            seq_list = f.read().splitlines()
    else:
        raise ValueError("Error dataset!")

    seq_list = [args.video] if args.video != "" else seq_list

    if len(seq_list) > 1:
        print(f"--- Warm-up with: {seq_list[0]}. not saving results. ---")
        run_sequence(seq_list[0], seq_home, dataset_name, tracker_obj, args.debug, save_results=False)
        print("--- Warm-up done. ---")

    tracker_obj.tracker.reset_timing()

    sys.stdout.flush()
    for seq_name in seq_list:
        run_sequence(seq_name, seq_home, dataset_name, tracker_obj, args.debug, save_results=True)

    timing = tracker_obj.tracker.get_timing()

    total_frames_all = timing["calls"]
    total_time_all = timing["time_track_s"]

    print(
        f"Total frames is  {total_frames_all} and total_time is {total_time_all}. "
        f"Average FPS is {total_frames_all / total_time_all}!"
    )
