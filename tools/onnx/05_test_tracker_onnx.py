import argparse
import json
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import onnxruntime as ort
import torch

from lib.test.tracker.data_utils import Preprocessor
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
from lib.utils.box_ops import clip_box
from tools.onnx.common import DEFAULT_PARAM_NAME, build_params_from_config, get_environment_metadata


RGB_DIR_CANDIDATES = ["visible", "vis", "rgb", "RGB", "v"]
IR_DIR_CANDIDATES = ["infrared", "ir", "thermal", "T", "t"]
GT_FILE_CANDIDATES = [
    "groundtruth.txt",
    "groundtruth_rect.txt",
    "init.txt",
    "visible.txt",
    "infrared.txt",
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run a short tracker-level ONNX smoke test on real RGBT234 frames.")
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Root directory of the RGBT234 dataset.",
    )
    parser.add_argument(
        "--onnx",
        default="tools/onnx/artifacts/mmmobilevit_track_fixed.onnx",
        help="Path to the exported ONNX model.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Requested execution device. CPU is sufficient for this smoke test.",
    )
    parser.add_argument("--num-frames", type=int, default=10, help="Total number of frames to use including the template frame.")
    parser.add_argument("--sequence", default=None, help="Optional explicit RGBT234 sequence name.")
    parser.add_argument(
        "--output-dir",
        default="tools/onnx/artifacts/tracker_smoke/",
        help="Directory for smoke-test artifacts.",
    )
    return parser.parse_args()


def read_image_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError("Failed to read image: {0}".format(path))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def choose_provider(requested_device: str):
    available = ort.get_available_providers()
    warnings_list = []
    if requested_device == "cuda" and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], warnings_list
    if requested_device == "cuda":
        warnings_list.append("CUDAExecutionProvider is not available; falling back to CPUExecutionProvider.")
    return ["CPUExecutionProvider"], warnings_list


def find_named_child(sequence_dir: Path, candidates):
    lowered = {child.name.lower(): child for child in sequence_dir.iterdir()}
    for candidate in candidates:
        child = lowered.get(candidate.lower())
        if child is not None:
            return child
    return None


def list_image_files(folder: Path):
    if folder is None or not folder.is_dir():
        return []
    return sorted([path for path in folder.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file()])


def parse_bbox_line(line: str):
    raw = line.strip().replace("\t", ",").replace(" ", ",")
    parts = [token for token in raw.split(",") if token]
    if len(parts) < 4:
        raise ValueError("Could not parse bounding box line: {0}".format(line))
    return [float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])]


def load_groundtruth(path: Path):
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    return [parse_bbox_line(line) for line in lines]


def discover_sequence(dataset_root: Path, sequence_name: str = None):
    if not dataset_root.is_dir():
        raise FileNotFoundError("Dataset root does not exist: {0}".format(dataset_root))

    if sequence_name is not None:
        candidates = [dataset_root / sequence_name]
    else:
        candidates = sorted([path for path in dataset_root.iterdir() if path.is_dir()])

    for sequence_dir in candidates:
        rgb_dir = find_named_child(sequence_dir, RGB_DIR_CANDIDATES)
        ir_dir = find_named_child(sequence_dir, IR_DIR_CANDIDATES)
        if rgb_dir is None or ir_dir is None:
            continue

        gt_visible = sequence_dir / "visible.txt"
        gt_infrared = sequence_dir / "infrared.txt"
        gt_path = None
        for candidate in GT_FILE_CANDIDATES:
            path = sequence_dir / candidate
            if path.is_file():
                gt_path = path
                break
        if gt_path is None:
            continue

        rgb_frames = list_image_files(rgb_dir)
        ir_frames = list_image_files(ir_dir)
        if not rgb_frames or not ir_frames:
            continue

        return {
            "sequence_dir": sequence_dir,
            "sequence_name": sequence_dir.name,
            "rgb_dir": rgb_dir,
            "ir_dir": ir_dir,
            "gt_path": gt_path,
            "gt_visible_path": gt_visible if gt_visible.is_file() else None,
            "gt_infrared_path": gt_infrared if gt_infrared.is_file() else None,
            "rgb_frames": rgb_frames,
            "ir_frames": ir_frames,
        }

    raise RuntimeError("No valid RGBT234 sequence was found under {0}".format(dataset_root))


def map_box_back(state, pred_box, resize_factor, search_size):
    cx_prev = state[0] + 0.5 * state[2]
    cy_prev = state[1] + 0.5 * state[3]
    cx, cy, w, h = pred_box
    half_side = 0.5 * search_size / resize_factor
    cx_real = cx + (cx_prev - half_side)
    cy_real = cy + (cy_prev - half_side)
    return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]


def decode_bbox(score_map, size_map, offset_map, output_window, resize_factor, search_size, state, image_hw):
    response = score_map * output_window
    response_flat = response.reshape(-1)
    best_index = int(np.argmax(response_flat))
    feat_sz = score_map.shape[-1]
    idx_y = best_index // feat_sz
    idx_x = best_index % feat_sz

    offset_x = float(offset_map[0, idx_y, idx_x])
    offset_y = float(offset_map[1, idx_y, idx_x])
    size_w = float(size_map[0, idx_y, idx_x])
    size_h = float(size_map[1, idx_y, idx_x])

    pred_box = [
        ((idx_x + offset_x) / feat_sz) * search_size / resize_factor,
        ((idx_y + offset_y) / feat_sz) * search_size / resize_factor,
        size_w * search_size / resize_factor,
        size_h * search_size / resize_factor,
    ]
    mapped = map_box_back(state, pred_box, resize_factor, search_size)
    return clip_box(mapped, image_hw[0], image_hw[1], margin=10)


def maybe_save_debug_frame(output_dir: Path, frame_index: int, rgb_image: np.ndarray, bbox):
    debug_dir = output_dir / "debug_frames"
    debug_dir.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    x, y, w, h = [int(round(value)) for value in bbox]
    cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imwrite(str(debug_dir / "{0:04d}.jpg".format(frame_index)), image_bgr)


def main():
    args = parse_args()
    if not args.dataset_root:
        raise ValueError("--dataset-root is required. Example: --dataset-root <RGBT234_ROOT>")
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    onnx_path = Path(args.onnx).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "tracker_onnx_smoke_report.json"
    boxes_npy_path = output_dir / "tracker_onnx_boxes.npy"
    boxes_txt_path = output_dir / "tracker_onnx_boxes.txt"

    report = {
        "status": "FAIL",
        "onnx_model_path": str(onnx_path),
        "dataset_root": str(dataset_root),
        "num_requested_frames": int(args.num_frames),
        "input_shapes": {
            "rgb_template": [1, 3, 128, 128],
            "ir_template": [1, 3, 128, 128],
            "rgb_search": [1, 3, 256, 256],
            "ir_search": [1, 3, 256, 256],
        },
        "warnings": [],
        "limitations": [
            "This is a short tracker-level ONNX smoke test on a real RGBT234 sequence, not full dataset evaluation.",
            "The current implementation keeps preprocessing and bbox decoding in Python and uses ONNX Runtime only for the core network forward pass.",
            "RGBT234 provides modality-specific annotation files; this smoke test uses visible.txt as the primary initial/state annotation when available.",
        ],
        "environment": get_environment_metadata(),
    }

    try:
        sequence_info = discover_sequence(dataset_root, args.sequence)
        report["sequence_name"] = sequence_info["sequence_name"]
        report["rgb_frame_count"] = len(sequence_info["rgb_frames"])
        report["thermal_frame_count"] = len(sequence_info["ir_frames"])
        report["groundtruth_path"] = str(sequence_info["gt_path"])
        report["first_rgb_frame_path"] = str(sequence_info["rgb_frames"][0])
        report["first_thermal_frame_path"] = str(sequence_info["ir_frames"][0])
        report["rgb_frame_folder"] = str(sequence_info["rgb_dir"])
        report["thermal_frame_folder"] = str(sequence_info["ir_dir"])

        if sequence_info["gt_visible_path"] is not None and sequence_info["gt_infrared_path"] is not None:
            report["warnings"].append(
                "Both visible.txt and infrared.txt exist; using visible.txt as the primary initial/state annotation file."
            )

        gt_boxes = load_groundtruth(sequence_info["gt_path"])
        if not gt_boxes:
            raise RuntimeError("Ground-truth file is empty: {0}".format(sequence_info["gt_path"]))

        num_total_frames = min(len(sequence_info["rgb_frames"]), len(sequence_info["ir_frames"]), len(gt_boxes), int(args.num_frames))
        if num_total_frames < 2:
            raise RuntimeError("Need at least 2 aligned frames for tracker-level ONNX smoke testing.")
        report["num_processed_frames"] = int(num_total_frames)

        providers, provider_warnings = choose_provider(args.device)
        report["warnings"].extend(provider_warnings)
        if not onnx_path.is_file():
            raise FileNotFoundError("ONNX model does not exist: {0}".format(onnx_path))

        session = ort.InferenceSession(str(onnx_path), providers=providers)
        report["onnx_input_names"] = [item.name for item in session.get_inputs()]
        report["onnx_output_names"] = [item.name for item in session.get_outputs()]
        report["onnx_execution_provider"] = session.get_providers()[0] if session.get_providers() else "unknown"

        params = build_params_from_config(DEFAULT_PARAM_NAME)
        template_size = int(params.template_size)
        search_size = int(params.search_size)
        template_factor = float(params.template_factor)
        search_factor = float(params.search_factor)
        feat_sz = int(search_size // params.cfg.MODEL.BACKBONE.STRIDE)
        output_window = hann2d(torch.tensor([feat_sz, feat_sz]).long(), centered=True).cpu().numpy()[0, 0]
        preprocessor = Preprocessor(device="cpu")

        rgb_first = read_image_rgb(sequence_info["rgb_frames"][0])
        ir_first = read_image_rgb(sequence_info["ir_frames"][0])
        if rgb_first.shape[:2] != ir_first.shape[:2]:
            raise RuntimeError("RGB and thermal frame sizes do not match on the first frame.")

        state = [float(value) for value in gt_boxes[0]]
        report["initial_bbox"] = state
        template_image = np.concatenate([rgb_first, ir_first], axis=2)
        template_crop, _, template_amask = sample_target(template_image, state, template_factor, output_sz=template_size)
        template_nested = preprocessor.process(template_crop, template_amask)
        template_np = template_nested.tensors.cpu().numpy().astype(np.float32)
        rgb_template = template_np[:, :3, :, :]
        ir_template = template_np[:, 3:, :, :]

        predicted_boxes = [state]
        output_shapes = {}
        finite_output_checks = {name: True for name in report["onnx_output_names"]}

        for frame_index in range(1, num_total_frames):
            rgb_frame = read_image_rgb(sequence_info["rgb_frames"][frame_index])
            ir_frame = read_image_rgb(sequence_info["ir_frames"][frame_index])
            if rgb_frame.shape[:2] != ir_frame.shape[:2]:
                raise RuntimeError("RGB and thermal frame sizes do not match on frame index {0}.".format(frame_index))

            search_image = np.concatenate([rgb_frame, ir_frame], axis=2)
            search_crop, resize_factor, search_amask = sample_target(search_image, state, search_factor, output_sz=search_size)
            search_nested = preprocessor.process(search_crop, search_amask)
            search_np = search_nested.tensors.cpu().numpy().astype(np.float32)
            rgb_search = search_np[:, :3, :, :]
            ir_search = search_np[:, 3:, :, :]

            ort_inputs = {
                "rgb_template": rgb_template,
                "ir_template": ir_template,
                "rgb_search": rgb_search,
                "ir_search": ir_search,
            }
            ort_outputs = session.run(report["onnx_output_names"], ort_inputs)
            outputs_by_name = {name: value for name, value in zip(report["onnx_output_names"], ort_outputs)}

            for name, value in outputs_by_name.items():
                output_shapes[name] = list(value.shape)
                is_finite = bool(np.isfinite(value).all())
                finite_output_checks[name] = finite_output_checks[name] and is_finite
                if not is_finite:
                    raise ValueError("Non-finite ONNX output detected in {0} on frame {1}.".format(name, frame_index))

            state = decode_bbox(
                score_map=outputs_by_name["score_map"][0, 0],
                size_map=outputs_by_name["size_map"][0],
                offset_map=outputs_by_name["offset_map"][0],
                output_window=output_window,
                resize_factor=resize_factor,
                search_size=search_size,
                state=state,
                image_hw=rgb_frame.shape[:2],
            )
            predicted_boxes.append([float(value) for value in state])

            if frame_index <= 3:
                maybe_save_debug_frame(output_dir, frame_index, rgb_frame, state)

        predicted_boxes_array = np.asarray(predicted_boxes, dtype=np.float32)
        np.save(str(boxes_npy_path), predicted_boxes_array)
        np.savetxt(str(boxes_txt_path), predicted_boxes_array, fmt="%.6f", delimiter=",")

        report.update(
            {
                "status": "PASS",
                "output_shapes": output_shapes,
                "finite_output_checks": finite_output_checks,
                "predicted_boxes_shape": list(predicted_boxes_array.shape),
                "predicted_boxes_path": str(boxes_npy_path),
                "predicted_boxes_text_path": str(boxes_txt_path),
            }
        )

        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)

        print("Sequence: {0}".format(report["sequence_name"]))
        print("RGB frames: {0}".format(report["rgb_frame_count"]))
        print("Thermal frames: {0}".format(report["thermal_frame_count"]))
        print("Ground truth: {0}".format(report["groundtruth_path"]))
        print("Frames used: {0}".format(report["num_processed_frames"]))
        print("ONNX provider: {0}".format(report["onnx_execution_provider"]))
        print("Saved report: {0}".format(report_path))
        print("Saved boxes: {0}".format(boxes_npy_path))
        print("PASS: Tracker-level ONNX smoke test completed.")
        return 0
    except Exception as exc:
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print("FAIL: {0}".format(exc), file=sys.stderr)
        print("Saved failure report: {0}".format(report_path), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
