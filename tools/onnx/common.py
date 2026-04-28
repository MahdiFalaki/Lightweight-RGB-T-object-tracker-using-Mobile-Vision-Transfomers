import copy
import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = PROJECT_ROOT / "tools" / "onnx"
ARTIFACTS_DIR = TOOLS_DIR / "artifacts"
DEFAULT_PARAM_NAME = "mobilevitv2_256_128x1_LasHeR_60ep"
DEFAULT_SEED = 123
INPUT_NAMES = ["rgb_template", "ir_template", "rgb_search", "ir_search"]
OUTPUT_NAMES = ["pred_boxes", "score_map", "size_map", "offset_map"]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.config.mmMobileViT_Track.config import cfg, update_config_from_file
from lib.models.mmMobileViT_Track import build_mmMobileViT_Track


@dataclass
class RuntimeContext:
    param_name: str
    device: torch.device
    checkpoint_path: Path
    seed: int


def ensure_artifacts_dir() -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR


def artifact_path(filename: str) -> Path:
    ensure_artifacts_dir()
    return ARTIFACTS_DIR / filename


def repo_relative(path: Path) -> str:
    return os.path.relpath(str(path), str(PROJECT_ROOT))


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this environment.")
    return torch.device(device_name)


def build_params_from_config(param_name: str) -> Any:
    yaml_path = PROJECT_ROOT / "experiments" / "mmMobileViT_Track" / "{0}.yaml".format(param_name)
    if not yaml_path.is_file():
        raise FileNotFoundError("Parameter YAML does not exist: {0}".format(yaml_path))

    cfg_copy = copy.deepcopy(cfg)
    update_config_from_file(str(yaml_path), base_cfg=cfg_copy)

    params = SimpleNamespace()
    params.cfg = cfg_copy
    params.template_factor = cfg_copy.TEST.TEMPLATE_FACTOR
    params.template_size = cfg_copy.TEST.TEMPLATE_SIZE
    params.search_factor = cfg_copy.TEST.SEARCH_FACTOR
    params.search_size = cfg_copy.TEST.SEARCH_SIZE
    params.checkpoint = str(
        PROJECT_ROOT
        / "models"
        / "mmMobileViT_Track_LasHeR_60ep"
        / "checkpoints"
        / "train"
        / "mmMobileViT_Track"
        / param_name
        / "mmMobileViT_Track_ep{0:04d}.pth.tar".format(int(cfg_copy.TEST.EPOCH))
    )
    params.save_all_boxes = False
    return params


def discover_checkpoint_candidates(param_name: str, epoch: int) -> Sequence[str]:
    filename = "mmMobileViT_Track_ep{0:04d}.pth.tar".format(epoch)
    roots = [PROJECT_ROOT, PROJECT_ROOT.parent, PROJECT_ROOT.parent / "mmMobileViT_Track"]
    matches = []
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for match in root.glob("**/{0}".format(filename)):
            text = str(match.resolve())
            if param_name in text and text not in seen:
                seen.add(text)
                matches.append(text)
    return sorted(matches)


def load_runtime_context(
    param_name: str,
    checkpoint: Optional[str],
    device_name: str,
    seed: int,
) -> Tuple[RuntimeContext, Any]:
    params = build_params_from_config(param_name)
    device = resolve_device(device_name)
    params.cfg.TEST.DEVICE = str(device)
    checkpoint_path = Path(checkpoint).expanduser().resolve() if checkpoint else Path(params.checkpoint).resolve()

    runtime = RuntimeContext(
        param_name=param_name,
        device=device,
        checkpoint_path=checkpoint_path,
        seed=seed,
    )
    return runtime, params


def build_model(params: Any, runtime: RuntimeContext) -> nn.Module:
    model = build_mmMobileViT_Track(params.cfg, training=False)

    if not runtime.checkpoint_path.is_file():
        candidates = discover_checkpoint_candidates(runtime.param_name, int(params.cfg.TEST.EPOCH))
        message = "Checkpoint file does not exist: {0}".format(runtime.checkpoint_path)
        if candidates:
            message += "\nCandidate checkpoints:\n- " + "\n- ".join(candidates)
        raise FileNotFoundError(message)

    checkpoint_obj = torch.load(str(runtime.checkpoint_path), map_location="cpu")
    state_dict = checkpoint_obj["net"] if isinstance(checkpoint_obj, Mapping) and "net" in checkpoint_obj else checkpoint_obj
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint load mismatch. Missing keys: {0}. Unexpected keys: {1}.".format(missing, unexpected)
        )

    model = model.to(runtime.device)
    model.eval()
    return model


def create_dummy_inputs(params: Any, device: torch.device, seed: int) -> Dict[str, torch.Tensor]:
    set_seed(seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    def _rand(shape: Tuple[int, ...]) -> torch.Tensor:
        return torch.rand(shape, generator=generator, dtype=torch.float32).to(device)

    template_size = int(params.template_size)
    search_size = int(params.search_size)
    return {
        "rgb_template": _rand((1, 3, template_size, template_size)),
        "ir_template": _rand((1, 3, template_size, template_size)),
        "rgb_search": _rand((1, 3, search_size, search_size)),
        "ir_search": _rand((1, 3, search_size, search_size)),
    }


def prepare_template_features(model: nn.Module, rgb_template: torch.Tensor, ir_template: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # The current upstream main branch uses the tracker-style test path:
    # template tensors are cached after conv_1/layer_1/layer_2 and then passed as z.
    z_v = model.backbone.conv_1(rgb_template)
    z_i = model.backbone.conv_1(ir_template)
    z_v = model.backbone.layer_1(z_v)
    z_i = model.backbone.layer_1(z_i)
    z_v = model.backbone.layer_2(z_v)
    z_i = model.backbone.layer_2(z_i)
    return z_v, z_i


def flatten_output_tensors(outputs: Any) -> Dict[str, torch.Tensor]:
    if not isinstance(outputs, Mapping):
        raise TypeError("Expected model output to be a mapping, got {0!r}".format(type(outputs)))

    flattened = {}
    bad = []
    for name, value in outputs.items():
        if torch.is_tensor(value):
            flattened[str(name)] = value
        else:
            bad.append("{0}: {1}".format(name, type(value).__name__))
    if bad:
        raise TypeError("Model outputs must be tensors. Unsupported entries: {0}".format(", ".join(bad)))
    return flattened


def forward_model_from_raw_inputs(model: nn.Module, named_inputs: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    with torch.no_grad():
        template_features = prepare_template_features(
            model,
            named_inputs["rgb_template"],
            named_inputs["ir_template"],
        )
        outputs = model(
            template=[template_features[0], template_features[1]],
            search=[named_inputs["rgb_search"], named_inputs["ir_search"]],
        )
    return flatten_output_tensors(outputs)


def check_all_finite(items: Mapping[str, torch.Tensor], label: str) -> None:
    bad = [name for name, value in items.items() if not torch.isfinite(value).all().item()]
    if bad:
        raise ValueError("{0} contains non-finite values in: {1}".format(label, ", ".join(bad)))


def tensor_summary(tensor: torch.Tensor) -> Dict[str, Any]:
    detached = tensor.detach()
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "min": float(detached.min().item()),
        "max": float(detached.max().item()),
        "mean": float(detached.mean().item()),
        "finite": bool(torch.isfinite(detached).all().item()),
    }


def summarize_mapping_tensors(items: Mapping[str, torch.Tensor]) -> Dict[str, Dict[str, Any]]:
    return {name: tensor_summary(value) for name, value in items.items()}


def array_summary(array: np.ndarray) -> Dict[str, Any]:
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "finite": bool(np.isfinite(array).all()),
    }


def save_json(path: Path, payload: MutableMapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_npz(path: Path, items: Mapping[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{name: value.detach().cpu().numpy() for name, value in items.items()})


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(str(path), allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def get_environment_metadata() -> Dict[str, Any]:
    metadata = {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    for module_name, key in (("onnx", "onnx_version"), ("onnxruntime", "onnxruntime_version")):
        try:
            module = __import__(module_name)
            metadata[key] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            metadata[key] = "unavailable: {0}".format(exc)
    return metadata


def default_export_opset() -> int:
    version = torch.__version__.split("+")[0]
    parts = version.split(".")[:2]
    try:
        major = int(parts[0])
        minor = int(parts[1])
    except Exception:
        return 16
    if major >= 2:
        return 18
    if major == 1 and minor >= 13:
        return 17
    return 16


def export_opset_reason(opset: int) -> str:
    if opset == 16:
        return (
            "Using opset 16 because this environment runs torch {0}, whose exporter supports "
            "opset 16 reliably. The export wrapper precomputes template layer-2 features and "
            "uses the CoreML-compatible fold/unfold path to avoid lower-opset col2im issues."
        ).format(torch.__version__)
    if opset == 18:
        return "Using opset 18 because newer torch exporters can target it directly."
    return "Using a user-requested opset override."


def prepare_model_for_onnx_export(model: nn.Module) -> None:
    for module in model.modules():
        if hasattr(module, "enable_coreml_compatible_fn"):
            module.enable_coreml_compatible_fn = True
            if hasattr(module, "_compute_unfolding_weights") and not hasattr(module, "unfolding_weights"):
                module.register_buffer(
                    name="unfolding_weights",
                    tensor=module._compute_unfolding_weights(),
                    persistent=False,
                )


class RawInputModelAdapter(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward_raw(
        self,
        rgb_template: torch.Tensor,
        ir_template: torch.Tensor,
        rgb_search: torch.Tensor,
        ir_search: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        z_v, z_i = prepare_template_features(self.model, rgb_template, ir_template)
        outputs = self.model(
            template=[z_v, z_i],
            search=[rgb_search, ir_search],
        )
        return flatten_output_tensors(outputs)


class ONNXExportWrapper(nn.Module):
    def __init__(self, adapter: RawInputModelAdapter):
        super().__init__()
        self.adapter = adapter

    def forward(
        self,
        rgb_template: torch.Tensor,
        ir_template: torch.Tensor,
        rgb_search: torch.Tensor,
        ir_search: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.adapter.forward_raw(rgb_template, ir_template, rgb_search, ir_search)
        return tuple(outputs[name] for name in OUTPUT_NAMES)


def warning_strings(caught_warnings: Sequence[warnings.WarningMessage]) -> Sequence[str]:
    return ["{0}: {1}".format(type(item.message).__name__, item.message) for item in caught_warnings]
