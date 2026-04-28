import argparse
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.onnx.common import (
    DEFAULT_PARAM_NAME,
    DEFAULT_SEED,
    INPUT_NAMES,
    RawInputModelAdapter,
    artifact_path,
    build_model,
    check_all_finite,
    create_dummy_inputs,
    get_environment_metadata,
    load_runtime_context,
    repo_relative,
    save_json,
    summarize_mapping_tensors,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect the current PyTorch model I/O with deterministic RGB-T inputs.")
    parser.add_argument("--param-name", default=DEFAULT_PARAM_NAME)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--report-path", default=str(artifact_path("model_io_report.json")))
    return parser.parse_args()


def main():
    args = parse_args()
    report_path = Path(args.report_path).resolve()
    report = {
        "status": "FAIL",
        "param_name": args.param_name,
        "device_request": args.device,
        "seed": args.seed,
        "environment": get_environment_metadata(),
        "model_execution_mode": "raw-input adapter around current upstream test-mode backbone",
    }
    try:
        runtime, params = load_runtime_context(args.param_name, args.checkpoint, args.device, args.seed)
        model = build_model(params, runtime)
        adapter = RawInputModelAdapter(model)
        named_inputs = create_dummy_inputs(params, runtime.device, runtime.seed)
        outputs = adapter.forward_raw(
            named_inputs["rgb_template"],
            named_inputs["ir_template"],
            named_inputs["rgb_search"],
            named_inputs["ir_search"],
        )
        check_all_finite(outputs, "Model outputs")

        report.update(
            {
                "status": "PASS",
                "device": str(runtime.device),
                "checkpoint_path": str(runtime.checkpoint_path),
                "model_class": model.__class__.__name__,
                "model_module": model.__class__.__module__,
                "input_names": INPUT_NAMES,
                "inputs": summarize_mapping_tensors(named_inputs),
                "output_type": type(outputs).__name__,
                "output_keys": list(outputs.keys()),
                "outputs": summarize_mapping_tensors(outputs),
                "pass_condition": "PASS: PyTorch model loaded and one controlled forward pass completed.",
            }
        )
        save_json(report_path, report)

        print("Model class: {0}".format(report["model_class"]))
        print("Device: {0}".format(report["device"]))
        for name, info in report["inputs"].items():
            print("Input {0}: shape={1} dtype={2}".format(name, info["shape"], info["dtype"]))
        print("Output type: {0}".format(report["output_type"]))
        print("Output keys: {0}".format(", ".join(report["output_keys"])))
        for name, info in report["outputs"].items():
            print(
                "Output {0}: shape={1} finite={2} min={3:.6f} max={4:.6f}".format(
                    name, info["shape"], info["finite"], info["min"], info["max"]
                )
            )
        print("Saved JSON report: {0}".format(repo_relative(report_path)))
        print("PASS: PyTorch model loaded and one controlled forward pass completed.")
        return 0
    except Exception as exc:
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        save_json(report_path, report)
        print("FAIL: {0}".format(exc), file=sys.stderr)
        print("Saved failure report: {0}".format(repo_relative(report_path)), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
