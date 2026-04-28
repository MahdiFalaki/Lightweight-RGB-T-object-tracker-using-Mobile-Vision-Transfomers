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
    RawInputModelAdapter,
    artifact_path,
    build_model,
    check_all_finite,
    create_dummy_inputs,
    get_environment_metadata,
    load_runtime_context,
    repo_relative,
    save_json,
    save_npz,
    summarize_mapping_tensors,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate deterministic PyTorch reference inputs and outputs.")
    parser.add_argument("--param-name", default=DEFAULT_PARAM_NAME)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--inputs-path", default=str(artifact_path("pytorch_inputs.npz")))
    parser.add_argument("--outputs-path", default=str(artifact_path("pytorch_outputs.npz")))
    parser.add_argument("--report-path", default=str(artifact_path("pytorch_smoke_report.json")))
    return parser.parse_args()


def main():
    args = parse_args()
    inputs_path = Path(args.inputs_path).resolve()
    outputs_path = Path(args.outputs_path).resolve()
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
        check_all_finite(outputs, "PyTorch outputs")
        save_npz(inputs_path, named_inputs)
        save_npz(outputs_path, outputs)

        report.update(
            {
                "status": "PASS",
                "device": str(runtime.device),
                "checkpoint_path": str(runtime.checkpoint_path),
                "inputs_path": str(inputs_path),
                "outputs_path": str(outputs_path),
                "inputs": summarize_mapping_tensors(named_inputs),
                "outputs": summarize_mapping_tensors(outputs),
                "pass_condition": "PASS: PyTorch smoke test completed and reference outputs saved.",
            }
        )
        save_json(report_path, report)

        for name, info in report["inputs"].items():
            print(
                "Input {0}: shape={1} min={2:.6f} max={3:.6f} mean={4:.6f}".format(
                    name, info["shape"], info["min"], info["max"], info["mean"]
                )
            )
        for name, info in report["outputs"].items():
            print(
                "Output {0}: shape={1} min={2:.6f} max={3:.6f} mean={4:.6f} finite={5}".format(
                    name, info["shape"], info["min"], info["max"], info["mean"], info["finite"]
                )
            )
        print("Saved inputs: {0}".format(repo_relative(inputs_path)))
        print("Saved outputs: {0}".format(repo_relative(outputs_path)))
        print("Saved report: {0}".format(repo_relative(report_path)))
        print("PASS: PyTorch smoke test completed and reference outputs saved.")
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
