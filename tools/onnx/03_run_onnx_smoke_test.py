import argparse
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import onnxruntime as ort

from tools.onnx.common import artifact_path, array_summary, get_environment_metadata, load_npz, repo_relative, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Run ONNX Runtime smoke test with the exported fixed-shape model.")
    parser.add_argument("--onnx-path", default=str(artifact_path("mmmobilevit_track_fixed.onnx")))
    parser.add_argument("--inputs-path", default=str(artifact_path("pytorch_inputs.npz")))
    parser.add_argument("--outputs-path", default=str(artifact_path("onnx_outputs.npz")))
    parser.add_argument("--report-path", default=str(artifact_path("onnx_smoke_report.json")))
    parser.add_argument("--providers", nargs="+", default=["CPUExecutionProvider"])
    return parser.parse_args()


def main():
    args = parse_args()
    onnx_path = Path(args.onnx_path).resolve()
    inputs_path = Path(args.inputs_path).resolve()
    outputs_path = Path(args.outputs_path).resolve()
    report_path = Path(args.report_path).resolve()
    report = {
        "status": "FAIL",
        "onnx_path": str(onnx_path),
        "inputs_path": str(inputs_path),
        "providers": args.providers,
        "environment": get_environment_metadata(),
    }
    try:
        if not onnx_path.is_file():
            raise FileNotFoundError("ONNX model does not exist: {0}".format(onnx_path))
        if not inputs_path.is_file():
            raise FileNotFoundError("Input NPZ does not exist: {0}".format(inputs_path))

        named_inputs = load_npz(inputs_path)
        session = ort.InferenceSession(str(onnx_path), providers=args.providers)
        session_input_names = [item.name for item in session.get_inputs()]
        ort_inputs = {name: named_inputs[name] for name in session_input_names}

        for name, value in ort_inputs.items():
            if not np.isfinite(value).all():
                raise ValueError("Input {0} contains non-finite values.".format(name))

        output_names = [item.name for item in session.get_outputs()]
        output_values = session.run(output_names, ort_inputs)
        ort_outputs = {name: value for name, value in zip(output_names, output_values)}

        for name, value in ort_outputs.items():
            if not np.isfinite(value).all():
                raise ValueError("ONNX output {0} contains non-finite values.".format(name))

        outputs_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(outputs_path), **ort_outputs)

        report.update(
            {
                "status": "PASS",
                "execution_provider": session.get_providers()[0] if session.get_providers() else "unknown",
                "session_inputs": session_input_names,
                "session_outputs": output_names,
                "output_summaries": {name: array_summary(value) for name, value in ort_outputs.items()},
                "outputs_path": str(outputs_path),
                "pass_condition": "PASS: ONNX Runtime executed the exported model successfully.",
            }
        )
        save_json(report_path, report)

        print("ONNX input names: {0}".format(", ".join(report["session_inputs"])))
        print("ONNX output names: {0}".format(", ".join(report["session_outputs"])))
        for name, info in report["output_summaries"].items():
            print(
                "Output {0}: shape={1} finite={2} min={3:.6f} max={4:.6f} mean={5:.6f}".format(
                    name, info["shape"], info["finite"], info["min"], info["max"], info["mean"]
                )
            )
        print("Saved outputs: {0}".format(repo_relative(outputs_path)))
        print("Saved report: {0}".format(repo_relative(report_path)))
        print("PASS: ONNX Runtime executed the exported model successfully.")
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
