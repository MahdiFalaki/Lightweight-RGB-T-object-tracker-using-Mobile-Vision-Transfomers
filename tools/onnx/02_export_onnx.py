import argparse
import sys
import traceback
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import onnx
import torch

from tools.onnx.common import (
    DEFAULT_PARAM_NAME,
    DEFAULT_SEED,
    INPUT_NAMES,
    ONNXExportWrapper,
    OUTPUT_NAMES,
    RawInputModelAdapter,
    artifact_path,
    build_model,
    create_dummy_inputs,
    default_export_opset,
    export_opset_reason,
    get_environment_metadata,
    load_runtime_context,
    prepare_model_for_onnx_export,
    repo_relative,
    save_json,
    summarize_mapping_tensors,
    warning_strings,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Export the ONNX deployment-validation model with fixed raw RGB-T inputs.")
    parser.add_argument("--param-name", default=DEFAULT_PARAM_NAME)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--opset", type=int, default=None)
    parser.add_argument("--onnx-path", default=str(artifact_path("mmmobilevit_track_fixed.onnx")))
    parser.add_argument("--report-path", default=str(artifact_path("export_report.json")))
    return parser.parse_args()


def main():
    args = parse_args()
    onnx_path = Path(args.onnx_path).resolve()
    report_path = Path(args.report_path).resolve()
    opset = args.opset if args.opset is not None else default_export_opset()
    report = {
        "status": "FAIL",
        "param_name": args.param_name,
        "device_request": args.device,
        "seed": args.seed,
        "opset_version": opset,
        "onnx_path": str(onnx_path),
        "environment": get_environment_metadata(),
        "opset_reason": export_opset_reason(opset),
        "model_execution_mode": "raw-input adapter that precomputes template layer-2 features",
    }
    try:
        runtime, params = load_runtime_context(args.param_name, args.checkpoint, args.device, args.seed)
        model = build_model(params, runtime)
        prepare_model_for_onnx_export(model)
        adapter = RawInputModelAdapter(model)
        wrapper = ONNXExportWrapper(adapter).to(runtime.device)
        wrapper.eval()

        named_inputs = create_dummy_inputs(params, runtime.device, runtime.seed)
        export_args = tuple(named_inputs[name] for name in INPUT_NAMES)
        output_tuple = wrapper(*export_args)
        output_summary = summarize_mapping_tensors(
            {name: value for name, value in zip(OUTPUT_NAMES, output_tuple)}
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with torch.no_grad():
                torch.onnx.export(
                    wrapper,
                    export_args,
                    str(onnx_path),
                    export_params=True,
                    opset_version=opset,
                    do_constant_folding=True,
                    input_names=INPUT_NAMES,
                    output_names=OUTPUT_NAMES,
                    dynamic_axes=None,
                )

        if not onnx_path.is_file():
            raise FileNotFoundError("ONNX export did not create a file at {0}".format(onnx_path))

        model_proto = onnx.load(str(onnx_path))
        onnx.checker.check_model(model_proto)

        warning_list = list(warning_strings(caught))
        report.update(
            {
                "status": "PASS",
                "device": str(runtime.device),
                "checkpoint_path": str(runtime.checkpoint_path),
                "input_names": INPUT_NAMES,
                "output_names": OUTPUT_NAMES,
                "input_summaries": summarize_mapping_tensors(named_inputs),
                "output_summaries": output_summary,
                "graph_inputs": [value.name for value in model_proto.graph.input],
                "graph_outputs": [value.name for value in model_proto.graph.output],
                "onnx_checker_passed": True,
                "warnings": warning_list,
                "file_size_bytes": onnx_path.stat().st_size,
                "pass_condition": "PASS: Fixed-shape ONNX export completed and ONNX checker passed.",
            }
        )
        save_json(report_path, report)

        print("Opset version: {0}".format(opset))
        print("ONNX file: {0}".format(repo_relative(onnx_path)))
        print("Input names: {0}".format(", ".join(report["graph_inputs"])))
        print("Output names: {0}".format(", ".join(report["graph_outputs"])))
        print("Captured warnings: {0}".format(len(warning_list)))
        print("Saved report: {0}".format(repo_relative(report_path)))
        print("PASS: Fixed-shape ONNX export completed and ONNX checker passed.")
        return 0
    except Exception as exc:
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        message = str(exc)
        if "Unsupported ONNX opset version" in message:
            report["likely_unsupported_operation"] = "torch.onnx exporter version/opset compatibility"
            report["recommended_next_fix"] = (
                "Use an opset supported by the active torch exporter. In mobilevit-track with torch 1.12.0, "
                "opset 16 is the safe default."
            )
        elif "col2im" in message or "fold" in message:
            report["likely_unsupported_operation"] = "fold/col2im inside MobileViT blocks or head"
            report["recommended_next_fix"] = (
                "Keep the export-only CoreML-compatible fold/unfold path enabled during export."
            )
        save_json(report_path, report)
        print("FAIL: {0}".format(exc), file=sys.stderr)
        print("Saved failure report: {0}".format(repo_relative(report_path)), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
