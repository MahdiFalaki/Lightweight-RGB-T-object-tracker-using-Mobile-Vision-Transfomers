import argparse
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from tools.onnx.common import artifact_path, get_environment_metadata, load_npz, repo_relative, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Compare PyTorch reference outputs against ONNX Runtime outputs.")
    parser.add_argument("--pytorch-outputs", default=str(artifact_path("pytorch_outputs.npz")))
    parser.add_argument("--onnx-outputs", default=str(artifact_path("onnx_outputs.npz")))
    parser.add_argument("--report-path", default=str(artifact_path("comparison_report.json")))
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-4)
    return parser.parse_args()


def safe_relative_error(reference, candidate):
    denominator = np.maximum(np.abs(reference), 1e-12)
    return np.abs(candidate - reference) / denominator


def compare_arrays(reference, candidate, rtol, atol):
    if reference.shape != candidate.shape:
        return {
            "shape_match": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
            "allclose": False,
        }

    abs_err = np.abs(candidate - reference)
    rel_err = safe_relative_error(reference, candidate)
    return {
        "shape_match": True,
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "max_abs_error": float(abs_err.max()),
        "mean_abs_error": float(abs_err.mean()),
        "max_rel_error": float(rel_err.max()),
        "mean_rel_error": float(rel_err.mean()),
        "allclose": bool(np.allclose(reference, candidate, rtol=rtol, atol=atol)),
    }


def main():
    args = parse_args()
    pytorch_path = Path(args.pytorch_outputs).resolve()
    onnx_path = Path(args.onnx_outputs).resolve()
    report_path = Path(args.report_path).resolve()
    report = {
        "status": "FAIL",
        "rtol": args.rtol,
        "atol": args.atol,
        "pytorch_outputs": str(pytorch_path),
        "onnx_outputs": str(onnx_path),
        "environment": get_environment_metadata(),
    }
    try:
        pytorch_outputs = load_npz(pytorch_path)
        onnx_outputs = load_npz(onnx_path)
        common_keys = sorted(set(pytorch_outputs) & set(onnx_outputs))
        if not common_keys:
            raise ValueError("No matching output names were found between PyTorch and ONNX artifacts.")

        comparisons = {
            name: compare_arrays(pytorch_outputs[name], onnx_outputs[name], args.rtol, args.atol)
            for name in common_keys
        }
        failed = [name for name, result in comparisons.items() if not result["allclose"]]

        report.update(
            {
                "status": "PASS" if not failed else "FAIL",
                "common_output_names": common_keys,
                "only_in_pytorch": sorted(set(pytorch_outputs) - set(onnx_outputs)),
                "only_in_onnx": sorted(set(onnx_outputs) - set(pytorch_outputs)),
                "comparisons": comparisons,
                "pass_condition": "PASS: ONNX outputs are numerically close to PyTorch outputs.",
            }
        )
        if failed:
            report["failed_outputs"] = failed
            report["likely_cause"] = (
                "Mismatch after export/runtime execution. Inspect export_report.json for graph warnings and "
                "confirm the raw-input adapter and export-only fold/unfold path preserve the expected behavior."
            )
        save_json(report_path, report)

        for name in common_keys:
            result = comparisons[name]
            if result["shape_match"]:
                print(
                    "{0}: allclose={1} max_abs_error={2:.6e} mean_abs_error={3:.6e} max_rel_error={4:.6e}".format(
                        name,
                        result["allclose"],
                        result["max_abs_error"],
                        result["mean_abs_error"],
                        result["max_rel_error"],
                    )
                )
            else:
                print(
                    "{0}: allclose=False shape mismatch {1} vs {2}".format(
                        name, result["reference_shape"], result["candidate_shape"]
                    )
                )
        print("Saved report: {0}".format(repo_relative(report_path)))

        if failed:
            print(
                "FAIL: ONNX outputs are not numerically close to PyTorch outputs for {0}".format(", ".join(failed)),
                file=sys.stderr,
            )
            return 1

        print("PASS: ONNX outputs are numerically close to PyTorch outputs.")
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
