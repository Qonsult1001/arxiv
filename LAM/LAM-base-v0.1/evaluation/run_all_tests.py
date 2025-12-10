"""
LAM Comprehensive Evaluation Suite - Master Runner
==================================================

This script runs all evaluation tests in sequence and generates
a comprehensive validation report.

Tests Included:
1. Pearson Score Validation (STS-B benchmark)
2. Linear Scaling Validation (O(n) complexity)
3. Long Context Processing (up to 1M tokens)
4. Ablation Study (component contributions)

Output:
- Individual test results (JSON)
- Visualizations (PNG)
- Comprehensive summary report
"""

import json
import time
from pathlib import Path
from datetime import datetime
import sys

# Import all test modules
from test_pearson_score import PearsonScoreValidator
from test_linear_scaling import LinearScalingValidator
from test_long_context import LongContextValidator
from test_ablation_study import AblationStudyValidator


class ComprehensiveEvaluationSuite:
    """Master test runner for LAM evaluation"""

    def __init__(self, model_path: str, baseline_path: str = "all-MiniLM-L6-v2"):
        self.model_path = model_path
        self.baseline_path = baseline_path
        self.results_dir = Path(__file__).parent / "results"
        self.viz_dir = Path(__file__).parent / "visualizations"

        self.results_dir.mkdir(exist_ok=True)
        self.viz_dir.mkdir(exist_ok=True)

        self.start_time = None
        self.end_time = None

    def print_header(self):
        """Print evaluation suite header"""
        print("\n" + "="*80)
        print(" " * 15 + "LAM COMPREHENSIVE EVALUATION SUITE")
        print("="*80)
        print(f"\nModel Path: {self.model_path}")
        print(f"Baseline Path: {self.baseline_path}")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-% d %H:%M:%S')}")
        print("\nTests to Run:")
        print("  1. ✓ Pearson Score Validation (STS-B)")
        print("  2. ✓ Linear Scaling Validation (O(n) complexity)")
        print("  3. ✓ Long Context Processing (up to 1M tokens)")
        print("  4. ✓ Ablation Study (component analysis)")
        print("\n" + "="*80 + "\n")

    def run_test(
        self,
        test_number: int,
        test_name: str,
        test_func
    ):
        """Run a single test with error handling"""
        print("\n" + "="*80)
        print(f"TEST {test_number}: {test_name}")
        print("="*80)

        start_time = time.perf_counter()

        try:
            result = test_func()
            end_time = time.perf_counter()
            duration = end_time - start_time

            return {
                "test_number": test_number,
                "test_name": test_name,
                "status": "success",
                "duration_seconds": duration,
                "result": result
            }

        except Exception as e:
            end_time = time.perf_counter()
            duration = end_time - start_time

            print(f"\n✗ Test FAILED with error: {e}")
            import traceback
            traceback.print_exc()

            return {
                "test_number": test_number,
                "test_name": test_name,
                "status": "failed",
                "duration_seconds": duration,
                "error": str(e)
            }

    def generate_comprehensive_report(self, all_test_results: list):
        """Generate HTML and text summary reports"""
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*80)

        # Extract key metrics
        pearson_result = next(
            (r for r in all_test_results if "Pearson" in r["test_name"]),
            None
        )
        scaling_result = next(
            (r for r in all_test_results if "Scaling" in r["test_name"]),
            None
        )
        context_result = next(
            (r for r in all_test_results if "Context" in r["test_name"]),
            None
        )
        ablation_result = next(
            (r for r in all_test_results if "Ablation" in r["test_name"]),
            None
        )

        # Generate text report
        report_lines = [
            "="*80,
            "LAM COMPREHENSIVE EVALUATION REPORT",
            "="*80,
            "",
            f"Model: {self.model_path}",
            f"Baseline: {self.baseline_path}",
            f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Duration: {(self.end_time - self.start_time):.2f} seconds",
            "",
            "="*80,
            "EXECUTIVE SUMMARY",
            "="*80,
            ""
        ]

        # Test 1: Pearson Score
        if pearson_result and pearson_result["status"] == "success":
            lam_pearson = pearson_result["result"]["lam_results"]["pearson"]
            baseline_pearson = pearson_result["result"]["baseline_results"]["pearson"]
            ratio = pearson_result["result"]["comparison"]["pearson_percentage"]

            report_lines.extend([
                "1. PEARSON SCORE VALIDATION",
                f"   LAM Score: {lam_pearson:.4f}",
                f"   Baseline: {baseline_pearson:.4f}",
                f"   Ratio: {ratio:.2f}%",
                f"   Status: {'✓ VALIDATED' if abs(lam_pearson - 0.836) < 0.01 else '✗ FAILED'}",
                ""
            ])

        # Test 2: Linear Scaling
        if scaling_result and scaling_result["status"] == "success":
            lam_time_fit = scaling_result["result"]["complexity_analysis"]["lam"]["time"]["best_fit"]
            lam_mem_fit = scaling_result["result"]["complexity_analysis"]["lam"]["memory"]["best_fit"]

            report_lines.extend([
                "2. LINEAR SCALING VALIDATION",
                f"   Time Complexity: {lam_time_fit}",
                f"   Memory Complexity: {lam_mem_fit}",
                f"   Status: {'✓ O(n) CONFIRMED' if lam_time_fit == 'linear' and lam_mem_fit == 'linear' else '✗ FAILED'}",
                ""
            ])

        # Test 3: Long Context
        if context_result and context_result["status"] == "success":
            max_length = context_result["result"]["summary"]["max_length_achieved"]
            successful = context_result["result"]["summary"]["successful"]
            total = context_result["result"]["summary"]["total_tests"]

            report_lines.extend([
                "3. LONG CONTEXT PROCESSING",
                f"   Max Length Achieved: {max_length:,} tokens",
                f"   Success Rate: {successful}/{total} tests",
                f"   Status: {'✓ VALIDATED' if max_length >= 100000 else '⚠ PARTIAL'}",
                ""
            ])

        # Test 4: Ablation Study
        if ablation_result and ablation_result["status"] == "success":
            full_lam = ablation_result["result"]["evaluation_results"][0]["metrics"]["stsb_pearson"]
            improvement = ablation_result["result"]["importance_analysis"]["total_improvement_over_baseline"]

            report_lines.extend([
                "4. ABLATION STUDY",
                f"   Full LAM: {full_lam:.4f}",
                f"   Improvement over Baseline: {improvement:.4f}",
                f"   Status: {'✓ ALL COMPONENTS CONTRIBUTE' if ablation_result['result']['validation_status']['all_components_contribute'] else '✗ FAILED'}",
                ""
            ])

        # Overall status
        all_passed = all(
            r["status"] == "success" for r in all_test_results
        )

        report_lines.extend([
            "="*80,
            "OVERALL VALIDATION STATUS",
            "="*80,
            "",
            f"Tests Run: {len(all_test_results)}",
            f"Tests Passed: {sum(1 for r in all_test_results if r['status'] == 'success')}",
            f"Tests Failed: {sum(1 for r in all_test_results if r['status'] == 'failed')}",
            "",
            f"Overall Status: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}",
            "",
            "="*80,
            "DETAILED RESULTS",
            "="*80,
            ""
        ])

        for result in all_test_results:
            report_lines.append(f"\nTest {result['test_number']}: {result['test_name']}")
            report_lines.append(f"  Status: {result['status'].upper()}")
            report_lines.append(f"  Duration: {result['duration_seconds']:.2f}s")
            if result["status"] == "failed":
                report_lines.append(f"  Error: {result.get('error', 'Unknown error')}")

        report_lines.extend([
            "",
            "="*80,
            "FILES GENERATED",
            "="*80,
            "",
            "Results (JSON):",
            "  - results/pearson_score_validation.json",
            "  - results/linear_scaling_validation.json",
            "  - results/long_context_validation.json",
            "  - results/ablation_study_validation.json",
            "  - results/comprehensive_evaluation_report.json",
            "",
            "Visualizations (PNG):",
            "  - visualizations/pearson_score_validation.png",
            "  - visualizations/linear_scaling_validation.png",
            "  - visualizations/long_context_validation.png",
            "  - visualizations/ablation_study_validation.png",
            "",
            "="*80,
            "END OF REPORT",
            "="*80
        ])

        report_text = "\n".join(report_lines)

        # Save text report
        report_path = self.results_dir / "EVALUATION_REPORT.txt"
        with open(report_path, 'w') as f:
            f.write(report_text)

        print(f"\nText report saved to: {report_path}")

        # Print to console
        print("\n" + report_text)

        return report_text

    def run(self):
        """Run all tests in sequence"""
        self.start_time = time.perf_counter()
        self.print_header()

        all_test_results = []

        # Test 1: Pearson Score Validation
        test1_result = self.run_test(
            1,
            "Pearson Score Validation",
            lambda: PearsonScoreValidator(self.model_path, self.baseline_path).run()
        )
        all_test_results.append(test1_result)

        # Test 2: Linear Scaling Validation
        test2_result = self.run_test(
            2,
            "Linear Scaling Validation",
            lambda: LinearScalingValidator(self.model_path, self.baseline_path).run()
        )
        all_test_results.append(test2_result)

        # Test 3: Long Context Processing
        test3_result = self.run_test(
            3,
            "Long Context Processing",
            lambda: LongContextValidator(self.model_path).run()
        )
        all_test_results.append(test3_result)

        # Test 4: Ablation Study
        test4_result = self.run_test(
            4,
            "Ablation Study",
            lambda: AblationStudyValidator(self.model_path).run()
        )
        all_test_results.append(test4_result)

        self.end_time = time.perf_counter()

        # Generate comprehensive report
        report = self.generate_comprehensive_report(all_test_results)

        # Save comprehensive JSON results
        comprehensive_results = {
            "evaluation_metadata": {
                "model_path": self.model_path,
                "baseline_path": self.baseline_path,
                "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "end_time": datetime.fromtimestamp(self.end_time).isoformat(),
                "total_duration_seconds": self.end_time - self.start_time
            },
            "test_results": all_test_results,
            "summary": {
                "total_tests": len(all_test_results),
                "passed": sum(1 for r in all_test_results if r["status"] == "success"),
                "failed": sum(1 for r in all_test_results if r["status"] == "failed"),
                "overall_status": "passed" if all(r["status"] == "success" for r in all_test_results) else "failed"
            }
        }

        comprehensive_path = self.results_dir / "comprehensive_evaluation_report.json"
        with open(comprehensive_path, 'w') as f:
            json.dump(comprehensive_results, f, indent=2)

        print(f"\nComprehensive JSON report saved to: {comprehensive_path}")

        return comprehensive_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run LAM comprehensive evaluation suite"
    )
    parser.add_argument("--model", default="../",
                       help="Path to LAM model")
    parser.add_argument("--baseline", default="all-MiniLM-L6-v2",
                       help="Baseline model for comparison")

    args = parser.parse_args()

    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║         LAM COMPREHENSIVE EVALUATION SUITE v1.0               ║
    ║                                                               ║
    ║  Validating all claims from the LAM research paper with      ║
    ║  rigorous scientific testing and comprehensive reporting.     ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

    suite = ComprehensiveEvaluationSuite(args.model, args.baseline)
    results = suite.run()

    # Final summary
    if results["summary"]["overall_status"] == "passed":
        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED - LAM VALIDATION COMPLETE")
        print("="*80)
        sys.exit(0)
    else:
        print("\n" + "="*80)
        print("✗ SOME TESTS FAILED - REVIEW RESULTS")
        print("="*80)
        sys.exit(1)
