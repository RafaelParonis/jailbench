#!/usr/bin/env python3

import argparse
import json
import sys

from src.benchmark import JailbreakBenchmark


def main():
    """
    Main entry point for the AI Jailbreaking Benchmark Tool.

    This function parses command-line arguments, initializes the JailbreakBenchmark,
    and executes the appropriate mode (standard benchmark, adversarial, or interactive)
    based on user input.

    Variables:
        parser (argparse.ArgumentParser): Argument parser for CLI options.
        args (argparse.Namespace): Parsed command-line arguments.
        benchmark (JailbreakBenchmark): Instance of the benchmark tool.
    """
    parser = argparse.ArgumentParser(description="AI Jailbreaking Benchmark Tool")
    parser.add_argument(
        "--credentials",
        default="credentials.json",
        help="Path to credentials JSON file"
    )
    parser.add_argument(
        "--tests",
        default="jailbreak_tests.json",
        help="Path to jailbreak tests JSON file"
    )
    parser.add_argument(
        "--output",
        default="benchmark_results.json",
        help="Path to save benchmark results"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show summary statistics after running benchmark"
    )
    parser.add_argument(
        "--adversarial",
        action="store_true",
        help="Run adversarial mode (model vs model)"
    )
    parser.add_argument(
        "--objective",
        help="Objective for adversarial mode (what attacker tries to achieve)"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum iterations for adversarial mode (default: 5)"
    )
    parser.add_argument(
        "--target-prompt",
        help="Custom system prompt for target model in adversarial mode"
    )
    parser.add_argument(
        "--attacker-prompt",
        help="Custom system prompt for attacker model in adversarial mode"
    )
    parser.add_argument(
        "--analytics-config",
        help="Path to advanced analytics configuration file"
    )
    parser.add_argument(
        "--disable-analytics",
        action="store_true",
        help="Disable all advanced analytics (faster execution)"
    )
    parser.add_argument(
        "--enable-tool-calling-tests",
        action="store_true",
        help="Enable tool calling jailbreak tests (requires explicit opt-in for security)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive testing mode for manual jailbreak analysis"
    )

    args = parser.parse_args()

    try:
        benchmark = JailbreakBenchmark(
            credentials_path=args.credentials,
            analytics_config_path=args.analytics_config
        )

        if args.disable_analytics:
            benchmark.analytics_config.semantic_analysis.enabled = False
            benchmark.analytics_config.defense_profiling.enabled = False
            benchmark.analytics_config.vulnerability_matrix.enabled = False
            benchmark.analytics_config.attack_vectors.enabled = False
            benchmark.analytics_config.conversation_flow.enabled = False
            benchmark.analytics_config.response_patterns.enabled = False
            benchmark.analytics_config.prompt_optimization.enabled = False

        if args.enable_tool_calling_tests:
            benchmark.analytics_config.tool_calling.enabled = True
            benchmark.analytics_config.tool_calling.malicious_tool_calls = True
            benchmark.analytics_config.tool_calling.tool_chaining = True
            benchmark.analytics_config.tool_calling.permission_escalation = True
            benchmark.analytics_config.tool_calling.data_extraction = True

        if not benchmark.clients:
            print("No enabled models found in credentials file")
            sys.exit(1)

        if args.interactive:
            """
            Interactive Testing Mode

            Launches a Textual TUI for manual jailbreak analysis.

            Variables:
                JailbreakTUI (type): TUI class for interactive mode.
                app (JailbreakTUI): Instance of the TUI application.
            """
            from src.textual_tui import JailbreakTUI

            print("Initializing Interactive Testing Mode...")
            print(f"Loaded {len(benchmark.clients)} enabled models")
            for i, client in enumerate(benchmark.clients):
                print(f"   {i+1}. {client.provider}/{client.model}")
            print(f"Starting with: {benchmark.clients[0].provider}/{benchmark.clients[0].model}")
            print("Use Ctrl+M or /model command to switch between models\n")
            print("Loading Textual TUI...")

            app = JailbreakTUI(benchmark.clients)
            app.run()
            return

        if args.adversarial:
            """
            Adversarial Mode

            Runs adversarial testing where one model attempts to jailbreak another.

            Variables:
                target_client: The target model client.
                attacker_client: The attacker model client.
                result: The result object from adversarial testing.
            """
            if not args.objective:
                print("--objective is required for adversarial mode")
                sys.exit(1)

            print("Initializing Adversarial AI Testing...")
            print(f"Loaded {len(benchmark.clients)} enabled models")

            if len(benchmark.clients) == 1:
                print("Using same model as both target and attacker")
                target_client = attacker_client = benchmark.clients[0]
            else:
                target_client = benchmark.clients[0]
                attacker_client = benchmark.clients[1]

            print(f"Target Model: {target_client.provider}/{target_client.model}")
            print(f"Attacker Model: {attacker_client.provider}/{attacker_client.model}")
            print(f"Objective: {args.objective}")
            print(f"Max Iterations: {args.max_iterations}")

            print(f"\nStarting adversarial test...")
            result = benchmark.run_adversarial_test(
                target_client=target_client,
                attacker_client=attacker_client,
                objective=args.objective,
                max_iterations=args.max_iterations,
                target_system_prompt=args.target_prompt,
                attacker_system_prompt=args.attacker_prompt
            )

            print(f"\n=== ADVERSARIAL TEST RESULTS ===")
            print(f"Success: {'YES' if result.success else 'NO'}")
            print(f"Iterations Used: {result.final_iteration}/{result.max_iterations}")
            print(f"Total Time: {result.total_time:.2f}s")

            print(f"\n=== CONVERSATION HISTORY ===")
            for i, msg in enumerate(result.conversation_history):
                role = "ATTACKER" if msg["role"] == "user" else "TARGET"
                print(f"{i+1}. {role}: {msg['content'][:100]}...")
                print()

            if result.success:
                print(f"=== SUCCESS RESPONSE ===")
                print(result.success_response)

            if result.evaluator_reasoning:
                print(f"\n=== EVALUATOR REASONING ===")
                print(result.evaluator_reasoning)

            print(f"\nDetailed results saved to {benchmark.output_dir}")

        else:
            """
            Standard Benchmark Mode

            Runs the standard set of jailbreak tests and outputs results.

            Variables:
                run_analytics (bool): Whether to run advanced analytics.
                results: Benchmark results.
                results_path: Path to save results.
                summary: Summary statistics.
            """
            print("Initializing AI Jailbreaking Benchmark...")
            print(f"Loaded {len(benchmark.clients)} enabled models")
            for client in benchmark.clients:
                print(f"   - {client.provider}/{client.model}")

            print(f"\nRunning benchmark tests...")
            run_analytics = not args.disable_analytics
            results = benchmark.run_benchmark(args.tests, run_advanced_analytics=run_analytics)

            results_path = benchmark.output_dir / args.output.split('/')[-1]
            print(f"\nExporting results to {results_path}...")
            benchmark.export_results(str(results_path))

            if args.summary:
                print("\nBenchmark Summary:")
                summary = benchmark.generate_summary()
                print(f"   Total tests run: {summary['total_tests_run']}")
                print(f"   Models tested: {summary['models_tested']}")
                print(f"   Overall jailbreak rate: {summary['overall_jailbreak_rate']:.2%}")

                print("\nPer-Model Results:")
                for model, stats in summary['by_model'].items():
                    print(f"   {model}:")
                    print(f"     Jailbreak rate: {stats['jailbreak_rate']:.2%}")
                    print(f"     Avg response time: {stats['avg_response_time']:.2f}s")
                    print(f"     Avg confidence: {stats['avg_confidence']:.2f}")

            print(f"\nBenchmark complete! Results saved to {results_path}")

    except FileNotFoundError as e:
        """
        Exception Handler

        Handles missing file errors.

        Variables:
            e (FileNotFoundError): The exception instance.
        """
        print(f"File not found: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        """
        Exception Handler

        Handles invalid JSON format errors.

        Variables:
            e (json.JSONDecodeError): The exception instance.
        """
        print(f"Invalid JSON format: {e}")
        sys.exit(1)
    except Exception as e:
        """
        Exception Handler

        Handles all other exceptions.

        Variables:
            e (Exception): The exception instance.
        """
        print(f"Error running benchmark: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()