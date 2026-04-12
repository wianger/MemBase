import argparse
from membase import (
    MEMORY_LAYERS_MAPPING,
    DATASET_MAPPING,
    ConstructionRunner,
    ConstructionRunnerConfig,
)
from membase.utils import import_function_from_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script used to evaluate various memory layers on various datasets."
    )
    parser.add_argument(
        "--memory-type", 
        choices=list(MEMORY_LAYERS_MAPPING.keys()), 
        type=str, 
        required=True, 
        help="The type of the memory layer to be evaluated."
    )
    parser.add_argument(
        "--dataset-type", 
        choices=list(DATASET_MAPPING.keys()), 
        type=str, 
        required=True, 
        help="The type of the dataset used to evaluate the memory layer."
    )
    parser.add_argument(
        "--dataset-path", 
        type=str, 
        required=True, 
        help="The path to the dataset."
    )
    parser.add_argument(
        "--dataset-standardized",
        action="store_true",
        help="Whether the dataset is already standardized."
    )
    parser.add_argument(
        "--num-workers", 
        type=int, 
        default=4, 
        help="The number of threads to use for the evaluation."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed used to sample the dataset if the user provides the sample size."
    )
    parser.add_argument(
        "--sample-size", 
        type=int, 
        default=None, 
        help="Subset size from dataset."
    )
    parser.add_argument(
        "--rerun", 
        action="store_true", 
        help="Ignore saved memory and rebuild the memory from scratch."
    )
    parser.add_argument(
        "--config-path", 
        type=str, 
        default=None,
        help="Path to the JSON config for the memory method."
    )
    parser.add_argument(
        "--token-cost-save-filename", 
        type=str, 
        default="token_cost", 
        help="Path to save the statistics related to the token consumption."
    )
    parser.add_argument(
        "--start-idx", 
        type=int, 
        default=None, 
        help="The starting index of the trajectories to be processed."
    )
    parser.add_argument(
        "--end-idx", 
        type=int, 
        default=None, 
        help="The ending index of the trajectories to be processed."
    )
    parser.add_argument(
        "--tokenizer-path", 
        type=str, 
        default=None, 
        help="The path to the tokenizer (only for backbone model)."
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help=(
            "Disable strict mode. When it is set, errors during the memory construction "
            "are logged and the message is skipped instead of aborting the trajectory."
        ),
    )
    parser.add_argument(
        "--message-preprocessor-path", 
        type=str, 
        default=None, 
        help=(
            "Path to a callable that preprocesses each message before it is added to the memory. "
            "It supports two formats: (1) Python module path, e.g. 'mypackage.module.func'; "
            "(2) file path with function name, e.g. 'path/to/file.py:func'."
        ),
    )
    parser.add_argument(
        "--sample-filter-path", 
        type=str, 
        default=None, 
        help=(
            "Path to a callable that filters dataset samples. "
            "It supports two formats: (1) Python module path, e.g. 'mypackage.module.func'; "
            "(2) file path with function name, e.g. 'path/to/file.py:func'."
        ),
    )
    parser.add_argument(
        "--online-eval-config-path",
        type=str,
        default=None,
        help="Path to a JSON config for the online evaluation environment.",
    )

    parser.add_argument(
        "--max-sessions",
        type=int,
        default=None,
        help="If set, only the first N sessions in the dataset will be used (for single-trajectory datasets)."
    )

    args = parser.parse_args()

    message_preprocessor = None
    if args.message_preprocessor_path is not None:
        message_preprocessor = import_function_from_path(args.message_preprocessor_path)
        print(f"A message preprocessor is loaded from '{args.message_preprocessor_path}'.")

    sample_filter = None
    if args.sample_filter_path is not None:
        sample_filter = import_function_from_path(args.sample_filter_path)
        print(f"A sample filter is loaded from '{args.sample_filter_path}'.")


    # --- max-sessions logic ---
    dataset_path = args.dataset_path
    if args.max_sessions is not None:
        import json
        from pathlib import Path
        path = Path(dataset_path)
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        # Only support single-trajectory datasets (like RealMem)
        if 'dialogues' in data:
            orig_n = len(data['dialogues'])
            data['dialogues'] = data['dialogues'][:args.max_sessions]
            tmp_path = path.parent / (path.stem + f"_first{args.max_sessions}_sessions.json")
            with tmp_path.open('w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"[memory_construction.py] Truncated dataset to first {args.max_sessions} sessions: {tmp_path} (original {orig_n})")
            dataset_path = str(tmp_path)
        else:
            print("[memory_construction.py] --max-sessions only supports RealMem-style single-trajectory datasets.")

    runner_config = ConstructionRunnerConfig(
        memory_type=args.memory_type,
        dataset_type=args.dataset_type,
        dataset_path=dataset_path,
        dataset_standardized=args.dataset_standardized,
        config_path=args.config_path,
        num_workers=args.num_workers,
        seed=args.seed,
        sample_size=args.sample_size,
        rerun=args.rerun,
        strict=not args.no_strict,
        token_cost_save_filename=args.token_cost_save_filename,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        tokenizer_path=args.tokenizer_path,
        message_preprocessor=message_preprocessor,
        sample_filter=sample_filter,
        online_eval_config_path=args.online_eval_config_path,
    )
    ConstructionRunner(runner_config).run()
