import argparse


def get_parser(parser_type, model_classes=None, all_models=None):
    parser = argparse.ArgumentParser()

    # Base parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input dataset directory.")

    parser.add_argument('--extra_embedding_dim', type=int, default=768,
                        help="Size of linear layer used for projecting extra embeddings.")

    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--limit_examples", type=int, default=None)
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--job_id", type=str, default="test")

    parser.add_argument("--target_style_override", type=str, default="none")
    parser.add_argument("--prefix_input_type", type=str, default="original",
                        help=("The text used as context for generation, which is 'original' for paraphrasing and a "
                              "paraphrase model ID for inverse paraphrasing."))

    parser.add_argument("--global_dense_feature_list", type=str, default="none")
    parser.add_argument("--specific_style_train", type=str, default="-1")

    if parser_type == "finetuning":
        parser.add_argument("--output_dir", default=None, type=str, required=True,
                            help="The output directory where the model predictions and checkpoints will be written.")
        parser.add_argument("--model_type", default="gpt2", type=str,
                            help="The model architecture to be fine-tuned.")
        parser.add_argument("--model_name_or_path", default="gpt2", type=str,
                            help="The model checkpoint for weights initialization.")

        parser.add_argument("--config_name", default="", type=str,
                            help="Optional pretrained config name or path if not the same as model_name_or_path")
        parser.add_argument("--tokenizer_name", default="", type=str,
                            help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
        parser.add_argument("--cache_dir", default="", type=str,
                            help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
        parser.add_argument("--do_train", action='store_true',
                            help="Whether to run training.")
        parser.add_argument("--do_eval", action='store_true',
                            help="Whether to run eval on the dev set.")
        parser.add_argument("--do_delete_old", action='store_true',
                            help="Whether to delete the older checkpoints based on perplexity.")
        parser.add_argument("--evaluate_during_training", action='store_true',
                            help="Run evaluation during training at each logging step.")

        parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                            help="Batch size per GPU/CPU for training.")
        parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                            help="Number of updates steps to accumulate before performing a backward/update pass.")
        parser.add_argument("--weight_decay", default=0.0, type=float,
                            help="Weight deay if we apply some.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                            help="Epsilon for Adam optimizer.")
        parser.add_argument("--max_grad_norm", default=1.0, type=float,
                            help="Max gradient norm.")
        parser.add_argument("--num_train_epochs", default=1.0, type=float,
                            help="Total number of training epochs to perform.")
        parser.add_argument("--max_steps", default=-1, type=int,
                            help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
        parser.add_argument("--warmup_steps", default=0, type=int,
                            help="Linear warmup over warmup_steps.")
        parser.add_argument("--optimizer", default="adam", type=str,
                            help="Type of optimizer to use.")

        parser.add_argument('--logging_steps', type=int, default=50,
                            help="Log every X updates steps.")
        parser.add_argument('--save_steps', type=int, default=50,
                            help="Save checkpoint every X updates steps.")
        parser.add_argument('--save_total_limit', type=int, default=None,
                            help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
        parser.add_argument('--overwrite_output_dir', action='store_true',
                            help="Overwrite the content of the output directory")

        parser.add_argument('--fp16', action='store_true',
                            help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
        parser.add_argument('--fp16_opt_level', type=str, default='O1',
                            help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                                 "See details at https://nvidia.github.io/apex/amp.html")
        parser.add_argument("--learning_rate", default="5e-5", type=str,
                            help="The initial learning rate for Adam.")
        parser.add_argument("--eval_frequency_min", type=int, default=0)
        parser.add_argument("--eval_patience", type=int, default=10)
        parser.add_argument("--evaluate_specific", type=str, default=None,
                            help="Specify path to certain checkpoint if you only want to evaluate that checkpoint")
    else:
        parser.add_argument("--model_type", default=None, type=str, required=True,
                            help="Model type selected in the list: " + ", ".join(model_classes.keys()))
        parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                            help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(all_models))
        parser.add_argument("--generation_output_dir", type=str, default="")
        parser.add_argument("--eval_split", type=str, default="dev")
        parser.add_argument("--length", type=int, default=-1)
        parser.add_argument("--num_samples", type=int, default=1)
        parser.add_argument("--temperature", type=float, default=1.0,
                            help="temperature of 0 implies greedy sampling")
        parser.add_argument("--top_k", type=int, default=0)
        parser.add_argument("--beam_size", type=int, default=1)
        parser.add_argument("--top_p", type=float, default=0.9)
        parser.add_argument("--beam_search_scoring", type=str, default="normalize")
        parser.add_argument('--stop_token', type=str, default=None,
                            help="Token at which text generation is stopped")
    return parser
