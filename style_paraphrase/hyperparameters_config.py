paraphrase = [
    [('model_name',), ['gpt2-large']],
    [('dataset',), ["datasets/paranmt_filtered"]],
    [('batch_size',), [5]],
    [('accumulation',), [4]],
    [('num_epochs',), [2]],
    [('beam_size',), [1]],
    [('eval_batch_size',), [1]],
    [('learning_rate',), ["5e-5"]],
    [('gpu',), ["m40"]],
    [('ngpus',), ["1"]],
    [('prefix_input_type',), ["original"]],
    [('global_dense_feature_list',), ["none"]],
    [('stop_token',), ["eos"]],
    [('save_steps',), [500]],
    [('save_total_limit',), [-1]],
    [('specific_style_train',), [-1]],
    [('optimizer',), ["adam"]]
]

inverse_paraphrase = [
    [('model_name',), ['gpt2']],
    [('dataset',), ["datasets/formality"]],
    [('batch_size',), [5]],
    [('accumulation',), [2]],
    [('num_epochs',), [3]],
    [('beam_size',), [1]],
    [('eval_batch_size',), [1]],
    [('learning_rate',), ["5e-5"]],
    [('gpu',), ["m40"]],
    [('ngpus',), ["1"]],
    [('prefix_input_type',), ["paraphrase_250"]],
    [('global_dense_feature_list',), ["none"]],
    [('stop_token',), ["eos"]],
    [('specific_style_train',), [0, 1]],
    [('save_steps',), [500]],
    [('save_total_limit',), [-1]],
    [('optimizer',), ["adam"]]
]
