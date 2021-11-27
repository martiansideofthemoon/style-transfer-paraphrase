import pickle
import torch

import numpy as np

from transformers import GPT2LMHeadModel, GPT2Tokenizer

from style_paraphrase.dataset_config import DATASET_CONFIG, BASE_CONFIG
from style_paraphrase.data_utils import update_config, Instance, get_label_dict

from style_paraphrase.utils import init_gpt2_model


class GPT2Generator(object):
    def __init__(self, model_path, upper_length="same_10", beam_size=1, top_p=0.0, data_dir=None):
        self.model_path = model_path
        self.args = torch.load("{}/training_args.bin".format(self.model_path))
        self.modify_args(upper_length, beam_size, top_p)
        self.config = BASE_CONFIG
        update_config(self.args, self.config)

        if self.args.global_dense_feature_list != "none":

            self.label_dict, self.reverse_label_dict = get_label_dict(data_dir)

            self.global_dense_features = []
            for gdf in self.args.global_dense_feature_list.split(","):
                with open(
                    "{}/{}_dense_vectors.pickle".format(data_dir, gdf), "rb"
                ) as f:
                    vector_data = pickle.load(f)

                final_vectors = {}
                for k, v in vector_data.items():
                    final_vectors[self.label_dict[k]] = v["sum"] / v["total"]

                self.global_dense_features.append((gdf, final_vectors))

        self.gpt2_model, self.tokenizer = init_gpt2_model(checkpoint_dir=model_path,
                                                          args=self.args,
                                                          model_class=GPT2LMHeadModel,
                                                          tokenizer_class=GPT2Tokenizer)

    def modify_args(self, upper_length, beam_size, top_p):
        args = self.args
        args.upper_length = upper_length
        args.stop_token = "eos" if upper_length == "eos" else None
        args.beam_size = beam_size
        args.num_samples = 1
        args.temperature = 0
        args.top_p = top_p
        args.top_k = 1
        args.device = torch.cuda.current_device()

    def modify_p(self, top_p):
        self.args.top_p = top_p

    def generate_batch(self, contexts, global_dense_features=None, get_scores=False,
                       interpolation=None, top_p=None):
        args = self.args
        tokenizer = self.tokenizer
        instances = []

        if global_dense_features is None:
            global_dense_features = [None for _ in contexts]

        for context, gdf in zip(contexts, global_dense_features):
            context_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(context))

            # NOTE - For model_110, use the older version of the code
            # The following code is only compatible with the newer models
            instance = Instance(
                self.args, self.config,
                {"sent1_tokens": context_ids, "sent2_tokens": context_ids}
            )
            instance.preprocess(tokenizer)

            if gdf is not None and self.args.global_dense_feature_list != "none":
                if self.global_dense_features:
                    global_dense_vectors = np.array(
                        [x[1][gdf] for x in self.global_dense_features],
                        dtype=np.float32,
                    )
                else:
                    global_dense_vectors = np.zeros((2, 20), dtype=np.float32)
                    global_dense_vectors[0, gdf["f1_bucket"]] = 1
                    global_dense_vectors[1, gdf["ed_bucket"] + 10] = 1
            else:
                global_dense_vectors = np.zeros((1, 768), dtype=np.float32)

            instance.gdv = global_dense_vectors
            instances.append(instance)

        output, _, scores = self.gpt2_model.generate(
            gpt2_sentences=torch.tensor([inst.sentence for inst in instances]).to(args.device),
            segments=torch.tensor([inst.segment for inst in instances]).to(args.device),
            global_dense_vectors=torch.tensor([inst.gdv for inst in instances]).to(args.device),
            init_context_size=instances[0].init_context_size,
            eos_token_id=tokenizer.eos_token_id,
            get_scores=get_scores,
            interpolation=interpolation,
            top_p=top_p
        )

        all_output = []
        for out_num in range(len(output)):
            instance = instances[out_num]
            curr_out = output[out_num, instance.init_context_size:].tolist()

            if tokenizer.eos_token_id in curr_out:
                curr_out = curr_out[:curr_out.index(tokenizer.eos_token_id)]

            if self.args.upper_length.startswith("same"):
                extra = int(self.args.upper_length.split("_")[-1])
                curr_out = curr_out[:len(instance.sent1_tokens) + extra]

            all_output.append(
                tokenizer.decode(curr_out, clean_up_tokenization_spaces=True, skip_special_tokens=True)
            )

        return all_output, scores

    def generate(self, context, global_dense_features=None, get_scores=False,
                 interpolation=None, top_p=None):
        return self.generate_batch([context],
                                   [global_dense_features] if global_dense_features is not None else None,
                                   get_scores=get_scores,
                                   interpolation=interpolation,
                                   top_p=top_p)[0][0]
