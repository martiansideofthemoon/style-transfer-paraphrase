import logging
import sys
import torch

from gpt2_finetune.inference_utils import GPT2Generator

if not torch.cuda.is_available():
    print("Please check if a GPU is available or your Pytorch installation is correct.")
    sys.exit()

print("Loading paraphraser...")
paraphraser = GPT2Generator(
    "/mnt/nfs/work1/miyyer/kalpesh/projects/style-embeddings/gpt2_finetune/saved_models/model_250"
)

print("\n\nNOTE: Ignore the weight mismatch error, this is due to different huggingface/transformer versions + minor modifications I did myself, shouldn't affect the paraphrases.\n\n")

input_sentence = input("Enter your sentence, q to quit: ")

while input_sentence != "q" and input_sentence != "quit" and input_sentence != "exit":
    paraphraser.modify_p(top_p=0.0)
    greedy_decoding = paraphraser.generate(input_sentence)
    print("\ngreedy sample:\n{}\n".format(greedy_decoding))
    paraphraser.modify_p(top_p=0.6)
    top_p_60_samples, _ = paraphraser.generate_batch([input_sentence, input_sentence, input_sentence])
    top_p_60_samples = "\n".join(top_p_60_samples)
    print("top_p = 0.6 samples:\n{}\n".format(top_p_60_samples))
    input_sentence = input("Enter your sentence, q to quit: ")

print("Exiting...")
