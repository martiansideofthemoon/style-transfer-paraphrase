import torch
from style_paraphrase.evaluation.similarity.sim_models import WordAveraging
from style_paraphrase.evaluation.similarity.sim_utils import Example
from nltk.tokenize import TreebankWordTokenizer
import sentencepiece as spm

tok = TreebankWordTokenizer()

model = torch.load('style_paraphrase/evaluation/similarity/sim/sim.pt')
state_dict = model['state_dict']
vocab_words = model['vocab_words']
args = model['args']
# turn off gpu
model = WordAveraging(args, vocab_words)
model.load_state_dict(state_dict, strict=True)
sp = spm.SentencePieceProcessor()
sp.Load('style_paraphrase/evaluation/similarity/sim/sim.sp.30k.model')
model.eval()

def make_example(sentence, model):
    sentence = sentence.lower()
    sentence = " ".join(tok.tokenize(sentence))
    sentence = sp.EncodeAsPieces(sentence)
    wp1 = Example(" ".join(sentence))
    wp1.populate_embeddings(model.vocab)
    return wp1

def find_similarity(s1, s2):
    with torch.no_grad():
        s1 = [make_example(x, model) for x in s1]
        s2 = [make_example(x, model) for x in s2]
        wx1, wl1, wm1 = model.torchify_batch(s1)
        wx2, wl2, wm2 = model.torchify_batch(s2)
        scores = model.scoring_function(wx1, wm1, wl1, wx2, wm2, wl2)
        return [x.item() for x in scores]

# s1 = "the dog ran outsideddd."
# s2 = "the puppy escape into the trees."
# print(find_similarity([s1, s2], [s2, s2]))
