import os
import pickle
import re
import spacy
import numpy as np

from preprocess.srl_utils import allennlp_srl

directory = "/mnt/nfs/work1/miyyer/datasets/harry-potter-fanfiction/Fanfic_Harry Potter/Harry Potter/Completed"

bad_starters = ["**", "_**", "Note:", "by", "- ^-^3", "End file.", "\t", "A HariPo ", "* * *", "> ", ">I", "_fin._",
                "A/N:", "Anne M.", "CP09", "_By ChristinaPotter09_", "_Disclaimer", "Authors Note:", '_A/N:',
                '_-Wizards_Pupil_', 'A/N **Acknowledgements**.', '()']


def get_template_two(doc):
    return doc.label() + " > " + " ".join([x.label() for x in doc])


def get_template_three(doc):
    template = doc.label() + " >> "
    for l2 in doc:
        l3_str = " ".join([l3.label() for l3 in l2 if hasattr(l3, "label")])
        if len(l3_str) > 0:
            template += l2.label() + " > " + l3_str + " | "
        else:
            template += l2.label() + " | "

    return template[:-3]


class SpecialBPE:
    def __init__(self, roberta, new_tokens_map):
        self.roberta_bpe = roberta.bpe
        self.new_tokens_map = new_tokens_map

    def encode(self, sentence):
        encoded_sentence = " " + self.roberta_bpe.encode(sentence) + " "
        for ntm in self.new_tokens_map:
            encoded_sentence = encoded_sentence.replace(ntm["bpe_no_space"] + " ", ntm["token"] + " ")
            encoded_sentence = encoded_sentence.replace(ntm["bpe_with_space"] + " ", ntm["token"] + " ")
        return encoded_sentence.strip()

    def decode(self, sentence):
        for ntm in self.new_tokens_map:
            sentence = sentence.replace(" " + ntm["token"], " " + ntm["bpe_with_space"])
            sentence = sentence.replace(ntm["token"], ntm["bpe_no_space"])
        return self.roberta_bpe.decode(sentence)


def build_new_token_map(new_tokens, roberta):
    new_tokens_map = []
    for token in new_tokens:
        new_tokens_map.append({
            "token": token,
            "bpe_no_space": roberta.bpe.encode(token),
            "bpe_with_space": roberta.bpe.encode(" " + token)
        })
    return new_tokens_map


def sentence_srl_pipeline(args, files, author_name, author_index, base_folder, split):
    # pipeline stage #1, filter the story and concatenate into single string object
    folder = "%s/cache_%d" % (base_folder, author_index)
    os.makedirs(folder, exist_ok=True)

    if os.path.exists("%s/%s.pkl" % (folder, split)):
        with open("%s/%s.pkl" % (folder, split), "rb") as f:
            sentence_data, srl_data = pickle.load(f)
    else:
        all_data = ""
        for file in files:
            with open(directory + "/" + file, "r") as f:
                orig_story = f.read().strip()
                data, orig = filter_story(orig_story, author_name)
                # possible due to occasional non-english story
                if len(data.strip()) == 0:
                    continue

                if len(data) > 0:
                    all_data += data + "\n"
        # pipeline stage #2, run spacy sentencizer
        sentence_data, total_sents = spacy_sentencizer(all_data, author_name)
        # pipeline stage #3, run AllenNLP SRL model
        srl_data = allennlp_srl(args, sentence_data, author_name, total_sents=total_sents)
        # cache the sentence, SRL data so that the bottleneck need not be re-computed
        with open("%s/%s.pkl" % (folder, split), "wb") as f:
            pickle.dump([sentence_data, srl_data], f)

    output = {
        "srl_data": srl_data,
        "sentence_data": sentence_data
    }
    return output


def is_bad_start(line):
    bad_start = False
    for bs in bad_starters:
        if line.startswith(bs):
            bad_start = True
            break
    return bad_start


def filter_story(story, author_name):
    # strip HTML
    if "Language: English" not in story:
        return "", story
    story = re.sub('<[^<]+?>', ' ', story)
    lines = story.split("\n")
    content_only = []
    # Remove the header from the story
    for i, line in enumerate(lines):
        if line.startswith("Summary:"):
            content_only = lines[i + 1:]
            break
    # Remove blank lines and comments
    stripped_content = []
    for i, line in enumerate(content_only):
        if len(line.strip()) == 0:
            continue
        if is_bad_start(line):
            continue
        stripped_content.append(line)

    # finally, remove all exact string matches with author name
    stripped_content = " ".join(stripped_content)
    stripped_content = stripped_content.replace(author_name, " ")
    stripped_content = " ".join(stripped_content.split())

    return stripped_content, story


def spacy_sentencizer(dataset, author_name):
    nlp = spacy.load("en_core_web_lg", disable=["tagger", "parser", "ner"])
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    stories = dataset.strip().split("\n")

    docs = list(nlp.pipe(stories))

    sent_lens = [len(sent.text.split()) for story in docs for sent in story.sents]
    long_sents = sum([1 for sl in sent_lens if sl >= 200])

    print(
        "Author %s = %d stories, %d sentences, %.4f avg sentence length, %d max sentence length, %d long sentence(s) to be discarded" %
        (author_name, len(docs), len(sent_lens), np.mean(sent_lens), np.max(sent_lens), long_sents)
    )

    sentence_data = [[sent.text for sent in doc.sents] for doc in docs]
    return sentence_data, len(sent_lens)
