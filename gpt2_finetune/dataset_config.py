from collections import defaultdict

BATCH_SIZE = 128
MAX_LENGTH = 502
MAX_GPT2_LENGTH = 1000

MAX_PARAPHRASE_LENGTH = 100
MAX_SIMPLEWIKI_LENGTH = 200

BASE_CONFIG = {
    "keys": [
        {"key": "sent1_tokens", "position": 3, "tokenize": True, "metadata": False},
        {"key": "sent2_tokens", "position": 4, "tokenize": True, "metadata": False},
        {"key": "f1_score", "position": 5, "tokenize": False, "metadata": True},
        {"key": "kt_score", "position": 6, "tokenize": False, "metadata": True},
        {"key": "ed_score", "position": 7, "tokenize": False, "metadata": True},
        {"key": "langid", "position": 8, "tokenize": False, "metadata": True},
    ],
    "max_total_length": MAX_PARAPHRASE_LENGTH,
    "max_prefix_length": int(MAX_PARAPHRASE_LENGTH / 2),
    "max_suffix_length": int(MAX_PARAPHRASE_LENGTH / 2),
    "max_dense_length": 2
}

BASE_SIMPLEWIKI_CONFIG = {
    "keys": [
        {"key": "sent1_tokens", "position": 0, "tokenize": True, "metadata": False},
        {"key": "sent2_tokens", "position": 1, "tokenize": True, "metadata": False},
    ],
    "max_total_length": MAX_SIMPLEWIKI_LENGTH,
    "max_prefix_length": int(MAX_SIMPLEWIKI_LENGTH / 2),
    "max_suffix_length": int(MAX_SIMPLEWIKI_LENGTH / 2),
    "max_dense_length": 0
}

DATASET_CONFIG = {
    "paranmt": {
        "keys": [
            {"key": "sent1_tokens", "position": 0, "tokenize": True, "metadata": False},
            {"key": "sent2_tokens", "position": 1, "tokenize": True, "metadata": False},
            {"key": "f1_score", "position": 4, "tokenize": False, "metadata": True},
            {"key": "ed_score", "position": 5, "tokenize": False, "metadata": True},
            {"key": "f1_bucket", "position": 6, "tokenize": False, "metadata": True},
            {"key": "ed_bucket", "position": 7, "tokenize": False, "metadata": True}
        ]
    },
    "paranmt/filter_lendiff_less_3_english_only": BASE_CONFIG,
    "paranmt/filter_lendiff_less_5_english_only": BASE_CONFIG,
    "paranmt/filter_lendiff_less_5_english_only_truncate_300000": BASE_CONFIG,
    "paranmt/no_trigram_filter/filter_lendiff_less_5_english_only_truncate_74554": BASE_CONFIG,
    "paranmt/filter_kt_less_0": BASE_CONFIG,
    "paranmt/filter_kt_less_0.5": BASE_CONFIG,
    "paranmt/filter_and_kt_less_precision_less_0.0_0.25": BASE_CONFIG,
    "paranmt/filter_and_kt_less_precision_less_0.0_0.5": BASE_CONFIG,
    "paranmt/filter_and_kt_less_precision_less_0.0_0.25_no_czech": BASE_CONFIG,
    "paranmt/filter_and_kt_less_precision_less_0.0_0.5_no_czech": BASE_CONFIG,
    "paranmt/filter_and_kt_less_precision_less_lendiff_less_0.0_0.5_5_english_only": BASE_CONFIG,
    "newsela/newsela_sentences": {
        "keys": [
            {"key": "sent1_tokens", "position": 0, "tokenize": True, "metadata": False},
            {"key": "sent2_tokens", "position": 1, "tokenize": True, "metadata": False},
            {"key": "f1_score", "position": 2, "tokenize": False, "metadata": True},
            {"key": "ed_score", "position": 3, "tokenize": False, "metadata": True},
            {"key": "doc_id", "position": 4, "tokenize": False, "metadata": True},
        ]
    },
    "shakespeare/supervised": {
        "keys": [
            {"key": "sent1_tokens", "position": 0, "tokenize": True, "metadata": False},
            {"key": "sent2_tokens", "position": 1, "tokenize": True, "metadata": False},
            {"key": "f1_score", "position": 2, "tokenize": False, "metadata": True},
            {"key": "ed_score", "position": 3, "tokenize": False, "metadata": True}
        ]
    },
    "shakespeare/supervised_filtered": {
        "keys": [
            {"key": "sent1_tokens", "position": 0, "tokenize": True, "metadata": False},
            {"key": "sent2_tokens", "position": 1, "tokenize": True, "metadata": False},
            {"key": "f1_score", "position": 2, "tokenize": False, "metadata": True},
            {"key": "ed_score", "position": 3, "tokenize": False, "metadata": True}
        ]
    },
    # Simplewiki configurations
    "simplewiki": BASE_SIMPLEWIKI_CONFIG,
    "wikilarge": BASE_SIMPLEWIKI_CONFIG,
    # HP Fan-Fiction configurations
    "author_data/authors_1M_tokens_37_classes_srl_arg0_arg1_single_sentence": {
        "max_dense_length": 4
    },
    # Shakespeare unsupervised data
    "shakespeare/unsupervised_filtered": BASE_CONFIG,
    "shakespeare/unsupervised_prior": BASE_CONFIG,
    "shakespeare/unsupervised_prior_detokenize": BASE_CONFIG,
    # Shakespeare + AAE + english tweets
    "dataset_pools/shakespeare_aae_tweets": BASE_CONFIG,
    "dataset_pools/shakespeare_aae_tweets_bible_romantic-poetry_joyce_congress-bills": BASE_CONFIG,
    "dataset_pools/shakespeare_aae_tweets_bible_romantic-poetry_switchboard": BASE_CONFIG,
    "politeness/politeness": BASE_CONFIG,
    "formality/formality": BASE_CONFIG,
    "formality/formality_prior": BASE_CONFIG,
    "formality/formality_prior2_lowercase": BASE_CONFIG,
    "formality/formality_prior_detokenize": BASE_CONFIG,
    "gender/gender": BASE_CONFIG,
    "political-slant/political-slant": BASE_CONFIG,
    "dataset_pools/shakespeare_aae_tweets_bible_romantic-poetry_switchboard_coha_3_bins_lyrics_full": BASE_CONFIG,
    "dataset_pools/aae": BASE_CONFIG,
    "dataset_pools/bible": BASE_CONFIG,
    "dataset_pools/romantic-poetry": BASE_CONFIG,
    "dataset_pools/switchboard": BASE_CONFIG,
    "dataset_pools/english_tweets": BASE_CONFIG,
    "dataset_pools/lyrics_full": BASE_CONFIG,
    "dataset_pools/joyce": BASE_CONFIG,
    "dataset_pools/congress-bills": BASE_CONFIG,
    "dataset_pools/shakespeare": BASE_CONFIG,
    "dataset_pools/coha_3_bins_1810s-1820s": BASE_CONFIG,
    "dataset_pools/coha_3_bins_1890s-1900s": BASE_CONFIG,
    "dataset_pools/coha_3_bins_1990s-2000s": BASE_CONFIG
}

BASE_HP_CONFIG = {
    "max_total_length": MAX_GPT2_LENGTH,
    "max_prefix_length": int(MAX_GPT2_LENGTH / 2),
    "max_suffix_length": int(MAX_GPT2_LENGTH / 2),
    "max_dense_length": 4
}

# Fill in DATASET_CONFIG with keys it was missing previously
for dataset, config in DATASET_CONFIG.items():
    for base_key, base_value in BASE_CONFIG.items():
        if base_key not in config:
            config[base_key] = base_value
