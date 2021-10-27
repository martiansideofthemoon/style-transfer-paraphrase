from collections import defaultdict

BATCH_SIZE = 128
MAX_LENGTH = 502
MAX_GPT2_LENGTH = 1000

MAX_PARAPHRASE_LENGTH = 100

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

DATASET_CONFIG = {
"datasets/test_custom": BASE_CONFIG,
"datasets/paranmt_filtered": BASE_CONFIG,
    "datasets/shakespeare": BASE_CONFIG,
    "datasets/formality": BASE_CONFIG,
    "datasets/shakespeare_aae_tweets_bible_romantic-poetry_switchboard_coha_3_bins_lyrics_full": BASE_CONFIG,
    "datasets/aae": BASE_CONFIG,
    "datasets/bible": BASE_CONFIG,
    "datasets/poetry": BASE_CONFIG,
    "datasets/switchboard": BASE_CONFIG,
    "datasets/english_tweets": BASE_CONFIG,
    "datasets/lyrics_full": BASE_CONFIG,
    "datasets/joyce": BASE_CONFIG,
    "datasets/congress-bills": BASE_CONFIG,
    "datasets/shakespeare": BASE_CONFIG,
    "datasets/coha_3_bins_1810s-1820s": BASE_CONFIG,
    "datasets/coha_3_bins_1890s-1900s": BASE_CONFIG,
    "datasets/coha_3_bins_1990s-2000s": BASE_CONFIG
}

# Fill in DATASET_CONFIG with keys it was missing previously
for dataset, config in DATASET_CONFIG.items():
    for base_key, base_value in BASE_CONFIG.items():
        if base_key not in config:
            config[base_key] = base_value
