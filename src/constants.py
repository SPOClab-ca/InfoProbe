
spacy_model_names = {
    "en": "en_core_web_md",
    "fr": "fr_core_news_md",
    "es": "es_core_news_md"
}

# 17 for English.
# 15 for French: replace "INTJ" (7 entries) or "SYM" with "X" (1296 entries).
# 16 for Spanish: Change "X" (1 entry) to "INTJ" (27 entries) 
spacy_pos_dict = {
    "en": ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X'],
    "fr": ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'VERB', 'X', ''],
    "es": ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB']
}

gensim_fasttext_models = {
    "en": "../data/embeddings/fasttext/cc.en.300.bin",
    "fr": "../data/embeddings/fasttext/cc.fr.300.bin",
    "es": "../data/embeddings/fasttext/cc.es.300.bin"
}