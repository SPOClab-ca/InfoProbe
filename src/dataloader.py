import flair 
import torch
import numpy as np 
import spacy
import transformers
import os, sys, time
from tqdm import tqdm
import psutil
import pickle
import gensim

from constants import spacy_pos_dict, spacy_model_names, gensim_fasttext_models

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def print_memory_usage(label):
    process = psutil.Process(os.getpid())
    #print(process.memory_info())
    mem = process.memory_info().rss / 1024 / 1024
    print ("{} using {:.2f} MB memory!".format(label, mem))

class DataLoader():
    def __init__(self, split, args):
        self.args = args
        self.pos_tags = spacy_pos_dict[args.lang]
        if args.lm == "bertmulti":
            self.emb_dim = 768
        elif args.lm == "fasttext":
            self.emb_dim = 100
        else:
            self.emb_dim = -1
        cache_dataset_head = "cache/{}_{}_{}_{}".format(args.lang, args.lm, split, args.task)
        if os.path.exists(cache_dataset_head):
            pass
            #print ("{} exists. Loading the datasets.".format(cache_dataset_head))
        else:
            print ("Generating dataset in {}".format(cache_dataset_head))
            os.makedirs(cache_dataset_head)
            self._generate_data_pairs(split, cache_dataset_head, args)
        filelist_short = os.listdir(cache_dataset_head)
        self.ckpt_filelist = [os.path.join(cache_dataset_head, fn_short) for fn_short in filelist_short]
        self.ckpt_ptr = 0
        # self.x and self.y are loaded from the checkpoint in ckpt_filelist[self.ckpt_ptr]
        self.x = []
        self.y = [] 
        self.ptr = 0  # iterates through self.x and self.y
        self.reset()

        if split == "train":
            self.batch_size = args.batch_size
        else:
            self.batch_size = 1

    def _generate_data_pairs(self, split, cache_dataset_head, args):
        """
        Return CPU torch tensors
        """
        # Corpus
        if args.lang == "en": 
            corpus = flair.datasets.UD_ENGLISH()
        elif args.lang == "fr":
            corpus = flair.datasets.UD_FRENCH()
        elif args.lang == "es":
            corpus = flair.datasets.UD_SPANISH()
        else:
            raise ValueError
        if split == "train":
            corpus_split = corpus.train 
        elif split == "dev":
            corpus_split = corpus.dev 
        elif split == "test":
            corpus_split = corpus.test 
        else:
            raise ValueError("split {} not accepted!".format(split))

        # SpaCy tagger
        spacy_nlp = spacy.load(spacy_model_names[args.lang])

        # Word Embedding (huggingface)
        if args.lm == "bertmulti":
            emb_tag = "bert-base-multilingual-cased"
            tokenizer = transformers.BertTokenizer.from_pretrained(emb_tag)
            emb = transformers.BertModel.from_pretrained(emb_tag)
            self.emb_dim = emb.config.hidden_size
        elif args.lm == "fasttext":
            tokenizer = BaselineTokenizer()
            emb = gensim.models.FastText(gensim_fasttext_models[args.lang])
            self.emb_dim = 100
        elif args.lm == "glove":
            raise NotImplementedError
        else:
            raise ValueError("lm {} not supported".format(args.lm))

        start_time = time.time()
        dumpcnt = 1
        all_x = []
        all_y = []
        for doc_id, article in enumerate(corpus_split):
            raw_s = article.to_plain_string()
            sent_x, sent_y = self._process_sentence(raw_s, spacy_nlp, tokenizer, emb, args)
            all_x += sent_x
            all_y += sent_y
            if doc_id>0 and doc_id % 1024 == 0:
                self._process_dump(all_x, all_y, cache_dataset_head, dumpcnt, args.task)
                all_x = []
                all_y = []
                dumpcnt += 1
                print ("Processed {} docs in {:.2f} seconds. Cacheing...".format(doc_id, time.time() - start_time))
                print_memory_usage(f"docid {doc_id}")
        self._process_dump(all_x, all_y, cache_dataset_head, dumpcnt, args.task)
        print ("Finished processing and cached {} docs in {:.2f} seconds.".format(len(corpus_split), time.time() - start_time))

    def _process_dump(self, all_x, all_y, cache_dataset_head, dumpcnt, task):
        all_x_tensors = torch.cat(all_x, dim=0).to(device)  # (N, D)
        all_y_tensors = torch.cat(all_y).to(device)  # (N,)

        if args.task == "probe":
            pass 
        elif args.task == "ctarget":
            rand_y_tensors = torch.LongTensor(np.random.randint(0, len(self.pos_tags), all_y_tensors.shape)).to(device)
            all_y_tensors = rand_y_tensors 
        elif args.task == "crep":
            all_x_tensors = torch.FloatTensor(np.random.normal(0, 1, all_x_tensors.shape)).to(device)
        else:
            raise ValueError("Task {} not accepted!".format(args.task))

        cache_name = os.path.join(cache_dataset_head, f"{dumpcnt}.pt")
        with open(cache_name, "wb+") as f:
            pickle.dump({
                "x": all_x_tensors,
                "y": all_y_tensors
            }, f)

        return all_x_tensors, all_y_tensors

    def _process_sentence(self, raw_s, spacy_nlp, tokenizer, emb, args):
        spacy_tokens = spacy_nlp(raw_s)
        spacy_token_texts = [token.text_with_ws for token in spacy_tokens]
        hf_tokens = tokenizer.tokenize(raw_s)
        clean_hf_tokens = []
        for token in hf_tokens:
            if token.startswith("##"):
                clean_hf_tokens.append(token[2:])
            else:
                clean_hf_tokens.append(token)
        cost, s2h, h2s, s2h_multi, h2s_multi = spacy.gold.align(spacy_token_texts, clean_hf_tokens)

        BERT_MAX_LEN = 510
        sent_x = []
        sent_y = []

        while len(hf_tokens) > BERT_MAX_LEN:
            hf_tokens_head = hf_tokens[:BERT_MAX_LEN]
            hf_tokens_rem = hf_tokens[BERT_MAX_LEN:]
            h2s_head = h2s[:BERT_MAX_LEN]
            h2s_rem = h2s[BERT_MAX_LEN:]
            h2s_rem = (np.array(h2s_rem) - BERT_MAX_LEN).tolist()

            spacy_tokens_head = spacy_tokens[:h2s[BERT_MAX_LEN]]
            spacy_tokens_rem = spacy_tokens[h2s[BERT_MAX_LEN]:]
            s2h_head = s2h[:h2s[BERT_MAX_LEN]]
            s2h_rem = s2h[:h2s[BERT_MAX_LEN]]
            s2h_rem = (np.array(s2h_rem) - h2s[BERT_MAX_LEN]).tolist()
            
            chunk_x, chunk_y = self._align_chunk(hf_tokens_head, h2s_head, spacy_tokens_head, s2h_head, tokenizer, emb)
            sent_x += chunk_x 
            sent_y += chunk_y 

            hf_tokens = hf_tokens_rem 
            h2s = h2s_rem 
            spacy_tokens = spacy_tokens_rem 
            s2h = s2h_rem 
        chunk_x, chunk_y = self._align_chunk(hf_tokens, h2s, spacy_tokens, s2h, tokenizer, emb)
        sent_x += chunk_x 
        sent_y += chunk_y
        return sent_x, sent_y 

    def _align_chunk(self, hf_tokens_head, h2s_head, spacy_tokens_head, s2h_head, tokenizer, emb):
        chunk_x = []
        chunk_y = []
        # Ok now that *_head does not overflow
        # Process the doc and alignments
        if self.args.lm == "bertmulti":
            hf_indices_head = tokenizer.encode(hf_tokens_head)  # list of int
            vecs, _ = emb(torch.tensor(hf_indices_head).unsqueeze(0))
            # vecs is [1, seq_len, ndim]
        elif self.args.lm == "fasttext":
            vecs = torch.tensor(np.array([emb.wv[w] for w in hf_tokens_head])).unsqueeze(0)
        else:
            raise NotImplementedError
        
        # Just traverse the spacy tokenization
        # When there is a miss, find the corresponding miss at hf tokenization
        # Handle the missing parts. Then repeat at subseq no-miss at spacy sequence
        i, j = 0, -1
        while i < len(s2h_head):
            if s2h_head[i] > 0:
                j = s2h_head[i]
                x = vecs[:, j]  # x is tensor of shape (1, d_emb)
                y = self._pos_to_label(spacy_tokens_head[i].pos_)
                chunk_x.append(x)
                chunk_y.append(y)
                i += 1
            else:
                start_i, end_i = i, i+1
                while end_i < len(s2h_head) and s2h_head[end_i] < 0:
                    end_i += 1
                if end_i == len(s2h_head):
                    break
                i = end_i
                if end_i > start_i + 1:  # Multiple spacy misses. Skip them
                    j = s2h_head[end_i]
                    continue 
                else:  # Only one spacy miss. Find corresponding hf misses
                    # Note1: j corresponds to the *previous* value
                    # Note2: end_j will at most be len(h2s_head)
                    start_j, end_j = j+1, j+2
                    while end_j < len(h2s_head) and h2s_head[end_j] < 0:
                        end_j += 1
                    x = torch.mean(vecs[:, start_j:end_j], dim=1)
                    y = self._pos_to_label(spacy_tokens_head[i].pos_)
                    chunk_x.append(x)
                    chunk_y.append(y)

        return chunk_x, chunk_y 

    def has_next(self):
        return self.ckpt_ptr < len(self.ckpt_filelist)-1 or self.ptr + self.batch_size <= len(self.x)

    def next(self):
        if self.ptr + self.batch_size < len(self.x):
            start = self.ptr 
            end = start + self.batch_size 
            xbatch, ybatch = self.x[start: end], self.y[start: end]
            self.ptr = end
        elif self.ckpt_ptr < len(self.ckpt_filelist)-1:
            self.ckpt_ptr += 1
            self.x, self.y = self._load_file(self.ckpt_ptr)
            start, end = 0, self.batch_size
            xbatch, ybatch = self.x[start: end], self.y[start: end]
            self.ptr = end 
        else:
            return None, None
        return xbatch, ybatch

    def _pos_to_label(self, pos_tag):
        if self.args.lang == "fr":
            if pos_tag == "INTJ":
                pos_tag = "X"
            elif pos_tag == "SYM":
                pos_tag = "X"
        elif self.args.lang == "es":
            if pos_tag == "X":
                pos_tag = "INTJ"
        return torch.LongTensor([self.pos_tags.index(pos_tag)])

    def reset(self):
        self.ckpt_ptr = 0
        self.ptr = 0
        self.x, self.y = self._load_file(self.ckpt_ptr)

    def _load_file(self, ckpt_ptr):
        with open(self.ckpt_filelist[ckpt_ptr], "rb") as f:
            checkpoint = pickle.load(f)
        return checkpoint["x"], checkpoint["y"]


class BaselineTokenizer:
    def __init__(self):
        pass  

    def tokenize(self, s):
        """
        Input: s (a string representation of a sentence)
        Output: tokens (list of string). 
        """
        return s.split()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lm", type=str, choices=["bertmulti", "fasttext", "glove"], default="bertmulti")
    parser.add_argument("--lang", type=str, choices=["en", "es", "fr"], default="en")
    parser.add_argument("--task", type=str, choices=["probe", "ctarget", "crep"], default="probe")

    parser.add_argument("--split", type=str, choices=["train", "dev", "test"], default="dev")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    
    dl = DataLoader(args.split, args)

    print("dl.has_next():", dl.has_next())

    print ("Checking if device is on GPU")
    x_tensor, y_tensor = dl.next()
    print("x_tensor.shape:", x_tensor.shape)
    print("y_tensor.shape:", y_tensor.shape)
    print("x_tensor.device:", x_tensor.device)
    print("y_tensor.device:", y_tensor.device)

    print ("Checking NaN for this dataset")
    dl.reset()
    success = True 
    while dl.has_next():
        x_tensor, y_tensor = dl.next()
        if x_tensor is None:
            break 
        if torch.isnan(x_tensor).sum() > 0:
            print ("x_tensor has nan entries! Stopping!")
            success = False 
            break
        if torch.isnan(y_tensor).sum() > 0:
            print ("y_tensor has nan entries! Stopping!")
            success = False 
            break
    if success:
        print ("NaN test passed!")
    else:
        print ("NaN test failed!")
        
    print ("Compute H(T)")
    import scipy
    from scipy.stats import entropy
    dl.reset()
    labels = []
    while dl.has_next():
        x_tensor, y_tensor = dl.next()
        if x_tensor is None:
            break
        labels += y_tensor.cpu().numpy().tolist()
    ent = entropy(labels, base=2)
    print ("H(T)={:.4f}".format(ent))