"""
Extract word-level embeddings from large HF models (e.g. Qwen2.5-3B) for
the PDEP/CL-Res and OntoNotes corpora, producing pickled datasets compatible
with the existing bssp analysis pipeline.

Usage:
    python modern_models/extract_embeddings.py \
        --model Qwen/Qwen2.5-3B --corpus clres --layer -1 [--batch-size 16]
"""

import os
import pickle
from collections import defaultdict

import click
import conllu
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel


def load_model_and_tokenizer(model_name, device):
    """Load a HF model and tokenizer. Tries causal LM first, falls back to AutoModel."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
            output_hidden_states=True,
            trust_remote_code=True,
        )
    except (ValueError, OSError):
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
            output_hidden_states=True,
            trust_remote_code=True,
        )
    model.eval()
    return model, tokenizer


def get_word_to_subword_map(tokenizer, words):
    """
    Given a list of original words, tokenize each independently and build
    a mapping from word index to subword token indices in the full sequence.

    Returns:
        input_ids: tensor of token ids for the full sentence
        word_to_subword: list of (start, end) subword index ranges per word
    """
    # Tokenize the full sentence as a joined string
    text = " ".join(words)
    encoding = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=True)
    input_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"][0].tolist()  # list of (char_start, char_end)

    # Build char offset → word index mapping
    char_to_word = {}
    char_pos = 0
    for word_idx, word in enumerate(words):
        start = text.find(word, char_pos)
        for c in range(start, start + len(word)):
            char_to_word[c] = word_idx
        char_pos = start + len(word)

    # Map each subword token to a word index
    word_to_subwords = defaultdict(list)
    for token_idx, (char_start, char_end) in enumerate(offsets):
        if char_start == char_end:
            # Special token (CLS, SEP, PAD) or empty
            continue
        # Find which word this subword belongs to
        for c in range(char_start, char_end):
            if c in char_to_word:
                word_to_subwords[char_to_word[c]].append(token_idx)
                break

    # Convert to ranges
    word_to_subword = []
    for word_idx in range(len(words)):
        indices = word_to_subwords.get(word_idx, [])
        if indices:
            word_to_subword.append((min(indices), max(indices) + 1))
        else:
            word_to_subword.append(None)

    return input_ids, word_to_subword


def extract_embedding(model, tokenizer, words, target_idx, layer, device):
    """
    Extract the embedding for words[target_idx] from the specified layer.
    Averages over subword tokens if the word is split into multiple pieces.

    Returns:
        numpy array of shape (hidden_dim,)
    """
    input_ids, word_to_subword = get_word_to_subword_map(tokenizer, words)
    mapping = word_to_subword[target_idx]
    if mapping is None:
        return None

    sub_start, sub_end = mapping
    input_ids = input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids)
        hidden_states = outputs.hidden_states  # tuple of (1, seq_len, hidden_dim)
        layer_output = hidden_states[layer][0]  # (seq_len, hidden_dim)

    target_embeddings = layer_output[sub_start:sub_end]  # (n_subwords, hidden_dim)
    word_embedding = target_embeddings.mean(dim=0).float().cpu().numpy()
    return word_embedding


# ---------------------------------------------------------------------------
# Corpus readers (lightweight, no AllenNLP dependency)
# ---------------------------------------------------------------------------

def read_clres_instances(filepath):
    """Read PDEP/CL-Res CoNLL-U file and yield (words, target_idx, label) tuples."""
    with open(filepath, "r") as f:
        tokenlists = conllu.parse(f.read())
    for tokenlist in tokenlists:
        prep_index = int(tokenlist.metadata["prep_id"]) - 1
        words = [t["form"] for t in tokenlist]
        lemma = tokenlist[prep_index]["lemma"]
        sense = tokenlist[prep_index]["misc"]["Sense"]
        label = f"{lemma}_{sense}"
        yield words, prep_index, label


def read_ontonotes_instances(dirpath):
    """Read OntoNotes CoNLL files. Requires allennlp_models for the Ontonotes reader."""
    from allennlp_models.common.ontonotes import Ontonotes

    reader = Ontonotes()
    for doc_path in tqdm(list(Ontonotes.dataset_path_iterator(dirpath)), desc="OntoNotes docs"):
        for doc in reader.dataset_document_iterator(doc_path):
            for sent in doc:
                if all(sense is None for sense in sent.word_senses):
                    continue
                words = sent.words
                for i, sense in enumerate(sent.word_senses):
                    if sense is not None:
                        lemma = sent.predicate_lemmas[i]
                        pos_tag = sent.pos_tags[i]
                        simplified_pos = (
                            "n" if pos_tag.startswith("N") else "v" if pos_tag.startswith("V") else None
                        )
                        label = f"{lemma}_{simplified_pos}_{sense}"
                        yield words, i, label


def read_corpus(corpus_name, split):
    """Dispatch to the right reader and return instances."""
    if corpus_name == "clres":
        filepath = f"data/pdep/pdep_{split}.conllu" if split != "dev" else None
        if filepath is None:
            return []
        return list(read_clres_instances(filepath))
    elif corpus_name == "ontonotes":
        if split == "train":
            dirpath = "data/conll-formatted-ontonotes-5.0/data/train"
        elif split == "dev":
            dirpath = "data/conll-formatted-ontonotes-5.0/data/development"
        elif split == "test":
            dirpath = "data/conll-formatted-ontonotes-5.0/data/test"
        else:
            raise ValueError(f"Unknown split: {split}")
        return list(read_ontonotes_instances(dirpath))
    else:
        raise ValueError(f"Unknown corpus: {corpus_name}")


# ---------------------------------------------------------------------------
# Dataset structure compatible with bssp analysis pipeline
# ---------------------------------------------------------------------------

class SimpleInstance:
    """Lightweight replacement for AllenNLP Instance for the analysis pipeline."""
    def __init__(self, words, target_idx, label, embedding=None):
        self.data = {
            "text": SimpleTextField(words),
            "label_span": SimpleSpanField(target_idx, target_idx),
            "label": SimpleLabelField(label),
            "lemma": SimpleLabelField(label[:label.rfind("_")]),
        }
        if embedding is not None:
            self.data["span_embeddings"] = SimpleArrayField(embedding.reshape(1, -1))

    def __getitem__(self, key):
        return self.data[key]

    def __contains__(self, key):
        return key in self.data

    def keys(self):
        return self.data.keys()


class SimpleTextField:
    def __init__(self, words):
        self.tokens = [SimpleToken(w) for w in words]


class SimpleToken:
    def __init__(self, text):
        self.text = text


class SimpleSpanField:
    def __init__(self, start, end):
        self.span_start = start
        self.span_end = end


class SimpleLabelField:
    def __init__(self, label):
        self.label = label


class SimpleArrayField:
    def __init__(self, array):
        self.array = array


def build_dataset(corpus_name, split, model, tokenizer, layer, device):
    """Read corpus, extract embeddings, return list of SimpleInstance."""
    raw_instances = read_corpus(corpus_name, split)
    dataset = []
    failed = 0
    for words, target_idx, label in tqdm(raw_instances, desc=f"Extracting {split}"):
        embedding = extract_embedding(model, tokenizer, words, target_idx, layer, device)
        if embedding is None:
            failed += 1
            continue
        dataset.append(SimpleInstance(words, target_idx, label, embedding))

    dataset.sort(key=lambda x: x["label"].label)
    if failed > 0:
        print(f"Warning: {failed} instances failed embedding extraction (skipped)")
    return dataset


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--model", "model_name", required=True, help="HuggingFace model name (e.g. Qwen/Qwen2.5-3B)")
@click.option("--corpus", required=True, type=click.Choice(["clres", "ontonotes"]))
@click.option("--layer", type=int, default=-1, help="Which hidden layer to extract (default: -1 = last)")
@click.option("--device", default="auto", help="Device: 'auto', 'cuda:0', 'cpu'")
@click.option("--output-dir", default="cache/modern_models", help="Output directory for pickled datasets")
def main(model_name, corpus, layer, device, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"Loading model {model_name} on {device}...")
    model, tokenizer = load_model_and_tokenizer(model_name, device)

    model_slug = model_name.replace("/", "_")

    # Train split (with embeddings)
    print(f"Processing train split for {corpus}...")
    train_dataset = build_dataset(corpus, "train", model, tokenizer, layer, device)
    train_path = os.path.join(output_dir, f"{corpus}_{model_slug}_train.pkl")
    with open(train_path, "wb") as f:
        pickle.dump(train_dataset, f)
    print(f"Saved {len(train_dataset)} train instances to {train_path}")

    # Test split (with embeddings — needed for query embedding at retrieval time)
    print(f"Processing test split for {corpus}...")
    test_dataset = build_dataset(corpus, "test", model, tokenizer, layer, device)

    if corpus == "ontonotes":
        # OntoNotes: combine dev + test as query set (matching original paper)
        print(f"Processing dev split for {corpus}...")
        dev_dataset = build_dataset(corpus, "dev", model, tokenizer, layer, device)
        test_dataset = dev_dataset + test_dataset

    test_path = os.path.join(output_dir, f"{corpus}_{model_slug}_test.pkl")
    with open(test_path, "wb") as f:
        pickle.dump(test_dataset, f)
    print(f"Saved {len(test_dataset)} test instances to {test_path}")

    print("Done.")


if __name__ == "__main__":
    main()
