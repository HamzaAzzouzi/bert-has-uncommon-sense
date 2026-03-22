# Introduction
This is the code accompanying our paper [_Bert Has Uncommon Sense_](https://aclanthology.org/2021.blackboxnlp-1.43/).

To cite:

```
@inproceedings{gessler-schneider-2021-bert,
    title = "{BERT} Has Uncommon Sense: Similarity Ranking for Word Sense {BERT}ology",
    author = "Gessler, Luke  and
      Schneider, Nathan",
    booktitle = "Proceedings of the Fourth BlackboxNLP Workshop on Analyzing and Interpreting Neural Networks for NLP",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.blackboxnlp-1.43",
    doi = "10.18653/v1/2021.blackboxnlp-1.43",
    pages = "539--547"
}
```

# Usage
## Data
The code expects datasets to live under the repository `data/` directory. In practice:

- PDEP / `clres` is read from `data/pdep/pdep_train.conllu` and `data/pdep/pdep_test.conllu`.
- STREUSLE fine-tuning data is read from `data/streusle/train/streusle.ud_train.json` (the `data/streusle` submodule).
- OntoNotes must be placed at `data/conll-formatted-ontonotes-5.0`, with splits under:
  - `data/conll-formatted-ontonotes-5.0/data/train`
  - `data/conll-formatted-ontonotes-5.0/data/development`
  - `data/conll-formatted-ontonotes-5.0/data/test`

If your local archive is unpacked as `data/conll-formatted-ontonotes-5.0-12/conll-formatted-ontonotes-5.0`, move or symlink the inner `conll-formatted-ontonotes-5.0` directory to `data/conll-formatted-ontonotes-5.0`, because that is the path hardcoded in `main.py`.

Cf. https://docs.allennlp.org/models/main/models/common/ontonotes/ and https://cemantix.org/data/ontonotes.html. 
(If you're a GU affiliate, email Luke at lg876@georgetown.edu.)

## Setup
1. Initiate submodules:

```bash
git submodule init
git submodule update
```

2. Create a new Anaconda environment:

```bash
$ conda create --name bhus python=3.8
```

3. Install dependencies:

```
conda activate bhus
pip install -r requirements.txt
```

4. Perform a test run to ensure everything's working:

```
mkdir models
# finetune the model on a small number of STREUSLE instances
python main.py finetune --num_insts 100 distilbert-base-cased models/distilbert-base-cased_100.pt
# run the trials using the finetuned model we just created on pdep--note `clres` is an alias we use for pdep
python main.py trial \
    --embedding-model distilbert-base-cased \
    --metric cosine \
    --query-n 1 \
    --bert-layer 5 \
    --override-weights models/distilbert-base-cased_100.pt \
    clres
# analyze results by bucket
python main.py summarize \
    --embedding-model distilbert-base-cased \
    --metric cosine \
    --query-n 1 \
    --bert-layer 5 \
    --override-weights models/distilbert-base-cased_100.pt \
    clres
```

## GPU troubleshooting

If fine-tuning fails with an error like `CUDA error: no kernel image is available for execution on the device`,
your installed PyTorch build does not support your GPU architecture. We observed this with an NVIDIA RTX A5000
(`sm_86`) and an older PyTorch build that only supported `sm_37 sm_50 sm_60 sm_70`.

To train on GPU, reinstall PyTorch with a CUDA build that supports your device:

```bash
pip uninstall -y torch torchvision torchaudio
pip cache purge
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Then verify that PyTorch sees the GPU and supports its architecture:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_arch_list())"
```

The output should include `sm_86` for an RTX A5000. If you prefer not to change the PyTorch installation,
you can fine-tune on CPU instead, but it will be slower.

## Execution

Type `bash scripts/all_experiments.sh` to yield the numbers used in the paper's tables, which will appear in TSV 
files by bucket. For individual results on each trial, see `.tsv` files under `cache/`.
In between runs, clear `cache/`, `models/`, and `*.tsv`.

# Acknowledgments
Thanks to Ken Litkowski for allowing us to distribute a subset of the 
[PDEP corpus](https://www.aclweb.org/anthology/P14-1120.pdf) with our code. 
