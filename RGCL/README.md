# EECE-7205-RGCL

## Overview
This project is based on the code from [RGCL](https://github.com/JingbiaoMei/RGCL/tree/main) and data from [HarMeme](https://github.com/di-dimitrov/mmf/tree/master/data/datasets/memes/defaults). The original code has been minimally modified to also give output for a custom input i.e., for a custom meme this code will return harmful or not harmful.

## Getting started
1. Clone this repository into your system
2. Data:
   - Dump all image data into `./data/image/HarMeme/All`
   - Dump all jsonl annotation files into `./data/gt/HarMeme`
3. Requirements:  
   Create env:
   - `conda create -n RGCL python=3.10 -y`
   - `conda activate RGCL`
   - `conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y`
   - `conda install -c pytorch -c nvidia faiss-gpu=1.7.4 mkl=2021 blas=1.0=mkl -y`
   - `pip install -r requirements.txt`
   - `pip install easyocr`
4. Running the script  
   *  Training
      - run generate_CLIP_embedding_HF.py
      - `bash experiments.sh > out.txt`  
   * Testing
      - place images in `./data/image/HarMeme/Test`
      - run create_jsonl.py
      - run generate_CLIP_embedding_HF_test.py
      - change run_rac.py to run_rac_test.py in experiments.sh
      - `bash experiments.sh`

## Results
Results of training are in out.txt
- highest accuracy reached = 0.8475
- highest roc achieved = 0.9132

## Reference
This is to clarify again that most of the code is adopted directly from [RGCL](https://github.com/JingbiaoMei/RGCL/tree/main)  
Citation:
```bibtex
@inproceedings{RGCL2024Mei,
    title = "Improving Hateful Meme Detection through Retrieval-Guided Contrastive Learning",
    author = "Mei, Jingbiao  and
      Chen, Jinghong  and
      Lin, Weizhe  and
      Byrne, Bill  and
      Tomalin, Marcus",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.291",
    doi = "10.18653/v1/2024.acl-long.291",
    pages = "5333--5347"
}
