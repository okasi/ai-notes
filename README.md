# ai-notes

---

## How to get started

Get a Nvidia GPU, preferably RTX 3090+

Install CUDA toolkit  
https://developer.nvidia.com/cuda-11-8-0-download-archive

Install cuDNN  
https://developer.download.nvidia.com/compute/redist/cudnn/v8.8.0/local_installers/11.8/

Install Python via chocolatey  
`choco install python --version=3.10.8`

Install necessary Python dependencies  
```
pip install tqdm boto3 requests regex sacremoses
pip install transformers
pip install sentencepiece
pip install accelerate
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
```

---

## Latest news

https://bleedingedge.ai/

---

## Recent benchmark (30 Jan 2023)

<img width="304" alt="image" src="https://user-images.githubusercontent.com/42962282/216824026-fd713e27-bca2-49d5-ba36-5249903c7811.png">
from https://github.com/facebookresearch/metaseq/tree/main/projects/OPT-IML paper

---

## Open source LLMs that's runnable on consumer grade GPUs (RTX 3090)

### Meta's OPT-IML
https://github.com/facebookresearch/metaseq/tree/main/projects/OPT-IML

https://huggingface.co/facebook/opt-iml-max-1.3b

https://huggingface.co/facebook/opt-13b


### Google's 
https://github.com/google-research/t5x/blob/main/docs/models.md

https://huggingface.co/google/flan-t5-xxl


### Together / EleutherAI
https://www.together.xyz/blog/releasing-v1-of-gpt-jt-powered-by-open-source-ai

https://huggingface.co/togethercomputer/GPT-JT-6B-v1

---


## Turkish

https://huggingface.co/dbmdz/bert-base-turkish-cased

https://github.com/agmmnn/turkish-nlp-resources

https://data.tdd.ai/


## Swedish

https://huggingface.co/AI-Nordics/bert-large-swedish-cased

---

## RLHF

https://github.com/CarperAI/trlx

https://github.com/LAION-AI/Open-Assistant


---

## Biomedical

https://github.com/microsoft/BioGPT

---

https://github.com/sw-yx/ai-notes/blob/main/TEXT.md
