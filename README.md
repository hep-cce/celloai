## CCE LLM-based Code Documentation, Generation and Optimization AI Assistant

### DISCLAIMER: Experimental Code

This repository contains experimental code under active development. It is 
intended for research/testing purposes only. We provide no guarantees of 
reliability or stability, and the code structure may change significantly 
at any time.


### Overview

CelloAI is a locally hosted coding assistant that leverages
Large Language Models with Retrieval-Augmented Generation to
support High Energy Physics code documentation and generation.
This local deployment ensures data privacy, eliminates recurring
costs, and provides access to large context windows without external
dependencies. CelloAI addresses code documentation and
code generation through specialized components. For code documentation,
the assistant provides: (a) Doxygen style comment
generation by retrieving relevant information from text sources, (b)
File-level summary generation, and (c) An interactive chatbot for
code comprehension queries. For code generation, CelloAI employs
syntax-aware chunking that preserve syntactic boundaries during
embedding thus improving retrieval accuracy in large codebases.
The system integrates callgraph knowledge to maintain dependency
awareness during code modifications and provides AI-generated
suggestions for performance optimization and accurate refactoring.


### Quickstart Guide: Setting up Cello-AI

Create a conda environment 
```
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH
conda create -n cello-ai-101525 python=3.10 ipython
conda activate cello-ai-101525
```

Installating the dependencies
```
pip install -r requirements.txt
```

### Adding Callgraphs into the database
Installating doxygen, graphviz
```
conda install -c conda-forge doxygen
conda install -c conda-forge graphviz
```

Running doxygen to create callgraphs
```
python scripts/run_doxygen.py
```

#### On BNL CSI Dahlia the above environment is available
```
conda activate /data/cello-ai-101525/
```

### Running

1. mkdir SOURCE_DOCUMENTS and copy documents into it

2. Ingest files into database
```
python localGPT/ingest.py
```

3. Select an LLM in src/config.py; see config_backup.py for examples
```
python scripts/run_celloai.py --model_type=llama3 --temperature=0.01
```

4. Running the setup using the selected LLM or LlamaCpp server at BNL
```
python scripts/run_chatbot.py --model_type=llama3 --temperature=0.01
python scripts/run_chatbot.py --llamacpp_server
```

### Citation
Please see the [CelloAI arxiv](https://arxiv.org/abs/2508.16713):

```bibtex
@article{atif2025celloai,
  title={CelloAI: Leveraging Large Language Models for HPC Software Development in High Energy Physics},
  author={Atif, Mohammad and Chopra, Kriti and Kilic, Ozgur and Wang, Tianle and Dong, Zhihua and Leggett, Charles and Lin, Meifeng and Calafiura, Paolo and Habib, Salman},
  journal={arXiv preprint arXiv:2508.16713},
  year={2025}
}
```

### Licensing

This project uses classes and routines from two software (PromtEngineer/localGPT, 
fynnfluegge/doc-comments-ai) with compatible, permissive MIT licenses. Their 
licenses are available at 'LICENSES/LICENSE1.MIT' and 'LICENSES/LICENSE2.MIT'. 
All new contributions are licensed under BSD 3-Clause License available in 
'LICENSE' file.

