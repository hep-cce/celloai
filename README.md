## CCE LLM-based Code Documentation, Generation and Optimization AI Assistant

### Setting up Cello-AI

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
