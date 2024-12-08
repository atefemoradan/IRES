# IRES

[![WSDM '25](https://img.shields.io/badge/WSDM%20'25-10.1145/3701551.3703566-blue)](https://doi.org/10.1145/3701551.3703566)

### Introduction

This repository contains the implementation of our approach to entity summarization using Relational Graph Convolutional
Networks (RGCN). Our method aims to leverage the structure and semantics of knowledge graphs by detecting the
anti-communities in the given graph.

### Installation

To get started with IRES, clone this repository and install the required dependencies:

```bash
git clone https://github.com/atefemoradan/IRES
cd IRES
curl -sSL https://install.python-poetry.org | python3 -
poetry install
```

### Running the Code

After setting up your environment and choosing your configuration, you can run the main experiment with the `main.py`
script. Here's a generic command to start the training and evaluation process:

```bash
python3 main.py [-h] [-b {esbm,esbm_plus}] [-d {dbpedia,lmdb}] [-f {rdf,json}] [-s SEED] [-p DATA_PATH] [-O OUTPUT_PATH] [-tf {transe_nodes,transe_node_relations,node_freq,node_relation_freq}] [-e {RGCN,GCN}]
                 [-o {ASGD,AdamW,Adam}] [-sch {StepLR,OneCycleLR,LinearLR,ReduceLROnPlateau,CosineAnnealingLR,CosineAnnealingWarmRestarts}] [-lr LEARNING_RATE] [-wd WEIGHT_DECAY] [-ss STEP_SIZE] [-ne NUM_EPOCHS]
                 [-nc NUM_CNN_LAYERS] [-he HIDDEN_EMBEDDINGS] [-af {ELU,RReLU,ReLU,LeakyReLU}] [-sl SKIP_LAYER] [-bn BATCH_NORM] [-do DROP_OUT] [-la LOSS_ALPHA] [-dt {cuda,cpu}] [-n NUM_EXECUTIONS] [-bm BEST_MODEL]
````

Note: Ensure your dataset is prepared and located in the path specified by `-p` or `--data_path`. The output, including
models and results, will be saved to the directory specified by `-O` or `--output_path`.

### Configuration

You can adjust parameters in the `config.py` file or directly via command line arguments.

**Parameters for Datasets:**

Using `node_freq` + relation embedding:

**ESBM**:

```bash
python3 main.py -b esbm -d lmdb  -tf node_relation_freq -lr 0.01 -wd 0.1 -ne 50 -ss 7 -he 64 -bn True -do 0.25 -la 0.0
python3 main.py -b esbm -d dbpedia -tf node_relation_freq -lr 0.01 -wd 0.1 -ne 50 -ss 7 -he 64 -bn True -do 0.25 -la 0.0
```

**ESBM_Plus**

```bash
python3 main.py -b esbm_plus -d lmdb -tf node_relation_freq -lr 0.01 -wd 0.1 -ne 50 -ss 7 -he 64 -bn True -do 0.25 -la 0.0
python3 main.py -b esbm_plus -d dbpedia -tf node_relation_freq -lr 0.01 -wd 0.1 -ne 50 -ss 7 -he 64 -bn True -do 0.25 -la 0.0
```

Using `transe` embedding:

**ESBM**:

```bash
python3 main.py -b esbm -d lmdb -tf transe_nodes -lr 0.01 -wd 0.01 -ne 50 -ss 7 -he 64 -bn True -do 0.25 -la 0.0
python3 main.py -b esbm -d dbpedia  -tf transe_nodes -lr 0.01 -wd 0.01 -ne 50 -ss 7 -he 64 -bn True -do 0.25 -la 0.0
```

**ESBM_Plus**:

```bash
python3 main.py -b esbm_plus -d lmdb -tf transe_nodes -lr 0.01 -wd 0.01 -ne 50 -ss 7 -he 64 -bn True -do 0.25 -la 0.0
python3 main.py -b esbm_plus -d dbpedia -tf transe_nodes -lr 0.01 -wd 0.01 -ne 50 -ss 7 -he 64 -bn True -do 0.25 -la 0.0
```

These command-line snippets allow for straightforward model execution tailored to your specific dataset and feature
preference. Adjust the parameters as necessary for your experiments.

### Evaluation

The effectiveness of the Indisum model is evaluated using F1 scores among other metrics, which are automatically
calculated during the testing phase of the script execution. Here's how you can interpret the results and understand the
model's performance:

- **F1 Score:** This metric provides a balance between precision and recall. A higher F1 score indicates better
  summarization quality.

- **Output Directory:** All evaluation results, including detailed F1 scores and other performance metrics, are saved in
  the output directory specified by the `--output_path` option. Within this directory, look for
  files `results_{configurations}.csv`
  for aggregated results and additional files for more granular insights.

To further analyze the model's performance or compare it against different configurations or benchmarks, you can
manually inspect the output files.

### Related Work and Comparative Results

Our research critically evaluates the performance of the INDISUM model against four notable related works within the
entity summarization domain. We conducted this comparison using the F1 Score metric across the ESBM and ESBM_Plus
datasets. The results underscore our model's significant advancements in entity summarization.


#### F1-Score on ESBM Dataset

| Method           | DBpedia (top5)   | DBpedia (top10)  | LinkedMDB (top5) | LinkedMDB (top10) | All (top5)   | All (top10)      |
|------------------|------------------|------------------|------------------|-------------------|--------------|------------------|
| RELIN            | 0.265            | 0.459            | 0.285            | 0.345             | 0.270        | 0.426            |
| MPSUM            | 0.306            | 0.504            | 0.265            | 0.424             | 0.293        | 0.480            |
| BAFREC           | <ins>0.321</ins> | 0.459            | <ins>0.335</ins> | 0.307             | <u>0.325</u> | 0.416            |
| KAFCA            | 0.318            | 0.509            | 0.229            | 0.402             | 0.273        | 0.455            |
| IRES (TransE)    | 0.306            | <ins>0.535</ins> | 0.322            | <ins>0.430</ins>  | 0.310        | <ins>0.505</ins> |
| IRES (frequency) | **0.327**        | **0.540**        | **0.350**        | **0.445**         | **0.334**    | **0.513**        |

<ins>Best performer in bold; second-best underlined.</ins>

#### F1-Score on ESBM_Plus Dataset

| Method           | DBpedia (top5)   | DBpedia (top10)  | LinkedMDB (top5) | LinkedMDB (top10) | All (top5)       | All (top10)      |
|------------------|------------------|------------------|------------------|-------------------|------------------|------------------|
| RELIN            | 0.211            | 0.441            | 0.20             | 0.312             | 0.208            | 0.404            |
| MPSUM            | 0.235            | 0.450            | 0.174            | 0.353             | 0.217            | 0.422            |
| BAFREC           | 0.143            | 0.180            | 0.155            | 0.128             | 0.146            | 0.165            |
| KAFCA            | 0.183            | 0.369            | 0.108            | 0.249             | 0.162            | 0.334            |
| IRES (TransE)    | <ins>0.267</ins> | <ins>0.467</ins> | 0.23             | <ins>0.38</ins>   | <ins>0.258</ins> | <ins>0.436</ins> |
| IRES (frequency) | **0.334**        | **0.534**        | **0.30**         | **0.433**         | **0.324**        | **0.505**        |

<ins>Best performer in bold; second-best underlined.</ins>

### MAP on ESBM Dataset.

| Method           | DBpedia (top5)   | DBpedia (top10)  | LinkedMDB (top5) | LinkedMDB (top10) | All (top5)       | All (top10)      |
|------------------|------------------|------------------|------------------|-------------------|------------------|------------------|
| RELIN            | 0.14             | 0.29             | 0.153            | 0.22              | 0.144            | 0.27             |
| MPSUM            | 0.181            | 0.328            | 0.152            | 0.254             | 0.17             | 0.306            |
| BAFREC           | 0.179            | 0.273            | 0.156            | 0.13              | 0.172            | 0.232            |
| KAFCA            | <ins>0.199</ins> | 0.337            | 0.137            | 0.24              | 0.181            | 0.309            |
| IRES (TransE)    | 0.19             | <ins>0.352</ins> | <ins>0.22</ins>  | <ins>0.267</ins>  | <ins>0.198</ins> | <ins>0.327</ins> |
| IRES (frequency) | **0.207**        | **0.367**        | **0.226**        | **0.272**         | **0.212**        | **0.34**         |

<ins>Best performer in bold; second-best underlined.</ins>

### MAP on ESBM_Plus Dataset

| Method           | DBpedia (top5)   | DBpedia (top10) | LinkedMDB (top5) | LinkedMDB (top10) | All (top5)       | All (top10)      |
|------------------|------------------|-----------------|------------------|-------------------|------------------|------------------|
| RELIN            | 0.123            | 0.274           | 0.11             | 0.19              | 0.12             | 0.25             |
| MPSUM            | 0.137            | 0.287           | 0.088            | 0.18              | 0.126            | 0.256            |
| BAFREC           | 0.073            | 0.091           | 0.081            | 0.04              | 0.075            | 0.76             |
| KAFCA            | 0.104            | 0.21            | 0.052            | 0.117             | 0.089            | 0.048            |
| IRES (TransE)    | <ins>0.153</ins> | <ins>0.3</ins>  | <ins>0.133</ins> | <ins>0.224</ins>  | <ins>0.147</ins> | <ins>0.278</ins> |
| IRES (frequency) | **0.214**        | **0.366**       | **0.168**        | **0.25**          | **0.2**          | **0.332**        |

<ins>Best performer in bold; second-best underlined.</ins>

For detailed insights into these related works and their approaches to entity summarization, please refer to their
respective publications:

- [MPSUM](https://github.com/msorkhpar/MPSUM)
- [BAFREC](https://github.com/msorkhpar/HermannKroll-EntityCharacterization)
- [KAFKA](https://github.com/msorkhpar/KAFCA)

The comparative analysis demonstrates the value of employing RGCN for entity summarization, particularly for its ability
to capture and utilize the rich semantic relationships and community structures inherent in knowledge graphs.

## Citation

```bibtex
@inproceedings{moradan2025untapping,
  title={Untapping the Power of Indirect Relationships in Entity Summarization},
  author={Moradan, Atefeh and Sorkhpar, Mohammad and Miyauchi, Atsushi and Mottin, Davide and Assent, Ira},
  booktitle={WSDM '25: Proceedings of The Eighteenth ACM International Conference on Web Search and Data Mining},
  year={2025},
  doi={10.1145/3701551.3703566},
  url={https://github.com/atefemoradan/IRES},
  version={0.1.0}
}
```

#### License

This project is licensed under the MIT License. For more details, see the [LICENSE](LICENSE) file in the repository.
Utilizing the IRES code in your work implies agreement with the license terms. The MIT License is permissive and allows
for considerable freedom in usage and redistribution, but it requires acknowledgment of the original source.
