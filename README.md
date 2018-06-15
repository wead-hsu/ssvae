## Variational Autoencoder for Semi-supervised Text Classification

#### All these repositories are used in the paper titled with 'Variational autoencoder for semi-supervised text classification'
------
- list:
	- data: all data files are kept in this directory, including data, word embeddings, pretrained_weights.
	- results: the directory where the resulting models are saved.
	- auxiliary_vae and avae_fixed: the model that uses the auxiliary variable in the VAE, this can produce good results. They differs in whether fixing the latent variable in generation.
	- SemiSample-S1 is the model with sampling-based optimizer with EMA baseline
	- SemiSample-S2 is the model with sampling-based optimizer with VIMCO baseline

#### Note
The code is a little rebundant, as the original model is proposed with auxiliary variable, but it turns out that it also works well without it. To run this code, you may need the pre-processed data, which can be obtained by emailing me (wead_hsu at pku.edu.cn). Or you can just create one with your own data (the format can be inferred from the code).

### Quick Start

```
cd avae_fixed
python imdb_new.py
```

#### Requirements:

- Package requirements:
	- Theano == 0.8
	- Lasagne == 0.2.dev1
	- CUDA == 8.0
