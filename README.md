# Re-implementation of ShaDDR

This repo contains an unofficial re-implementation of the paper [ShaDDR](https://arxiv.org/abs/2306.04889)


# Citation

Please cite the original ShaDDR paper:

```
@misc{chen2023shaddr,
      title={ShaDDR: Interactive Example-Based Geometry and Texture Generation via 3D Shape Detailization and Differentiable Rendering}, 
      author={Qimin Chen and Zhiqin Chen and Hang Zhou and Hao Zhang},
      year={2023},
      eprint={2306.04889},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Dependencies

Please use the provided `conda.yml` file in `envs` folder to install the dependencies.

## Data

To use your own data, follow the [DECORGAN Data Preparation pipeline](https://github.com/czq142857/DECOR-GAN/tree/main/data_preparation).
## Run Training

You can run an experiment by first modifying the config yaml file at `configs/shaddr.yaml`. To run a training run, use the following commands:

```
conda activate shaddr
python main.py --conf <path to config file> --run <name of the wandb run, optional> 
```

In order to disable WandB logging, use the `--debug` flag.

