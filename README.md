# Cross-Domain Ensemble Distillation for Domain Generalization (ECCV 2022)

Official PyTorch implementation of "Cross-Domain Ensemble Distillation for Domain Generalization" (ECCV 2022)

For more information, please checkout our [website](https://github.com/leekyungmoon/XDED) and [paper](https://github.com/leekyungmoon/XDED).

<!---
Code will be available as soon as possible.
--->

## Get started
### Prepare environment
```bash
conda env create --file environment.yaml
conda activate xded
```
### Quick start
```bash
python pacs_cartoon_train.py  --gpu-id 0 --IPC 16 \
--dataset-config-file configs/datasets/domain_ipc_pacs.yaml \
--config-file configs/xded_default.yaml \
--trainer XDED --remark XDED_UniStyle12 \
MODEL.BACKBONE.NAME resnet18_UniStyle_12
```

## Acknowledgements
Our code is based on [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch). We thank [Kaiyang Zhou](https://kaiyangzhou.github.io/) for this great repository.


## Citation
In case of using this source code for your research, please cite our paper.

```
@inproceedings{lee2022xded,
  title={Cross-Domain Ensemble Distillation for Domain Generalization},
  author={Lee, Kyungmoon and Kim, Sungyeon and Kwak, Suha},
  booktitle={Proceedings of European Conference on Computer Vision (ECCV)},
  year={2022}
}
```
