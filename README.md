# Cross-Domain Ensemble Distillation for Domain Generalization (ECCV 2022)

Official PyTorch implementation of "Cross-Domain Ensemble Distillation for Domain Generalization" (ECCV 2022)

For more information, please checkout our [website](https://github.com/leekyungmoon/XDED) and [paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850001.pdf).

<!---
Code will be available as soon as possible.
--->

	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cross-domain-ensemble-distillation-for-domain/domain-generalization-on-pacs-2)](https://paperswithcode.com/sota/domain-generalization-on-pacs-2?p=cross-domain-ensemble-distillation-for-domain)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cross-domain-ensemble-distillation-for-domain/domain-generalization-on-office-home)](https://paperswithcode.com/sota/domain-generalization-on-office-home?p=cross-domain-ensemble-distillation-for-domain)

<a href="url" >:arrow_right:</a> We remark that, when with ResNet-18 for a fair comparison, our approach records the second and first place in the leaderboard of paperwithcode for PACS and OfficeHome, respectively.

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
@inproceedings{lee2022cross,
  title={Cross-Domain Ensemble Distillation for Domain Generalization},
  author={Lee, Kyungmoon and Kim, Sungyeon and Kwak, Suha},
  booktitle={Proceedings of European Conference on Computer Vision (ECCV)},
  year={2022}
}
```
