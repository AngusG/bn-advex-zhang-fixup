# Evaluating models 
as in **Batch Norm is a Cause of Adversarial Vulnerability**

To evaluate a checkpoint on adversarial examples:
You will need to `pip install advertorch`, then

```
 CUDA_VISIBLE_DEVICES=0 python cifar_eval_advertorch.py --dataroot /scratch/gallowaa/cifar10 --resume /scratch/gallowaa/logs/bn-robust/cifar10/zhang-fixup/checkpoint/fixup_resnet32_benchmark_resnet32_22222.ckpt -a fixup_resnet32
 ```
The script is set for PGD examples, but see [AdverTorch docs](https://github.com/BorealisAI/advertorch) to test other types).

To evaluate a checkpoint on CIFAR-10-C:

- CIFAR-10-C can be downloaded [here](https://zenodo.org/record/2535967#.XREQUHVKgUE).
- Supply the directory you save it in to `--dataroot`.
- The script logs data to Google sheets by default. You can comment this out if you wish, otherwise to authorize Google sheets you will need to create a `token.pickle` file. More info can be found [here](https://developers.google.com/sheets/api/quickstart/python). You will then need to create a new spreadsheet and obtain the `sheet_id`, which is part of the sheet url like: `https://docs.google.com/spreadsheets/d/<this_is_the_sheet_id>/edit#gid=1631286911`. Finally, the script expects the sheet to have a tab called `CIFAR-10-C`, otherwise you can set this variable to "Sheet1" which is the default name.

Putting this all together:
```
CUDA_VISIBLE_DEVICES=0 python cifar_eval_common.py -a fixup_resnet20 --resume /scratch/ssd/logs/bn-robust/cifar10/zhang-fixup/checkpoint/fixup_resnet20_benchmark_fixup_resnet20_11111.ckpt --dataroot /scratch/ssd/data/CIFAR-10-C --sheet_id <google_spreadsheet_id>
```
will loop through all the corruption files at each intensity, convert them to pytorch dataloaders, and evaluate the model. 
[Here is some sample output in the spreadsheet](cifar10c_fixup.png)

# Fixup (Original README)
**A Re-implementation of Fixed-update Initialization (https://arxiv.org/abs/1901.09321). *(requires Pytorch 1.0)***

**Cite as:**

*Hongyi Zhang, Yann N. Dauphin, Tengyu Ma. Fixup Initialization: Residual Learning Without Normalization. 7th International Conference on Learning Representations (ICLR 2019).*

----
## ResNet for CIFAR-10
The default arguments will train a ResNet-110 (https://arxiv.org/abs/1512.03385) with Fixup + Mixup (https://arxiv.org/abs/1710.09412).

*Example:*

The following script will train a ResNet-32 model (https://arxiv.org/abs/1512.03385) on GPU 0 with Fixup and no Mixup (alpha=0), with weight decay 5e-4 and (the default) learning rate 0.1 and batch size 128.
```
CUDA_VISIBLE_DEVICES=0 python cifar_train.py -a fixup_resnet32 --sess benchmark_a0d5e4lr01 --seed 11111 --alpha 0. --decay 5e-4
```

----
## ResNet for ImageNet
ImageNet models with training scripts are now available. (Thanks @tjingrant for help!) 

Top-1 accuracy for ResNet-50 at Epoch 100 with Mixup (alpha=0.7) is around 76.0%.

----
## Transformer for machine translation
Transformer model with Fixup (instead of layer normalization) is available. To run the experiments, you will need to download and install the fairseq library (the provided code was tested on an earlier version: https://github.com/pytorch/fairseq/tree/5d00e8eea2644611f397d05c6c8f15083388b8b4). You can then copy the files into corresponding folders.

An example script `run.sh` is provided to run the IWSLT experiments described in the paper. For more information, please refer to the instructions in fairseq repo (https://github.com/pytorch/fairseq/tree/5d00e8eea2644611f397d05c6c8f15083388b8b4/examples/translation).
