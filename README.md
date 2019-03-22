# LQ-MXNet

## Cifar10 Experiments(Still need a lot of effort to train better models)
| Model(cifar10) | Bit-width(W/A) | Top-1(%) in MXNet | Top-1(%) in Paper |                 scripts                      |
|----------------|----------------|-------------------|-------------------|----------------------------------------------|
| VGG-Small      | 32/32          | 93.55             | 93.8              | [baseline](scripts/train_vggsmall_cifar10_baseline.sh)|
| VGG-Small      | 1/32           | 92.87(A/W)        | 93.5              | [w1a32](scripts/train_vggsmall_cifar10_w1a32.sh)|
| VGG-Small      | 2/32           | 93.60(A/W)        | 93.8              | [w2a32](scripts/train_vggsmall_cifar10_w2a32.sh)|
| VGG-Small      | 3/32           | 93.70(A/W)        | 93.8              | [w3a32](scripts/train_vggsmall_cifar10_w3a32.sh)|
| VGG-Small      | 1/2            | 92.28(A/W)        | 93.4              | [w1a1](scripts/train_vggsmall_cifar10_w1a2.sh) |
| VGG-Small      | 2/2            | 93.25(A/W)        | 93.5              | [w2a2](scripts/train_vggsmall_cifar10_w2a2.sh) |
| VGG-Small      | 2/3            | 93.22(A/W)        | 93.8              | [w2a3](scripts/train_vggsmall_cifar10_w2a3.sh) |
| VGG-Small      | 3/3            | 93.42             | 93.8              | [w3a3](scripts/train_vggsmall_cifar10_w3a3.sh) |