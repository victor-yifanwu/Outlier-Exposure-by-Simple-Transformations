# Outlier-Exposure-by-Simple-Transformations

Outlier-Exposure-by-Simple-Transformations built on [OpenOOD](https://github.com/Jingkang50/OpenOOD).

## Installation
To set up OEST, follow these steps to download OpenOOD as the base repository and integrate OEST into it:

1. **Clone the OpenOOD repository**:

   Clone the OpenOOD repository into your working directory:

    ```bash
    git clone https://github.com/Jingkang50/OpenOOD.git
    ```

2. **Clone the OEST repository**:

   Clone the OEST repository into the same directory:

   ```bash
   git clone https://github.com/victor-yifanwu/Outlier-Exposure-by-Simple-Transformations.git OEST
   ```

3. **Copy the OEST files into the OpenOOD directory**:

   Copy all files and directories from the OEST repository into the OpenOOD directory:

   ```bash
   cp -r OEST/* ./OpenOOD/
   ```

After completing these steps, the OpenOOD repository will be integrated with OEST functionalities, and any overlapping files will be replaced by those from OEST.

## Directory Structure

```plaintext
Root Directory (OpenOOD with OEST integrated)
├── cifar10_resnet18_32x32/  # OpenOOD model checkpoints for CIFAR-10
├── cifar100_resnet18_32x32/ # OpenOOD model checkpoints for CIFAR-100
├── evaluate.py              # Main script to evaluate model checkpoints
├── oest/                    # Core functionality
│   ├── cifar10/             # Codebase for CIFAR-10
│   │   ├── checkpoint/      # Final model checkpoints
│   │   ├── main.py          # Main script for training and testing CIFAR-10 models
│   │   ├── train.sh         # Training script
│   │   └── utils/           # Helper utilities
│   ├── cifar100/            # Codebase for CIFAR-100
│   │   ├── checkpoint/      # Final model checkpoints
│   │   ├── main.py          # Main script for training and testing CIFAR-100 models
│   │   ├── train.sh         # Training script
│   │   └── utils/           # Helper utilities
├── openood/                 # Tuned OpenOOD library
├── README.md                # Project documentation
```

## Quick Start

Follow these steps to quickly train models and evaluate their performance using the integrated OpenOOD and OEST setup.

### 1. Train a Model

For Cifar-10
```bash
cd OpenOOD/oest/cifar10
bash train.sh
```
For Cifar-100
```bash
cd OpenOOD/oest/cifar100
bash train.sh
```

### 2. Evaluate a Model
```bash
cd OpenOOD
python evaluate.py
```