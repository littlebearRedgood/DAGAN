# DA_GAN


# Usage
## Dependencies
- Linux: Ubuntu 20.04

- Python: Recommend to use Anaconda

- Anaconda:

        1) Create an Anaconda virtual environmentconda: create --name my_env python=3.8 --yes;
        2) Activate the virtual environment: source activate my_env
- PyTorch: 

        pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117        
- Packages:

        pip install -r requirements.txt

## Datasets
- The datasets used include SUIM, UIEB, EUVP, LSUI, UCCS, and SQUID.
- SUIM (https://pan.baidu.com/s/1cgdUaANiftDBSNuEzDdGdw?pwd=suim)
- UIEB (https://pan.baidu.com/s/1gKx1LqgzBNiesAsbtEbTbg?pwd=uieb)
- EUVP (https://pan.baidu.com/s/1HeBVXw2PLNSLaemYuBi6_w?pwd=eubp)
- LSUI (https://pan.baidu.com/s/1BUTwL55wI2h56LlB9Lk7Yw?pwd=lsui)
- UCCS (https://pan.baidu.com/s/1kqMi-syxTSmx_SZEvpbk8Q?pwd=uccs)
- SQUID (https://pan.baidu.com/s/1ICmcWaxIOJuaID_4rnqUqQ?pwd=squi)
## Get Started
#### Pretrained models
- Models are available in ```'./saveModels/generator/generator.pth' and './saveModels/discriminator/discriminator.pth'```
- The download link for the Pretrained models is: https://drive.google.com/file/d/1EsG5z_yJecoreN_HXV6M_MPubjTlPlBW/view?usp=sharing
#### Test:
- Run this command to execute

        python test.py --generator_weights ./saveModels/generator/generator.pth --discriminator_weights ./saveModels/discriminator/discriminator.pth --input_path ./DataSets/dataset/raw/ --output_path ./DataSets/dataset/our/
#### Train:
- Run this command to execute

        python train.py --batch 32 --epoch 100 --lr 0.0001 --checkpoint_interval 10 --sample_interval 50 --resume True 

# Citation
If our work is useful for your research, please cite our work
