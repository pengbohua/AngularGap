# Angular Gap
This is the code necessary to run experiments described in the ACM MM'22 paper [Paper](https://arxiv.org/abs/2207.08525)
## Requirements
All the required packages can be installed by running `pip install -r requirements.txt`.

Or pull our domain adaptation docker image and run experiments with 
```
docker pull marvinpeng2022/da-testbed
```
## Difficulty estimation
```shell
python main.py --dst cifar10 --arch resnet18
```
## Visualization
```shell
python main.py --dst cifar10 --arch visualization
```
## Domain adaptation
For domain adaptation, we have released our implementation of CRST and Curricular DSAN.
```shell
cd DeepDA
bash run.sh
```
## Video
[Presentation](https://files.atypon.com/acm/f7197189de64e2075eb0a2c2d1eee630) and [Slides](https://github.com/pengbohua/AngularGap/blob/main/AngularGapPre.pdf)

If you make use of this code in your work, please cite the following paper:
```
@article{peng2022angular,
  title={Angular Gap: Reducing the Uncertainty of Image Difficulty through Model Calibration},
  author={Peng, Bohua and Islam, Mobarakol and Tu, Mei},
  journal={arXiv preprint arXiv:2207.08525},
  year={2022}
}
```
