# Unofficial Implementation of Prediction Depth
This is a community script for [Deep Learning Through the Lens of Example Difficulty](https://arxiv.org/abs/2106.09647).

## requirement
```shell script
pip3 install -r requirement.txt
```
## Get Started
### Modify CIFAR10 to get index of data point (Important)
Change __getitem__ of torchvision.datasets.CIFAR10 to output index of current data point
```python
#130        return img, target
        return (img, target), index
```
Make a log directory for ResNet18 with Weight Standardization and Group Norm / original ResNet18 / VGG16
```shell script
mkdir ./cl_results_resnet
mkdir ./cl_results_vgg
```
Changing number of random seeds allows you to train more models to get average PD (line 284 in get_pd_resnet.py).
Run training and plot the 2D histogram for train split and validation split afterwards.
```shell script
python3 get_pd_resnet_wsgn.py --result_dir ./cl_results_wsgn --train_ratio 0.5 --knn_k 30
python3 plot_pd_hist.py --result_dir ./cl_results_wsgn
python3 get_pd_resnet.py --result_dir ./cl_results_resnet --train_ratio 0.5 --knn_k 30
python3 plot_pd_hist.py --result_dir ./cl_results_resnet
```

## Run PD in oneline
Alternatively, run the following code to get all previous results in one line
```shell script
sh run_pd.sh
```