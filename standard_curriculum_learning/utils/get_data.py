import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def get_dataset(dataset_name, data_dir, split, order=None, rand_fraction=None,clean=False, transform=None, imsize=None, bucket='pytorch-data', **kwargs):
  dataset = globals()[f'get_{dataset_name}'](dataset_name, data_dir, split, transform=imsize, imsize=imsize, bucket=bucket, **kwargs)

  item = dataset.__getitem__(0)[0]
  print (item.size(0))
  dataset.nchannels = item.size(0)
  dataset.imsize = item.size(1)
  return dataset


def get_aug(split, imsize=None, aug='large'):
  if aug == 'large':
    if split == 'train':
      return [transforms.RandomHorizontalFlip(0.5),
              transforms.Resize(224),
              ]
    else:
      # center crop down imagenet
      return [transforms.Resize(224)]
  else:
    imsize = imsize if imsize is not None else 32
    if split == 'train':
        train_transform = []
      #return [transforms.RandomCrop(imsize, padding=round(imsize / 8))]
        train_transform.append(transforms.RandomCrop(32, padding=4))
        train_transform.append(transforms.RandomHorizontalFlip())
        return train_transform
    else:
      return [transforms.Resize(imsize), transforms.CenterCrop(imsize)]


def get_transform(dataset_name, split, normalize=None, transform=None, imsize=None, aug='large'):
  if transform is None:
    if normalize is None:
        if aug == 'large':
          if 'cifar100' in dataset_name:
            normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
          else:
            # imagenet
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
          if 'cifar10' in dataset_name:
            normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
          if 'cifar100' in dataset_name:
            normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

    transform = transforms.Compose(get_aug(split, imsize=imsize, aug=aug)
                                   + [transforms.ToTensor(), normalize])
  return transform

# warning validation set and test set is the same thing in this implementation

def get_cifar10(dataset_name, data_dir, split, transform=None, imsize=None, bucket='pytorch-data', **kwargs):
  transform = get_transform(dataset_name, split, transform=transform, imsize=imsize, aug='small')
  return datasets.CIFAR10(data_dir, train=(split=='train'), transform=transform, download=True, **kwargs)

def get_cifar100(dataset_name, data_dir, split, transform=None, imsize=None, bucket='pytorch-data', **kwargs):
  transform = get_transform(dataset_name, split, transform=transform, imsize=imsize, aug='small')
  return datasets.CIFAR100(data_dir, train=(split=='train'), transform=transform, download=True,)
