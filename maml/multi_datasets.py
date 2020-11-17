import torch.nn.functional as F

from collections import namedtuple
from torchmeta.datasets import Omniglot, MiniImagenet, CUB, DoubleMNIST, TripleMNIST
from torchmeta.toy import Sinusoid
from torchmeta.transforms import ClassSplitter, Categorical, Rotation
from torchvision.transforms import ToTensor, Resize, Compose

from maml.model import ModelConvOmniglot, ModelConvMiniImagenet, ModelMLPSinusoid
from maml.utils import ToTensor1D

Benchmark = namedtuple('Benchmark', 'meta_train_datasets meta_val_datasets '
                                    'meta_test_datasets model loss_function')

def get_benchmark_by_name_1(name,
                          folder,
                          num_ways,
                          num_shots,
                          num_shots_test,
                          hidden_size=None):
    dataset_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=num_shots,
                                      num_test_per_class=num_shots_test)


    meta_train_datasets = []
    meta_val_datasets = []
    meta_test_datasets = []

    if 'miniimagenet' in name: 
      model = ModelConvMiniImagenet(num_ways, hidden_size=hidden_size)
    else: 
      model = ModelConvOmniglot(num_ways, hidden_size=hidden_size)
    loss_function = F.cross_entropy

    for nm in name: 
      if nm == 'miniimagenet':
          transform = Compose([Resize(84), ToTensor()])

          meta_train_datasets.append(MiniImagenet(folder,
                                            transform=transform,
                                            target_transform=Categorical(num_ways),
                                            num_classes_per_task=num_ways,
                                            meta_train=True,
                                            dataset_transform=dataset_transform,
                                            download=True))
          meta_val_datasets.append(MiniImagenet(folder,
                                          transform=transform,
                                          target_transform=Categorical(num_ways),
                                          num_classes_per_task=num_ways,
                                          meta_val=True,
                                          dataset_transform=dataset_transform))
          meta_test_datasets.append(MiniImagenet(folder,
                                           transform=transform,
                                           target_transform=Categorical(num_ways),
                                           num_classes_per_task=num_ways,
                                           meta_test=True,
                                           dataset_transform=dataset_transform))

      elif nm == 'cub':
          transform = Compose([Resize(84), ToTensor()])

          meta_train_datasets.append(CUB(folder,
                                        transform=transform,
                                        target_transform=Categorical(num_ways),
                                        num_classes_per_task=num_ways,
                                        meta_train=True,
                                        dataset_transform=dataset_transform,
                                        download=True))
          meta_val_datasets.append(CUB(folder,
                                      transform=transform,
                                      target_transform=Categorical(num_ways),
                                      num_classes_per_task=num_ways,
                                      meta_val=True,
                                      dataset_transform=dataset_transform))
          meta_test_datasets.append(CUB(folder,
                                       transform=transform,
                                       target_transform=Categorical(num_ways),
                                       num_classes_per_task=num_ways,
                                       meta_test=True,
                                       dataset_transform=dataset_transform))
      elif nm == 'omniglot':
        class_augmentations = [Rotation([90, 180, 270])]
        transform = Compose([Resize(64), ToTensor()])

        meta_train_datasets.append(Omniglot(folder,
                                      transform=transform,
                                      target_transform=Categorical(num_ways),
                                      num_classes_per_task=num_ways,
                                      meta_train=True,
                                      class_augmentations=class_augmentations,
                                      dataset_transform=dataset_transform,
                                      download=True))
        meta_val_datasets.append(Omniglot(folder,
                                    transform=transform,
                                    target_transform=Categorical(num_ways),
                                    num_classes_per_task=num_ways,
                                    meta_val=True,
                                    class_augmentations=class_augmentations,
                                    dataset_transform=dataset_transform))
        meta_test_datasets.append(Omniglot(folder,
                                     transform=transform,
                                     target_transform=Categorical(num_ways),
                                     num_classes_per_task=num_ways,
                                     meta_test=True,
                                     dataset_transform=dataset_transform))
      elif nm == 'doublemnist':
          transform = Compose([Resize(64), ToTensor()])

          meta_train_datasets.append(DoubleMNIST(folder,
                                        transform=transform,
                                        target_transform=Categorical(num_ways),
                                        num_classes_per_task=num_ways,
                                        meta_train=True,
                                        dataset_transform=dataset_transform,
                                        download=True))
          meta_val_datasets.append(DoubleMNIST(folder,
                                      transform=transform,
                                      target_transform=Categorical(num_ways),
                                      num_classes_per_task=num_ways,
                                      meta_val=True,
                                      dataset_transform=dataset_transform))
          meta_test_datasets.append(DoubleMNIST(folder,
                                       transform=transform,
                                       target_transform=Categorical(num_ways),
                                       num_classes_per_task=num_ways,
                                       meta_test=True,
                                       dataset_transform=dataset_transform))    
      elif nm == 'triplemnist':
          transform = Compose([Resize(64), ToTensor()])

          meta_train_datasets.append(TripleMNIST(folder,
                                        transform=transform,
                                        target_transform=Categorical(num_ways),
                                        num_classes_per_task=num_ways,
                                        meta_train=True,
                                        dataset_transform=dataset_transform,
                                        download=True))
          meta_val_datasets.append(TripleMNIST(folder,
                                      transform=transform,
                                      target_transform=Categorical(num_ways),
                                      num_classes_per_task=num_ways,
                                      meta_val=True,
                                      dataset_transform=dataset_transform))
          meta_test_datasets.append(TripleMNIST(folder,
                                       transform=transform,
                                       target_transform=Categorical(num_ways),
                                       num_classes_per_task=num_ways,
                                       meta_test=True,
                                       dataset_transform=dataset_transform))
      else:
          raise NotImplementedError('Unknown dataset `{0}`.'.format(name))

    return Benchmark(meta_train_datasets=meta_train_datasets,
                     meta_val_datasets=meta_val_datasets,
                     meta_test_datasets=meta_test_datasets,
                     model=model,
                     loss_function=loss_function)
