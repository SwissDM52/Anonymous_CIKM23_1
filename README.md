# Anonymous_AAAI24_1

### The codes for reviewers.

- dataset.py ------- load a dataset
- enc.py -------  PKE codes and loss function.
- model.py -------  a function for getting a model 
- resnet.py ------- model structure for cifar
- resnet_mnist.py ------- model structure for mnist
- tester.py ------- deploying process
- train.py ------- main function is here.
- train_EncryIP_incrorect.py ------- evaluate EncryIP with an  incrorect secrest key.
- trainer.py ------- learning process

Train original default (dataset : cifar10 , original) :
python train.py

Train random (dataset : cifar10 , p : 11) :
python train.py --random

Train fake (dataset : cifar10 , p : 11) :
python train.py --fake

Train EncryIP cifar100 (dataset:cifar100 , p : 11):
python train.py --EncryIP --dataset=cifar100

Change number p (p must be a prime number):
python train.py --EncryIP --dataset=cifar100 --p=23

Change number p (p must be a prime number):
python train.py --EncryIP --dataset=cifar100 --p=23

