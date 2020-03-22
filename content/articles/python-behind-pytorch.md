Title: Native Python Features Behind PyTorch
Date: 2020-03-22 10:00
Modified: 2020-03-22 10:00
Category: python
Slug: python-behind-pytorch
Summary: Learn about the advanced python native features that powers PyTorch
Status: draft
Authors: Amit Chaudhary

PyTorch has emerged as one of the go to deep learning frameworks in recent years. This popularity can be attributed to its easy to use API and it being more "pythonic".

PyTorch leverages numerous native features of Python to give us a consistent and clean API. In this article, I will explain those native features in detail. Learning these will help you better understand why you do things certain way in PyTorch and become a better user.

- Imports
- Callable classes
- Magic functions: __call__, __getitem__, __len__
- Generators: Dataset, DataLoader


## Callable Classes
You know the various layers such as ```nn.Linear()``` that we combine in PyTorch to build our models. You import the layer and apply them on tensors.
```python
import torch
import torch.nn as nn

x = torch.rand(1, 784)
layer = nn.Linear(784, 10)
output = layer(x)
```
Here we are able to call layer on some tensor `x`, so it must be a function right? Is `nn.Linear()` returning a function? Let's check the type.
```python
>>> type(layer)
<class 'torch.nn.modules.linear.Linear'>
```
Surprise! `nn.Linear` is actually a class and layer an object of that class.  

> "How the heck could we call it then? Aren't only functions supposed to be callable?"

Nope. You can create callable objects as well. Python provides a native way to make objects created from classes callable by using magic functions. Let's see an example:
```python
class Double(object):
    def __call__(self, x):
        return 2*x
```

Here we add a magic method `__call__` in the class which double any number passed to it. Now, you can create an object out of this class and call it on some number.
```python
>>> d = Double()
>>> d(2)
4
```
Alternatively, the above code can be combined in the single line itself.
```python
>>> Double()(2)
4
```

