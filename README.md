# Xformer_Pytorch

## Table of Contents
- [Introduction](#Introduction)
- [Usage](#Usage)
- [FAQ](#FAQ)

## Introduction
According to ([2009.06732](https://arxiv.org/abs/2009.06732)), many Transformer-based models by the name of x-formers have been proprosed to improve the efficiency during inference, especially by means of reducing the inference time. The main focus of most x-formers is the modification of the attention modules since the dot-product operation in an attention module makes the square computation bottleneck of a Transformer model. In this repository, we implement the Xformer attention module following the ideas from [Performer](https://arxiv.org/abs/2009.14794) and [Linformer](https://arxiv.org/abs/2006.04768). The module has been made compatible with [fairseq](https://github.com/pytorch/fairseq).
### Xformer architecture
We keep the original attention operation while modifying the projection matrices of Q,K and V. The projection matrices for K and V share the same weights and pool the sequence-length dimension of the input sequence by <em>β</em> (> 1). The projection matrix for Q pools the input sequence's hidden dimension by <em>α</em> (> 1). Also, K is projected from Q instead of the input sequence of the attention module. Denote **n** as input sequence length and **d** as sequnce length. The computation complexity of attention probability is reduced from <em>O(n<sup>2</sup>d)</em> to <em>O(n<sup>2</sup>d/(αβ))</em>, and that of the dot-product of attention probability with V is reduced from <em>O(n<sup>2</sup>d)</em> to <em>O(n<sup>2</sup>d/β)</em>.
The performance of the baseline Transformer model on the IWSLT 14 DE-EN translation task is **34.20** bleu score and our Xformer's performance is **32.09** bleu score while achieving substantial acceleration.
### Inference time comparison
The experiment is to measure the total inference time of the baseline and the proposed attention modules on an input tensor of shape (sequence length, batchsize, hidden dimension), while batchsize and hidden dimension are set 10 and 512 respectively, we vary the sequence length as {128, 256, 512, 1024, 2048, 4096}, and α and β as {2, 4, 8, 16}. 

The numbers in the table are inference time  in seconds.
| Sequence length \\ αβ | 2 | 4  |8  |16  |baseline  |
| ------------- |:-------------:| :-----:|:-----:|:-----:|:-----:|
| 128        | 0.82 | 0.84 | 0.83 | 0.83 | 1.05 |
| 256        | 1.55 | 1.13 | 1.00 | 0.91 | 2.44 |
| 512        | 3.41 | 2.30 | 1.83 | 1.62 | 5.43 |
| 1024       | 8.84 | 5.70 | 4.05 | 3.36 | 14.82|
| 2048       |26.33 | 15.41| 10.45| 7.72 | 50.29|
| 4096       |93.78 | 47.57| 28.42| 20.03|206.27|

Speedup with respect to the baseline attention module:
| Sequence length \\ αβ | 2 | 4  |8  |16  |
| ------------- |:-------------:| :-----:|:-----:|:-----:|
| 128        | 21.91 % | 20.01 % | 20.96 % | 20.96 % |
| 256        | 36.48 % | 53.69 % | 59.02 % | 62.71 % |
| 512        | 37.21 % | 57.65 % | 66.30 % | 70.17 % |
| 1024       | 40.36 % | 61.54 % | 72.68 % | 77.33 % |
| 2048       | 47.65 % | 69.36 % | 79.23 % | 84.65 % |
| 4096       | 54.54 % | 76.94 % | 86.23 % | 90.29 % |
## Usage
For using the Xformer attention module, first do
```
git clone https://github.com/YNNEKUW/Xformer_Pytorch.git
pip install -r Xformer_Pytorch/requirement.txt
mv Xformer_Pytorch/xformer_pytorch .
```
and then
```python
from xformer_pytorch import Xformer
```
### Example
```python
import torch
from fairseq.modules.multihead_attention import MultiheadAttention
from xformer_pytorch import Xformer


hidden_dim = 512
n_heads = 4
batch_size = 40
length = 1024

baseline_attn = MultiheadAttention(hidden_dim, n_heads, self_attention=True).cuda()
test_input = torch.ones((length, batch_size, hidden_dim)).cuda()
dummy_out = baseline_attn(test_input, test_input, test_input)

# To use less hyperparameters, we let scalar = alpha = beta here.
scalar = 2
xformer_attn = Xformer(hidden_dim, n_heads, max_seq_len=length, scalar=scalar).cuda()
output = xformer_attn(test_input)
```

## FAQ
