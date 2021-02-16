# Xformer_Pytorch

## Table of Contents
- [Introduction](#Introduction)
- [Example](#Example)
- [FAQ](#FAQ)

## Introduction
According to ([2009.06732](https://arxiv.org/abs/2009.06732)), many Transformer-based models by the name of x-formers have been proprosed to improve the efficiency during inference, especially by means of reducing the inference time. The main focus of most x-formers is the modification of the attention modules since the dot-product operation in an attention module makes the square computation bottleneck of a Transformer model. In this repository, we implement the Xformer attention module following the ideas from [Performer](https://arxiv.org/abs/2009.14794) and [Linformer](https://arxiv.org/abs/2006.04768). The module has been made compatible with [fairseq](https://github.com/pytorch/fairseq).
### Xformer architecture
We keep the original attention operation while modifying the projection matrices of Q,K and V. The projection matrices for K and V share the same weights and pool the sequence-length dimension of the input sequence by <em>$\beta$</em> (> 1). The projection matrix for Q pools the input sequence's hidden dimension by <em>$\alpha$</em> (> 1). Also, K is projected from Q instead of the input sequence of the attention module. Denote **n** as input sequence length and **d** as sequnce length. The computation complexity of attention probability is reduced from <em>O(n<sup>2</sup>d)</em> to <em>O(n<sup>2</sup>d/($\alpha\beta$))</em>, and that of the dot-product of attention probability with V is reduced from <em>O(n<sup>2</sup>d)</em> to <em>O(n<sup>2</sup>d/$\beta$)</em>.
The performance of the baseline Transformer model on the IWSLT 14 DE-EN translation task is and our Xformer's performance is 32.09 bleu score while achieving substantial acceleration.
### Inference time comparison
The experiment is to measure the total inference time of the baseline and the proposed attention modules on an input tensor of shape (sequence length, batchsize, hidden dimension), while batchsize and hidden dimension are set 40 and 512 respectively, we vary the sequence length as {128, 256, 512, 1024, 2048} ,and $\alpha$ and $\beta$ as {2, 4, 8, 16}. The numbers in the table are inference time in second.
| Sequence length\$\alpha$$\beta$        | Are           | Cool  |
| ------------- |:-------------:| -----:|
| 128        | 0      | $1600 |
| 256        | 0      |   $12 |
| 512      | 0      |    $1 |
## Example

## FAQ
