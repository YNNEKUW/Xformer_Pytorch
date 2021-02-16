# Xformer_Pytorch

## Table of Contents
- [Introduction](#Introduction)
- [Example](#Example)
- [FAQ](#FAQ)

## Introduction
According to ([2009.06732](https://arxiv.org/abs/2009.06732)) , many Transformer-based models by the name of x-formers have been proprosed to improve the efficiency during inference, especially by means of reducing the inference time. The main focus of most x-formers is the modification of the attention modules since the dot-product operation in an attention module makes the square computation bottleneck of a Transformer model. In this repository, we implement the Xformer attention module combining the ideas from [Performer](https://arxiv.org/abs/2009.14794) and [Linformer](https://arxiv.org/abs/2006.04768). The module has been made compatible with [fairseq](https://github.com/pytorch/fairseq).
### Xformer architecture
We keep the original attention operation while modifying the projection matrices of Q,K and V. The projection matrices for K and V share the same weights and pool the sequence-length dimension of the input sequence by <em>beta</em> (> 1). The projection matrix for Q pools the input sequence's hidden dimension by <em>alpha</em> (> 1). 
### Inference time comparison

## Example

## FAQ
