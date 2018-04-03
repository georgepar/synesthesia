# Correlational Neural Networks

Implementation of the neural network architecture described in [https://arxiv.org/abs/1504.07225](https://arxiv.org/abs/1504.07225).

This architecture is an extension of Multimodal Autoencoders that expects the vector representations of two modalities as inputs and:

1. Creates a common representation of the two modalities (fusion)
2. Reconstructs one modality given the other (crossmodal mapping)

Demo available in the `corrnet.ipynb` notebook.
