def count_parameters(layers: list):
    """
    Count the total number of trainable parameters in a neural network.

    Args:
        layers: A list of dictionaries describing each layer.
                Each dict contains 'type' and layer-specific parameters.

    Returns:
        Total number of trainable parameters as an integer.
    """

    params = 0
    for layer in layers:
        if layer['type'] == 'dense':
            params += layer['input_size'] * layer['output_size']
            if layer['use_bias']:
                params += layer['output_size']
        elif layer['type'] == 'conv2d':
            params += (
                layer['in_channels']
                * layer['out_channels']
                * (
                    layer['kernel_size'] ** 2
                    if isinstance(layer['kernel_size'], int)
                    else layer['kernel_size'][0] * layer['kernel_size'][1]
                )
            )
            if layer['use_bias']:
                params += layer['out_channels']
        elif layer['type'] == 'embedding':
            params += layer['num_embeddings'] * layer['embedding_dim']
    return params
