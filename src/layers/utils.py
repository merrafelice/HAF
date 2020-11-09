import re

from src.layers.saliency import SaliencyLayer
from tensorflow.keras import Model


def summarize_feature_maps(resnet50):
    for i in range(len(resnet50.layers)):
        layer = resnet50.layers[i]
        # check for convolutional layer
        # if 'conv' not in layer.name:
        if re.match('.*conv[0-9]_block[0-9].*', layer.name):
            continue
        # summarize output shape
        print(i, layer.name, layer.output.shape)


def insert_saliency_layers(model, list_layer_regex, layer_names=None, position='after'):
    """
    Code inspired by https://www.xspdf.com/help/52662515.html
    """
    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Set the input layers of each layer
    for layer in model.layers[:-1]:
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                    {layer_name: [layer.name]})
            else:
                network_dict['input_layers_of'][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update(
        {model.layers[0].name: model.input})

    # Iterate over all layers after the input
    model_outputs = []
    i = 0

    for layer in model.layers[1:]:

        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux]
                       for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # Insert layer if name matches the regular expression

        if i < len(list_layer_regex) and re.match(list_layer_regex[i], layer.name):

            if position == 'replace':
                x = layer_input
            elif position == 'after':
                x = layer(layer_input)
            elif position == 'before':
                pass
            else:
                raise ValueError('position must be: before, after or replace')

            new_layer = SaliencyLayer()
            if layer_names[i]:
                new_layer._init_set_name(layer_names[i])
            else:
                new_layer._init_set_name('{}_{}'.format(layer.name,
                                                        new_layer.name))
            x = new_layer(x)

            print('New layer added: {} Type: {}'.format(layer.name, position))
            i += 1
            if position == 'before':
                x = layer(x)
        else:
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

        # Save tensor in output list if it is output in initial model
        # if layer_name in model.output_names:

    model_outputs.append(x)

    return Model(inputs=model.inputs, outputs=model_outputs)
