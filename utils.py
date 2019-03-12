from __future__ import print_function

def print_layers(layers, list_layers):
    count = 1
    for layer in list_layers:
        print('')
        print('========================== Layer{} =========================='.format(count))
        print(layers[layer].name)
        print('-'*10)
        print(list(layers[layer].shape)[1:])
        count += 1
    print('')
