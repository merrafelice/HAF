import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf

from bin.utils.IO import read_config
from bin.utils.parser import train_parse_args
from src.dataset.data_loader import CustomDataLoader
from src.cnn.hafresnet50 import HAFResNet50Model


def name_dir(arg_name, args):
    return arg_name.format(args.dataset, args.epochs, args.lr, args.batch_size, '_sc' if args.loss_sc == 1 else '_full',
                           'after' if args.after == 1 else 'before', args.reg)


def change_activation(model):
    assert model.layers[-1].activation == tf.keras.activations.softmax

    config = model.layers[-1].get_config()
    weights = [x.numpy() for x in model.layers[-1].weights]

    config['activation'] = tf.keras.activations.linear
    config['name'] = 'logits'

    new_layer = tf.keras.layers.Dense(**config)(model.layers[-2].output)
    new_model = tf.keras.Model(inputs=[model.input], outputs=[new_layer])
    new_model.layers[-1].set_weights(weights)

    assert new_model.layers[-1].activation == tf.keras.activations.linear
    return new_model


def train_haf():
    args = train_parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print('Device gpu: {0}'.format(os.environ['CUDA_VISIBLE_DEVICES']))

    print(train_parse_args())

    train_dir, _, path_classes, path_weights, path_saved_smaps = read_config(
        sections_fields=[('ORIGINAL', 'Images'),
                         ('ORIGINAL', 'Features'),
                         ('ALL', 'ImagenetClasses'),
                         ('OUTPUT', 'Weights'),
                         ('OUTPUT', 'Saliency')])

    train_dir, path_weights, path_saved_smaps = train_dir.format(args.dataset), name_dir(path_weights, args), name_dir(
        path_saved_smaps, args)

    # Load Baseline CNN (e.g., ResNet50)

    resnet50 = tf.keras.applications.ResNet50(
        include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000
    )

    resnet50 = change_activation(resnet50)
    # model = resnet_50()
    # model.build(input_shape=(None, 224, 224, 3))
    # model.summary()

    # Read The Data

    loader = CustomDataLoader(train_dir=train_dir, image_size=(224, 224), batch_size=args.batch_size,
                              window=args.window, resnet50=resnet50)

    print('Start Image Loading...')
    # loader.load(resnet50)
    loader.load_files()
    print('End Image Loading!')

    # HAF Creation
    haf_model = HAFResNet50Model(resnet50, args.loss_sc, args.reg)

    ## Make the base network (resnet50) untrainable
    haf_model.make_base_model_untrainable()

    ## Insert Layers
    # new_layers = ['.*conv2_block1_out.*', '.*conv2_block2_out.*', '.*conv2_block3_out*.']

    new_layers = ['.*conv2_block3_add.*', '.*conv3_block4_add.*', '.*conv4_block6_add*.', '.*conv5_block3_add*.']
    new_layers = ['.*conv2_block3_add.*','.*conv3_block3_add.*', '.*conv4_block5_add*.', '.*conv5_block3_add*.']
    haf_model.insert_saliency_layers(new_layers[-1:], 1)

    haf_model.train(loader, lr=args.lr, epochs=args.epochs, train_dir=train_dir, path_saved_smaps=path_saved_smaps)


if __name__ == '__main__':
    train_haf()
