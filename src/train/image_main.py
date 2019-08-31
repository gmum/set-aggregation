from argparse import ArgumentParser

import tensorflow as tf

from src.metrics.monitors import Monitor
from src.models.model import Model, MultiImageModel
from src.train.image_parsing_utils import *


def prepare_parser():
    parser = ArgumentParser()
    parser.add_argument("--datatype", default="cifar", choices=["cifar","fashion-mnist"], help="which dataset to use")
    parser.add_argument("--data-provider-type", default='default', choices=["default"],
                        help="how to preprocess data before feeding them to network. For now, only deafult (normal processing)"
                             "is available.")
    parser.add_argument("--output-dim", type=int, default=10, help="the dimension of the output of the network")
    parser.add_argument("--batch-size", type=int, default=50, help="the size of one batch")
    parser.add_argument("--epochs", type=int, default=40,
                        help="number of epochs.")
    parser.add_argument("--head-hidden-dim", type=int, default=512,
                        help="The dimension of hidden layers in head architecture")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout probability used in head architecture")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="learning step in gradient optimization algorithms")
    parser.add_argument("--folder", default=".",
                        help="the folder where to store outputs, losses and configuration files")
    parser.add_argument("--restore-file", type=str, help="if set, the model will be loaded from the provided file to "
                                                         "count the whole dataset loss and accuracy")
    parser.add_argument("--print-params", action="store_true", help="whether to print the parameters")
    parser.add_argument("--valid-ratio", type=float, default=1.0/12,
                        help="how much of the dataset should form the valid dataset. Note that if the dataset "
                             "comes with a set partiion into train, valid, test (or train and test) all the "
                             "datasets are joined together to from a whole corpus, from which the ratio is computed")
    parser.add_argument("--test-ratio", type=float, default=1.0/12,
                        help="how much of the dataset should form the test dataset. See notes on valid-ratio "
                             "for further explainations.")
    parser.add_argument("--l2", type=float, default=3.0, help="if non-zero, specifies the L2-regularization parameter")
    parser.add_argument("--config-name", default="config.txt", help="name of the file where the configuration is set")
    parser.add_argument("--model-name", default="model.ckpt", help="name of the file where the model is saved.")
    parser.add_argument("--body-type", default="cnn-max", choices=["cnn","cnn-max", "cnn-set","cnn-1x1","cnn-avg"],
                        help="which type of body (architecture applied after embedding and before the head)"
                             " will be used in the model.\n * 'cnn-max' for classical architecture with global max pooling."
                             "'cnn-set' for SAN instead of global max pooling, 'cnn-1x1' for 1x1 convolutions on feature maps. 'cnn-avg' for global average pooling.")
    parser.add_argument("--projection-dim", type=int, default=512,
                        help="if 'cnn-set' architecture is used, specifies the projection dimension")
    parser.add_argument("--shuffle", action="store_true", help="whether to shuffle the dataset before partition into"
                                                               "train, valid and test")
    parser.add_argument("--batch-norm", action="store_true", help="whether to use batch-norm")
    parser.add_argument("--random-state", type=int, default=10, help="the random state used for datasets preparations")
    parser.add_argument("--head-layers", type=int, default=0, help="how many hidden FC layers to use in head")
    parser.add_argument("--reduce-type", default="mean", choices=["mean","sum"], help="how to reduce in the SAN layer")

    parser.add_argument("--multi-image", action="store_true", help="whether to train the net on multiple sizes")

    parser.add_argument("--shapes-nr", type=int, default=5, choices=[3,5], help="how many shapes to use")
    parser.add_argument("--multi-train-mode", default="all", choices=["single","all","every-second"],
                        help="which mode to use in training. Note that all the networks will be validated for all shapes.")

    return parser


def main(args):
    tf.reset_default_graph()
    

    loader = get_loader(args)
    provider = get_provider(args, loader)
    is_training = tf.placeholder(tf.bool, name="is_training_2")
    body_builder = get_body_builder(args, is_training)
    head_builder = get_head_builder(args, is_training)
    monitor = Monitor(args.folder)
    scopes, trainable_scopes = get_scopes()
    ph_builder = get_placeholder_builder(args)
    monitor.save_args(args, args.config_name)
    lr_scheduler = get_lr_scheduler(args)
    if not args.multi_image:
        model = Model(DataProvider=provider, BodyBuilder=body_builder, HeadBuilder=head_builder,
                      Monitor=monitor, scopes=scopes, trainable_scopes=trainable_scopes, PlaceholderBuilder=ph_builder,
                      is_training=is_training, learning_rate=args.learning_rate, lr_scheduler=lr_scheduler)
    else:
        shapes = get_shapes(args)
        train_mode = get_mode(args.multi_train_mode,shapes)
        model = MultiImageModel(DataProvider=provider,BodyBuilder=body_builder,HeadBuilder=head_builder,
                                MultiImagePlaceholderBuilder=ph_builder,Monitor=monitor,scopes=scopes,
                                trainable_scopes=trainable_scopes,shapes=shapes,is_training=is_training,
                                learning_rate=args.learning_rate, train_mode=train_mode, lr_scheduler=lr_scheduler)
    start_session(args, model)


def start_session(args, model):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:
        model.set_up()
        saver = tf.train.Saver()
        if args.restore_file is None:
            session.run(tf.global_variables_initializer())
            model.train(session, args.epochs, args.batch_size, args, saver, args.config_name, args.model_name)
            model.predict_dataset(session, args.batch_size, "train", args)
            model.predict_dataset(session, args.batch_size, "test", args)
            model.predict_dataset(session, args.batch_size, "valid", args)
        else:
            saver.restore(session, args.restore_file)
            model.predict_dataset(session, args.batch_size, "train", args)
            model.predict_dataset(session, args.batch_size, "test", args)
            model.predict_dataset(session, args.batch_size, "valid", args)


def get_scopes():
    scopes = {}
    scopes["head"] = "trainable"
    scopes["body"] = "trainable"
    trainable_scopes = ["trainable"]
    return scopes, trainable_scopes


if __name__ == "__main__":
    args = prepare_parser().parse_args()
    main(args)
