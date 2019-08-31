from src.data_handling.loaders import CifarDataLoader, FashionMnistDataLoader
from src.data_handling.providers import DataProvider, MultiSizeDataProvider
from src.models.body_builders import BaselineCNNBodyBuilder, SetCNNBodyBuilder, Conv1x1CNNBodyBuilder
from src.models.head_builders import SimpleCrossEntropyHeadBuilder
from src.models.placeholders_builders import ImagePlaceholderBuilder, MultiImagePlaceholderBuilder
from src.models.regularizers import L2RegularizerProvider
import numpy as np

def get_provider(args, loader):
    if args.data_provider_type == "default" and not args.multi_image:
        provider = DataProvider(DataLoader=loader)
    elif args.multi_image:
        shapes = get_shapes(args)
        provider = MultiSizeDataProvider(DataLoader=loader,shapes=shapes)
    elif args.data_provider_type == "multi-default":
        assert args.multi_image, "both the multi default provider and the multi images flag must be set"
        shapes = get_shapes(args)
        provider = MultiSizeDataProvider(DataLoader=loader,shapes=shapes)
    else:
        raise ValueError("Unknown provider type")
    return provider

def get_lr_scheduler(args):
    # not implemented
    return None

def get_mode(mode_desc,shapes):
    if mode_desc == "all":
        return None
    elif mode_desc == "single":
        if len(shapes) % 2 == 0:
            return [len(shapes)//2]
        else:
            return [len(shapes)//2+1]
    elif mode_desc == "every-second":
        return np.arange(0,len(shapes),2,dtype=np.int64)
    else:
        raise ValueError("unknown mode description {}".format(mode_desc))

def get_shapes(args):
    if args.datatype == "fashion-mnist":
        if args.shapes_nr == 3:
            return [12, 28, 44]
        elif args.shapes_nr == 5:
            return [12, 20, 28, 36, 44]
        else:
            raise ValueError("unknwon number of shapes")
    elif args.datatype == "cifar":
        if args.shapes_nr == 3:
            return [16, 32, 48]
        elif args.shapes_nr == 5:
            return [16, 24, 32, 40, 48]
        else:
            raise ValueError("unknown number of shapes")
    else:
        raise ValueError("unknown datatype {}".format(args.datatype))


def get_loader(args):
    if args.datatype == "cifar":
        if args.multi_image:
                loader = CifarDataLoader(valid_ratio=args.valid_ratio, test_ratio=args.test_ratio,raw=True)
        else:
                loader = CifarDataLoader(valid_ratio=args.valid_ratio, test_ratio=args.test_ratio)
    elif args.datatype == "fashion-mnist":
        if args.multi_image:
                 loader = FashionMnistDataLoader(valid_ratio=args.valid_ratio, test_ratio=args.test_ratio,raw=True)
        else:  
                 loader = FashionMnistDataLoader(valid_ratio=args.valid_ratio, test_ratio=args.test_ratio)
    else:
        raise ValueError("Unknwon dataset name")
    return loader


def get_placeholder_builder(args):
    if not args.multi_image:
        if args.datatype == "fashion-mnist":
             return ImagePlaceholderBuilder(width=28,height=28,channels=1)
        elif args.datatype == "cifar":
             return ImagePlaceholderBuilder(width=32,height=32,channels=3)
        else:
            raise ValueError("unknown datatype {}".format(args.datatype))
    else:
        shapes = get_shapes(args)
        if args.datatype == "fashion-mnist":
            return MultiImagePlaceholderBuilder(shapes=shapes, channels=1)
        elif args.datatype == "cifar":
            return MultiImagePlaceholderBuilder(shapes=shapes, channels=3)
        else:
            raise ValueError("unknown datatype {}".format(args.datatype))

def get_head_builder(args,is_training):
    regularizer_provider = L2RegularizerProvider(args.l2)
    print(args.dropout, args.batch_norm)
    return SimpleCrossEntropyHeadBuilder(args.output_dim, args.head_hidden_dim, regularizer_provider, batch_norm=args.batch_norm, dropout=args.dropout,
                                         layers=args.head_layers, is_training=is_training)


def get_body_builder(args, is_training):
    regularizer_provider = L2RegularizerProvider(args.l2)
    if args.body_type == "cnn":
        return BaselineCNNBodyBuilder(kernel_sizes=[3, 3, 3], filter_nums=[32, 64, 64], max_poolings=[True, True, False],
                                      regularizer_provider=regularizer_provider, pooling="flatten",
                                      batch_norm=args.batch_norm, dropout=args.dropout, is_training=is_training)
    elif args.body_type == "cnn-max":
        return BaselineCNNBodyBuilder(kernel_sizes=[3, 3, 3], filter_nums=[32, 64, 64],
                                      max_poolings=[True, True, False],
                                      regularizer_provider=regularizer_provider, pooling="max",
                                      batch_norm=args.batch_norm, dropout=args.dropout,is_training=is_training)
    elif args.body_type == "cnn-avg":
        return BaselineCNNBodyBuilder(kernel_sizes=[3, 3, 3], filter_nums=[32, 64, 64],
                                      max_poolings=[True, True, False],
                                      regularizer_provider=regularizer_provider, pooling="avg",
                                      batch_norm=args.batch_norm, dropout=args.dropout,is_training=is_training)
    elif args.body_type == "cnn-set":
        return SetCNNBodyBuilder(kernel_sizes=[3, 3, 3], filter_nums=[32, 64, 64], max_poolings=[True, True, False],
                                 projection_dim=args.projection_dim, regularizer_provider=regularizer_provider,
                                 batch_norm=args.batch_norm, dropout=args.dropout,is_training=is_training, reduce_type = args.reduce_type)
    elif args.body_type == "cnn-1x1":
        return Conv1x1CNNBodyBuilder(kernel_sizes=[3, 3, 3], filter_nums=[32, 64, 64], max_poolings=[True, True, False],
                                     regularizer_provider=regularizer_provider, batch_norm=args.batch_norm,
                                     dropout=args.dropout,is_training=is_training)
    else:
        raise ValueError("Unknown body type")
