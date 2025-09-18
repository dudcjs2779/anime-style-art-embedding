import argparse
import toml
import yaml
from collections import OrderedDict

def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

def read_config(args: argparse.Namespace, parser: argparse.ArgumentParser):
    if not args.config:
        return args
    
    config_path = args.config

    with open(config_path, "r", encoding="utf-8") as f:
        config = toml.load(f)

    config_dict = {}
    for section_name, section_dict in config.items():
        if not isinstance(section_dict, dict):
            config_dict[section_name] = section_dict
            continue

        if section_name in ['model_type']:
            config_dict[section_name] = section_dict
            continue

        for key, value in section_dict.items():
            config_dict[key] = value

    config_args = argparse.Namespace(**config_dict)
    args = parser.parse_args(namespace=config_args)

    # with open(args.model_config, mode='r') as f:
    #     opt = yaml.load(f, Loader=ordered_yaml()[0])

    # opt['is_train'] = True
    # args.opt = opt


    return args

def read_config_ipynb(parser, config_path):
    args = parser.parse_args([])

    with open(config_path, "r", encoding="utf-8") as f:
        config = toml.load(f)
    
    config_dict = {}
    for section_name, section_dict in config.items():
        if not isinstance(section_dict, dict):
            config_dict[section_name] = section_dict
            continue
        
        if section_name in ['model_type']:
            config_dict[section_name] = section_dict
            continue

        for key, value in section_dict.items():
            config_dict[key] = value

    for key, value in config_dict.items():
        setattr(args, key, value)
        # if hasattr(args, key):
        #     setattr(args, key, value)
    
    return args


def print_nested_dict(d, indent=0):
    """
    dict 계층구조를 탐색하며 구조를 출력하는 코드
    """
    for key, value in d.items():
        if isinstance(value, dict):
            print("    " * indent + f"- {key}")
            print_nested_dict(value, indent + 1)
        else:
            print("    " * indent + f"- {key}: {value}")
