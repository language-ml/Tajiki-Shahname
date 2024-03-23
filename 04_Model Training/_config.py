import json
from os import PathLike
from pathlib import Path
from typing import Any, Union, Optional, Literal
import yaml

class Config(object):
    def __init__(self, data: dict, base_path: str):
        self._write_mode = True
        self._base_path = base_path

        for key, val in data.items():
            if isinstance(val, (list, tuple)):
                generator = (self.__parse_value(item) for item in val)
                setattr(self, key, tuple(generator))
            else:
                setattr(self, key, self.__parse_value(val))

        delattr(self, '_base_path')
        delattr(self, '_write_mode')

    def __parse_value(self, value: Any):
        if isinstance(value, dict):
            return self.__class__(value, self._base_path)

        if isinstance(value, str):
            if value.startswith('path:'):
                value = value[len('path:'):]
                value = str((Path(self._base_path) / value).absolute())

        return value

    def __setattr__(self, key, value):
        if key == '_write_mode' or hasattr(self, '_write_mode'):
            super().__setattr__(key, value)
        else:
            raise Exception('Set config')

    def __delattr__(self, item):
        if item == '_write_mode' or hasattr(self, '_write_mode'):
            super().__delattr__(item)
        else:
            raise Exception('Del config')

    def __contains__(self, name):
        return name in self.__dict__

    def __getitem__(self, name):
        return self.__dict__[name]

    def __repr__(self):
        return repr(self.to_dict())

    @staticmethod
    def __item_to_dict(val):
        if isinstance(val, Config):
            return val.to_dict()
        if isinstance(val, (list, tuple)):
            generator = (Config.__item_to_dict(item) for item in val)
            return list(generator)
        return val
    
    def merge(self, other_conf):
        return Config(
            data={**self.to_dict(), **other_conf.to_dict()},
            base_path=''
        )

    def to_dict(self) -> dict:
        """
        Convert object to dict recursively!
        :return: Dictionary output
        """
        return {
            key: Config.__item_to_dict(val) for key, val in self.__dict__.items()
        }


def load_config(config_file_path: Union[str, PathLike], base_path: Optional[Union[str, PathLike]] = None,
                file_type: Literal['json', 'JSON', 'yml', 'YML', 'yaml', 'YAML', None] = None) -> Config:
    """
    Load configs from a YAML or JSON file.
    :param config_file_path: File path as a string or pathlike object
    :param base_path: Base path for `path:` strings, default value is parent of `config_file_path`
    :param file_type: What is the format of the file. If none it will look at the file extension
    :return: A config object
    """
    if base_path is None:
        base_path = str(Path(config_file_path).resolve().parent)
    if file_type is None:
        file_type = Path(config_file_path).suffix
        file_type = file_type[1:]  # remove extra first dot!

    content = Path(config_file_path).read_text(encoding='utf-8')
    load_content = {
        'json': json.loads,
        'yaml': yaml.safe_load,
        'yml': yaml.safe_load
    }[file_type.lower()]
    return Config(load_content(content), base_path)