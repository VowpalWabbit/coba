"""Coba global configuration functionality."""

import sys
import json
import collections

from pathlib import Path
from typing import Dict, Any

from coba.registry import CobaRegistry
from coba.utilities import coba_exit

from coba.config.loggers import Logger
from coba.config.cachers import Cacher

class CobaConfig_meta(type):
    """To support class properties before python 3.9 we must implement our properties directly 
       on a meta class. Using class properties rather than class variables is done to allow 
       lazy loading. Lazy loading moves errors to execution time instead of import time where
       they are easier to debug.
    """

    def __init__(cls, *args, **kwargs):
        cls._api_keys  = None
        cls._cache     = None
        cls._log       = None
        cls._benchmark = None

    @staticmethod
    def _load_file_configs() -> Dict[str,Any]:
        search_paths = [Path.home() , Path.cwd(), Path(sys.path[0]) ]

        config = {}

        for search_path in search_paths:

            potential_coba_config = search_path / ".coba"

            if potential_coba_config.exists() and potential_coba_config.read_text().strip() != "":
                try:
                    file_config = json.loads(potential_coba_config.read_text())

                    if not isinstance(file_config, dict):
                        raise Exception(f"The file at {potential_coba_config} should be a json object.")

                    CobaConfig_meta._resolve_and_expand_paths(file_config, str(search_path))

                    config.update(file_config)
                except Exception as e:
                    
                    print(
                        f"The coba configuration file at {potential_coba_config} has the following formatting error, "
                        f"'{str(e)}'. To protect against unexpected behavior execution is being stopped until this is fixed."
                    )

                    coba_exit()

        return config

    @staticmethod
    def _resolve_and_expand_paths(config_dict: dict, current_dir:str):
        for key,item in config_dict.items():
            if isinstance(item, dict):
                CobaConfig_meta._resolve_and_expand_paths(item, current_dir)

            if isinstance(item,str) and item.strip().startswith("~/"):
                config_dict[key] = str(Path(item).expanduser().resolve())

            if isinstance(item,str) and (item.strip().startswith("../") or item.strip().startswith("./")):
                config_dict[key] = str(Path(current_dir,item).resolve())

    @staticmethod
    def _load_config() -> Dict[str,Any]:

        config: Dict[str,Any] = {
            "api_keys" : collections.defaultdict(lambda:None),
            "cacher"   : "NoneCacher",
            "logger"   : { "IndentLogger": "ConsoleSink" },
            "benchmark": {"processes": 1, "maxtasksperchild": None, "chunk_by": "source", "file_fmt": "BenchmarkFileV2"}
        }

        for key,value in CobaConfig_meta._load_file_configs().items():
            if key in config and isinstance(config[key], dict):
                config[key].update(value)
            else:
                config[key] = value

        return config

    @property
    def Api_Keys(cls):
        if cls._api_keys is None:
            cls._api_keys = cls._load_config()['api_keys']
        return cls._api_keys

    @Api_Keys.setter
    def Api_Keys(cls, value):
        cls._api_keys = value

    @property
    def Cacher(cls) -> Cacher[str,bytes]:
        if cls._cache is None:
            cls._cache = CobaRegistry.construct(cls._load_config()['cacher'])
        return cls._cache
    
    @Cacher.setter
    def Cacher(cls, value) -> None:
        cls._cache = value

    @property
    def Logger(cls) -> Logger:
        if cls._log is None:
            cls._log = CobaRegistry.construct(cls._load_config()['logger'])
        return cls._log
    
    @Logger.setter
    def Logger(cls, value) -> None:
        cls._log = value

    @property
    def Benchmark(cls) -> Dict[str, Any]:
        if cls._benchmark is None:
            cls._benchmark = cls._load_config()['benchmark']
        return cls._benchmark

    @Benchmark.setter
    def Benchmark(cls, value) -> None:
        cls._benchmark = value

class CobaConfig(metaclass=CobaConfig_meta):
    """Create a global configuration context to allow easy mocking and customization.

    In short, So long as the same modulename is always used to import and the import
    always occurs on the same thread I'm fairly confident this pattern will always work.

    In long, I'm somewhat unsure about this pattern for the following reasons:
        > While there seems concensus that multi-import doesn't repeat [1] it may if different modulenames are used [2]
            [1] https://stackoverflow.com/a/19077396/1066291
            [2] https://stackoverflow.com/q/13392038/1066291

        > Python 3.7 added in explicit context management in [3,4] but this implementation is thread local only
            [3] https://www.python.org/dev/peps/pep-0567/
            [4] https://docs.python.org/3/library/contextvars.html
    """
    pass