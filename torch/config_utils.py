import copy
import dataclasses
import functools
import inspect
import pickle
import torch
import contextlib
import unittest


class ContextDecorator(contextlib.ContextDecorator):
    """
    Same as contextlib.ContextDecorator, but with support for
    `unittest.TestCase`
    """

    def __call__(self, func):
        if isinstance(func, type) and issubclass(func, unittest.TestCase):

            class _TestCase(func):
                @classmethod
                def setUpClass(cls):
                    self.__enter__()
                    try:
                        super().setUpClass()
                    except Exception:
                        self.__exit__(None, None, None)
                        raise

                @classmethod
                def tearDownClass(cls):
                    try:
                        super().tearDownClass()
                    finally:
                        self.__exit__(None, None, None)

            _TestCase.__name__ = func.__name__
            return _TestCase

        return super().__call__(func)


class ConfigMixin:

    def save_config(self):
        config = copy.copy(self)
        for key in self._save_config_ignore:
            config._update_single(key, None)
        return pickle.dumps(config, protocol=2)

    def load_config(self, content):
        state = pickle.loads(content)
        self.__dict__.update(state.__dict__)
        return self

    def _update_single(self, key, val):
        pieces = key.split('.')
        current = self
        for token in pieces[:-1]:
            current = getattr(current, token)
        setattr(current, pieces[-1], val)

    def _get_single(self, key):
        pieces = key.split('.')
        current = self
        for token in pieces:
            current = getattr(current, token)
        return current

    def update(self, content_dict):
        for k, v in content_dict.items():
            self._update_single(k, v)

    def __setattr__(self, key, val):
        if not inspect.isclass(val) and key not in type(self).__dataclass_fields__:
            raise AttributeError(
                f"Trying to set attribute {key} that is not part of this config {type(self).__name__}")
        super().__setattr__(key, val)

    def to_dict(self):
        res = dataclasses.asdict(self)
        to_delete = []
        to_append = {}
        for k, v in res.items():
            if isinstance(v, dict):
                # further flatten
                for k2, v2 in v.items():
                    to_append[f"{k}.{k2}"] = v2
                to_delete.append(k)
        for k in to_delete:
            del res[k]
        res.update(to_append)
        return res

    def is_fbcode(self):
        return not hasattr(torch.version, "git_version")

    def patch(self, arg1=None, arg2=None, **kwargs):
        """
        Decorator and/or context manager to make temporary changes to a config.

        As a decorator:

            @config.patch("name", val)
            @config.patch(name1=val1, name2=val2):
            @config.patch({"name1": val1, "name2", val2})
            def foo(...):
                ...

        As a context manager:

            with config.patch("name", val):
                ...
        """
        if arg1 is not None:
            if arg2 is not None:
                # patch("key", True) syntax
                changes = {arg1: arg2}
            else:
                # patch({"key": True}) syntax
                changes = arg1
            assert not kwargs
        else:
            # patch(key=True) syntax
            changes = kwargs
            assert arg2 is None
        assert isinstance(changes, dict), f"expected `dict` got {type(changes)}"
        prior = {}
        config = self

        class ConfigPatch(ContextDecorator):
            def __enter__(self):
                assert not prior
                for key in changes.keys():
                    # KeyError on invalid entry
                    prior[key] = config._get_single(key)
                config.update(changes)

            def __exit__(self, exc_type, exc_val, exc_tb):
                config.update(prior)
                prior.clear()

        return ConfigPatch()


# Types saved/loaded in configs
CONFIG_TYPES = (int, float, bool, type(None), str, list, set, tuple, dict)


def make_config_dataclass(name, config_module):
    fields = []
    module_name = ".".join(config_module.__name__.split(".")[:-1])
    def identity(x):
        return x
    for fname, annotation in config_module.__annotations__.items():
        default_value = getattr(config_module, fname)
        field = dataclasses.field(default_factory=functools.partial(identity, default_value))
        fields.append((fname, annotation, field))
    cls = dataclasses.make_dataclass(name, fields, bases=(ConfigMixin, ))

    # NOTE: this is to make pickle work. In Python 3.12 make_dataclass
    # will take a module argument that it would set __module__ field inside.
    cls.__module__ = module_name
    return cls
