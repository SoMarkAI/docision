"""Docision: document candidate scoring, serving and evaluation."""
__version__ = '0.2.1rc2'


def __getattr__(name):
    if name == 'Docision':
        from .api import Docision
        return Docision
    if name == 'DocisionEngine':
        from .engine import DocisionEngine
        return DocisionEngine
    raise AttributeError(name)
