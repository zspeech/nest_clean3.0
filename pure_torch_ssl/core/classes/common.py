# Simplified common utilities - no type checking

from functools import wraps


def typecheck():
    """No-op decorator (type checking disabled)."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


class Typing:
    """Base class for typed modules (no-op)."""
    @property
    def input_types(self):
        return None
    
    @property
    def output_types(self):
        return None
