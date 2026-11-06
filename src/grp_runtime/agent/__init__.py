__all__ = ["RuntimeHook"]


def __getattr__(name):
    if name == "RuntimeHook":
        from .hook import RuntimeHook
        return RuntimeHook
    raise AttributeError(name)
