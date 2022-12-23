from typing import Protocol, Any, SupportsFloat as Numeric


class GetItem(Protocol):
    # noinspection PyUnresolvedReferences
    def __getitem__(self: 'Getitem', key: Any) -> Any:
        pass


class SetItem(Protocol):
    # noinspection PyUnresolvedReferences
    def __setitem__(self: 'SetItem', key: Any, value: Any) -> Any:
        pass


__all__ = (
    'GetItem',
    'SetItem',
    'Numeric',
)
