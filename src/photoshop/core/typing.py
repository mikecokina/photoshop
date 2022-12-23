from typing import Protocol, Any, SupportsFloat as Numeric


class GetItem(Protocol):
    # noinspection PyUnresolvedReferences
    def __getitem__(self: 'Getitem', key: Any) -> Any:
        pass


__all__ = (
    'GetItem',
    'Numeric',
)
