from __future__ import annotations
from abc import abstractmethod, ABCMeta


class Command(metaclass=ABCMeta):
    """
    The Command interface declares a method for executing a command.
    """
    def __init__(self, *args, **kwargs):
        self._result = None

    @property
    def result(self):
        return self._result

    @abstractmethod
    def execute(self) -> None:
        pass


class Invoker:
    _on_start = None
    _on_finish = None

    """
    Initialize commands.
    """

    def set_on_start(self, command: Command):
        self._on_start = command

    def set_on_finish(self, command: Command):
        self._on_finish = command

    def do_something_important(self) -> None:
        if isinstance(self._on_start, Command):
            self._on_start.execute()

        if isinstance(self._on_finish, Command):
            self._on_finish.execute()
