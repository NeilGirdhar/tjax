from typing import Generic, TypeVar

import equinox as eqx

T = TypeVar('T')


class EditableMemory(eqx.Module, Generic[T]):
    state: eqx.nn.State
    memory: 'Memory[T]'

    @property
    def value(self) -> T:
        return self.state.get(self.memory)

    @value.setter
    def value(self, value: T, /) -> None:
        self.state.set(self.memory, value)


class Memory(eqx.nn.StateIndex[T]):
    """A simpler interface for StateIndex.

    For example:
        class C:
            m: Memory
            def f(self, state: eqx.nn.State) -> None:
                m = self.m.modify(state)
                m.value += 1.2
                m.value = f(m.value, z)
    """
    def modify(self, state: eqx.nn.State) -> EditableMemory[T]:
        return EditableMemory(state, self)
