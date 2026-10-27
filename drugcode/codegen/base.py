"""
DrugCode DSL - Code Generation Base Interface
"""

from abc import ABC, abstractmethod
from drugcode.ir.ir_nodes import IRProgram


class BaseCodeGen(ABC):
    @abstractmethod
    def generate(self, program: IRProgram) -> str:
        """Translates the compiled IRProgram into target source code."""
        pass
