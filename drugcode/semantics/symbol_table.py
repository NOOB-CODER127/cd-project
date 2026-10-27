"""
DrugCode DSL - Symbol Table and Scope Management
Manages identifiers, types, and scoping for targets, datasets, and pipeline stages.
"""

from typing import Dict, Any, Optional
from enum import Enum, auto


class SymbolType(Enum):
    TARGET = auto()
    DATASET = auto()
    STAGE_FILTER = auto()
    STAGE_SCORE = auto()
    STAGE_RANK = auto()
    STAGE_EXPORT = auto()
    PROPERTY = auto()


class Symbol:
    def __init__(self, name: str, sym_type: SymbolType, metadata: Optional[Dict[str, Any]] = None):
        self.name = name
        self.sym_type = sym_type
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"Symbol({self.name}, {self.sym_type.name}, {self.metadata})"


class Scope:
    def __init__(self, name: str, parent: Optional["Scope"] = None):
        self.name = name
        self.parent = parent
        self.symbols: Dict[str, Symbol] = {}

    def define(self, symbol: Symbol) -> bool:
        if symbol.name in self.symbols:
            return False  # Already exists in current scope
        self.symbols[symbol.name] = symbol
        return True

    def lookup(self, name: str, recursive: bool = True) -> Optional[Symbol]:
        if name in self.symbols:
            return self.symbols[name]
        if recursive and self.parent:
            return self.parent.lookup(name, recursive=True)
        return None

    def __repr__(self) -> str:
        return f"Scope({self.name}, symbols={list(self.symbols.keys())})"


class SymbolTable:
    def __init__(self):
        self.global_scope = Scope("global")
        self.current_scope = self.global_scope
        self._init_builtins()

    def _init_builtins(self):
        # Known molecular properties and their expected data types ("number", "boolean", "string")
        builtins = {
            # Physicochemical properties
            "MW": ("number", "Molecular Weight (Da)"),
            "LogP": ("number", "Octanol-water partition coefficient"),
            "HBD": ("number", "Hydrogen Bond Donors"),
            "HBA": ("number", "Hydrogen Bond Acceptors"),
            "TPSA": ("number", "Topological Polar Surface Area"),
            "RotatableBonds": ("number", "Number of rotatable single bonds"),
            "AromaticRings": ("number", "Count of aromatic rings"),
            "HeavyAtoms": ("number", "Number of non-hydrogen atoms"),
            "QED": ("number", "Quantitative Estimate of Drug-likeness [0, 1]"),
            "SyntheticAccessibility": ("number", "Synthetic accessibility score [1 (easy) - 10 (hard)]"),
            # ADMET Properties
            "hERG_Inhibition": ("boolean", "Cardiotoxicity indicator (hERG potassium channel block)"),
            "OralBioavailability": ("number", "Predicted oral bioavailability probability [0, 1]"),
            "BBB_Permeability": ("boolean", "Blood-Brain Barrier penetration indicator"),
            "CYP3A4_Inhibition": ("boolean", "Cytochrome P450 3A4 inhibition indicator"),
        }
        for prop, (ptype, desc) in builtins.items():
            self.global_scope.define(Symbol(prop, SymbolType.PROPERTY, {"val_type": ptype, "description": desc}))

    def enter_scope(self, name: str) -> Scope:
        new_scope = Scope(name, parent=self.current_scope)
        self.current_scope = new_scope
        return new_scope

    def exit_scope(self):
        if self.current_scope.parent:
            self.current_scope = self.current_scope.parent
