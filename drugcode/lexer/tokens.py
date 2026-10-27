"""
DrugCode DSL - Token Definitions
Defines all lexical tokens recognized by the DrugCode compiler.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Any, Optional


class TokenType(Enum):
    # End of File / Special
    EOF = auto()
    ERROR = auto()

    # Identifiers & Literals
    IDENTIFIER = auto()
    STRING = auto()
    NUMBER = auto()
    BOOLEAN = auto()

    # DrugCode Keywords
    KW_TARGET = auto()
    KW_PIPELINE = auto()
    KW_DATASET = auto()
    KW_FILTER = auto()
    KW_SCORE = auto()
    KW_RANK = auto()
    KW_EXPORT = auto()
    KW_AGAINST = auto()
    KW_TO = auto()
    KW_AND = auto()
    KW_OR = auto()
    KW_NOT = auto()
    KW_TRUE = auto()
    KW_FALSE = auto()
    KW_ASC = auto()
    KW_DESC = auto()
    KW_ORDER_BY = auto()
    KW_TOP_K = auto()
    KW_LOAD_SMILES = auto()
    KW_LOAD_SDF = auto()
    KW_PARALLEL = auto()
    KW_OPTIMIZE = auto()

    # Punctuation & Delimiters
    LPAREN = auto()       # (
    RPAREN = auto()       # )
    LBRACE = auto()       # {
    RBRACE = auto()       # }
    LBRACKET = auto()     # [
    RBRACKET = auto()     # ]
    COMMA = auto()        # ,
    COLON = auto()        # :
    SEMICOLON = auto()    # ;
    DOT = auto()          # .
    ASSIGN = auto()       # =
    AT = auto()           # @

    # Comparison Operators
    EQ = auto()           # ==
    NEQ = auto()          # !=
    LT = auto()           # <
    LE = auto()           # <=
    GT = auto()           # >
    GE = auto()           # >=

    # Arithmetic Operators
    PLUS = auto()         # +
    MINUS = auto()        # -
    STAR = auto()         # *
    SLASH = auto()        # /


KEYWORDS = {
    "target": TokenType.KW_TARGET,
    "pipeline": TokenType.KW_PIPELINE,
    "dataset": TokenType.KW_DATASET,
    "filter": TokenType.KW_FILTER,
    "score": TokenType.KW_SCORE,
    "rank": TokenType.KW_RANK,
    "export": TokenType.KW_EXPORT,
    "against": TokenType.KW_AGAINST,
    "to": TokenType.KW_TO,
    "and": TokenType.KW_AND,
    "or": TokenType.KW_OR,
    "not": TokenType.KW_NOT,
    "true": TokenType.KW_TRUE,
    "false": TokenType.KW_FALSE,
    "asc": TokenType.KW_ASC,
    "desc": TokenType.KW_DESC,
    "order_by": TokenType.KW_ORDER_BY,
    "top_k": TokenType.KW_TOP_K,
    "load_smiles": TokenType.KW_LOAD_SMILES,
    "load_sdf": TokenType.KW_LOAD_SDF,
    "parallel": TokenType.KW_PARALLEL,
    "optimize": TokenType.KW_OPTIMIZE,
}


@dataclass
class SourceLocation:
    line: int
    column: int
    file_path: Optional[str] = None

    def __str__(self) -> str:
        loc = f"line {self.line}, col {self.column}"
        if self.file_path:
            return f"{self.file_path}:{loc}"
        return loc


@dataclass
class Token:
    type: TokenType
    value: Any
    lexeme: str
    location: SourceLocation

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, {self.location})"
