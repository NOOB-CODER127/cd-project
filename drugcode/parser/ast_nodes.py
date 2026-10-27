"""
DrugCode DSL - Abstract Syntax Tree (AST) Nodes
Defines node classes representing the syntactic structure of DrugCode programs.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from drugcode.lexer.tokens import SourceLocation


class ASTNode(ABC):
    location: SourceLocation

    def accept(self, visitor: "ASTVisitor") -> Any:
        method_name = f"visit_{self.__class__.__name__}"
        visitor_func = getattr(visitor, method_name, visitor.generic_visit)
        return visitor_func(self)


class ASTVisitor(ABC):
    def generic_visit(self, node: ASTNode) -> Any:
        raise NotImplementedError(f"No visit_{node.__class__.__name__} method defined in {self.__class__.__name__}")


# =====================================================================
# Expressions
# =====================================================================

@dataclass
class ExpressionNode(ASTNode):
    pass


@dataclass
class LiteralNode(ExpressionNode):
    value: Any
    literal_type: str  # "number", "string", "boolean"
    location: SourceLocation


@dataclass
class IdentifierNode(ExpressionNode):
    name: str
    location: SourceLocation


@dataclass
class QualifiedIdentifierNode(ExpressionNode):
    base: str
    member: str
    location: SourceLocation


@dataclass
class ListNode(ExpressionNode):
    elements: List[ExpressionNode]
    location: SourceLocation


@dataclass
class BinaryOpNode(ExpressionNode):
    left: ExpressionNode
    op: str  # "and", "or", "==", "!=", "<", "<=", ">", ">=", "+", "-", "*", "/"
    right: ExpressionNode
    location: SourceLocation


@dataclass
class UnaryOpNode(ExpressionNode):
    op: str  # "not", "-"
    operand: ExpressionNode
    location: SourceLocation


@dataclass
class AnnotationNode(ASTNode):
    name: str
    arguments: Dict[str, ExpressionNode]
    location: SourceLocation


# =====================================================================
# Statements & Declarations
# =====================================================================

@dataclass
class StatementNode(ASTNode):
    pass


@dataclass
class DatasetStmtNode(StatementNode):
    name: str
    source_type: str  # "smiles" or "sdf"
    path_expr: ExpressionNode
    location: SourceLocation


@dataclass
class FilterStmtNode(StatementNode):
    name: str
    input_dataset: str
    predicates: List[ExpressionNode]
    location: SourceLocation


@dataclass
class ScoreStmtNode(StatementNode):
    name: str
    input_dataset: str
    target_name: str
    params: Dict[str, ExpressionNode]
    location: SourceLocation


@dataclass
class RankStmtNode(StatementNode):
    name: str
    input_dataset: str
    order_by_field: ExpressionNode
    direction: str  # "asc" or "desc"
    top_k: Optional[int]
    location: SourceLocation


@dataclass
class ExportStmtNode(StatementNode):
    source_dataset: str
    destination_path: ExpressionNode
    location: SourceLocation


@dataclass
class TargetDeclNode(ASTNode):
    name: str
    properties: Dict[str, ExpressionNode]
    location: SourceLocation


@dataclass
class PipelineDeclNode(ASTNode):
    name: str
    annotations: List[AnnotationNode]
    statements: List[StatementNode]
    location: SourceLocation


@dataclass
class ProgramNode(ASTNode):
    targets: List[TargetDeclNode] = field(default_factory=list)
    pipelines: List[PipelineDeclNode] = field(default_factory=list)
    location: SourceLocation = field(default_factory=lambda: SourceLocation(1, 1))
