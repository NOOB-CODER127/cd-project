"""
DrugCode DSL - Static Semantic Analyzer and Type Checker
Performs type checking, symbol resolution, dataflow validation, and scope verification.
"""

from typing import List, Dict, Any, Optional
from drugcode.lexer.tokens import SourceLocation
from drugcode.parser.ast_nodes import (
    ProgramNode, TargetDeclNode, PipelineDeclNode, StatementNode,
    DatasetStmtNode, FilterStmtNode, ScoreStmtNode, RankStmtNode,
    ExportStmtNode, ExpressionNode, BinaryOpNode, UnaryOpNode,
    LiteralNode, IdentifierNode, QualifiedIdentifierNode, ListNode,
    ASTVisitor
)
from .symbol_table import SymbolTable, Symbol, SymbolType


class SemanticError(Exception):
    def __init__(self, message: str, location: SourceLocation):
        super().__init__(f"Semantic error at {location}: {message}")
        self.message = message
        self.location = location


class TypeChecker(ASTVisitor):
    def __init__(self):
        self.symbol_table = SymbolTable()
        self.errors: List[SemanticError] = []

    def check(self, program: ProgramNode) -> bool:
        self.errors.clear()
        self.visit_ProgramNode(program)
        if self.errors:
            raise self.errors[0]
        return True

    def visit_ProgramNode(self, node: ProgramNode):
        # 1. First pass: Register all targets in global scope
        for target in node.targets:
            self.visit_TargetDeclNode(target)

        # 2. Second pass: Check pipelines
        for pipeline in node.pipelines:
            self.visit_PipelineDeclNode(pipeline)

    def visit_TargetDeclNode(self, node: TargetDeclNode):
        if not self.symbol_table.global_scope.define(Symbol(node.name, SymbolType.TARGET, {"ast": node})):
            self.errors.append(SemanticError(f"Duplicate target declaration '{node.name}'", node.location))
            return

        # Check required target properties
        props = node.properties
        if "pdb_id" not in props and "grid_center" not in props:
            self.errors.append(SemanticError(f"Target '{node.name}' must specify either 'pdb_id' or 'grid_center'", node.location))

        # Check grid_center format if present
        if "grid_center" in props:
            gc = props["grid_center"]
            if not isinstance(gc, ListNode) or len(gc.elements) != 3:
                self.errors.append(SemanticError(f"Target '{node.name}' grid_center must be a list of 3 coordinates [x, y, z]", gc.location))

    def visit_PipelineDeclNode(self, node: PipelineDeclNode):
        self.symbol_table.enter_scope(f"pipeline_{node.name}")

        # Check each statement sequentially
        dataset_count = 0
        export_count = 0

        for stmt in node.statements:
            if isinstance(stmt, DatasetStmtNode):
                dataset_count += 1
                self.visit_DatasetStmtNode(stmt)
            elif isinstance(stmt, FilterStmtNode):
                self.visit_FilterStmtNode(stmt)
            elif isinstance(stmt, ScoreStmtNode):
                self.visit_ScoreStmtNode(stmt)
            elif isinstance(stmt, RankStmtNode):
                self.visit_RankStmtNode(stmt)
            elif isinstance(stmt, ExportStmtNode):
                export_count += 1
                self.visit_ExportStmtNode(stmt)

        if dataset_count == 0:
            self.errors.append(SemanticError(f"Pipeline '{node.name}' must contain at least one dataset input statement", node.location))

        self.symbol_table.exit_scope()

    def visit_DatasetStmtNode(self, node: DatasetStmtNode):
        scope = self.symbol_table.current_scope
        if not scope.define(Symbol(node.name, SymbolType.DATASET, {"source_type": node.source_type})):
            self.errors.append(SemanticError(f"Identifier '{node.name}' is already defined in pipeline", node.location))
            return

        path_type = self._eval_expr_type(node.path_expr)
        if path_type != "string":
            self.errors.append(SemanticError(f"Dataset path must be a string, got '{path_type}'", node.path_expr.location))

    def visit_FilterStmtNode(self, node: FilterStmtNode):
        scope = self.symbol_table.current_scope
        # Verify input dataset exists
        in_sym = scope.lookup(node.input_dataset)
        if not in_sym or in_sym.sym_type not in (SymbolType.DATASET, SymbolType.STAGE_FILTER, SymbolType.STAGE_SCORE, SymbolType.STAGE_RANK):
            self.errors.append(SemanticError(f"Input stream '{node.input_dataset}' for filter '{node.name}' is undefined", node.location))
            return

        if not scope.define(Symbol(node.name, SymbolType.STAGE_FILTER, {"input": node.input_dataset})):
            self.errors.append(SemanticError(f"Stage name '{node.name}' is already defined", node.location))
            return

        # Check each predicate expression evaluates to boolean
        for pred in node.predicates:
            p_type = self._eval_expr_type(pred)
            if p_type not in ("boolean", "unknown"):
                self.errors.append(SemanticError(f"Filter predicate must evaluate to boolean, got '{p_type}'", pred.location))

    def visit_ScoreStmtNode(self, node: ScoreStmtNode):
        scope = self.symbol_table.current_scope
        # Verify input dataset
        in_sym = scope.lookup(node.input_dataset)
        if not in_sym:
            self.errors.append(SemanticError(f"Input stream '{node.input_dataset}' for score stage '{node.name}' is undefined", node.location))
            return

        # Verify target exists in global scope
        target_sym = self.symbol_table.global_scope.lookup(node.target_name)
        if not target_sym or target_sym.sym_type != SymbolType.TARGET:
            self.errors.append(SemanticError(f"Target '{node.target_name}' referenced in score stage '{node.name}' is not defined", node.location))
            return

        if not scope.define(Symbol(node.name, SymbolType.STAGE_SCORE, {"input": node.input_dataset, "target": node.target_name})):
            self.errors.append(SemanticError(f"Stage name '{node.name}' is already defined", node.location))
            return

    def visit_RankStmtNode(self, node: RankStmtNode):
        scope = self.symbol_table.current_scope
        in_sym = scope.lookup(node.input_dataset)
        if not in_sym:
            self.errors.append(SemanticError(f"Input stream '{node.input_dataset}' for rank stage '{node.name}' is undefined", node.location))
            return

        if not scope.define(Symbol(node.name, SymbolType.STAGE_RANK, {"input": node.input_dataset})):
            self.errors.append(SemanticError(f"Stage name '{node.name}' is already defined", node.location))
            return

        # Check top_k positive
        if node.top_k is not None and node.top_k <= 0:
            self.errors.append(SemanticError(f"Rank top_k must be a positive integer, got {node.top_k}", node.location))

    def visit_ExportStmtNode(self, node: ExportStmtNode):
        scope = self.symbol_table.current_scope
        in_sym = scope.lookup(node.source_dataset)
        if not in_sym:
            self.errors.append(SemanticError(f"Cannot export undefined stream '{node.source_dataset}'", node.location))
            return

        dest_type = self._eval_expr_type(node.destination_path)
        if dest_type != "string":
            self.errors.append(SemanticError(f"Export destination path must be a string, got '{dest_type}'", node.destination_path.location))

    # =========================================================================
    # Type Inference & Checking for Expressions
    # =========================================================================

    def _eval_expr_type(self, expr: ExpressionNode) -> str:
        if isinstance(expr, LiteralNode):
            return expr.literal_type

        if isinstance(expr, ListNode):
            return "list"

        if isinstance(expr, IdentifierNode):
            # Check if it's a known property
            sym = self.symbol_table.global_scope.lookup(expr.name)
            if sym and sym.sym_type == SymbolType.PROPERTY:
                return sym.metadata.get("val_type", "unknown")
            # Or stage reference
            stage_sym = self.symbol_table.current_scope.lookup(expr.name)
            if stage_sym:
                return "dataset"
            self.errors.append(SemanticError(f"Unknown property or variable '{expr.name}'", expr.location))
            return "unknown"

        if isinstance(expr, QualifiedIdentifierNode):
            # e.g. DockingScreen.affinity
            stage_sym = self.symbol_table.current_scope.lookup(expr.base)
            if not stage_sym:
                self.errors.append(SemanticError(f"Unknown stage reference '{expr.base}' in '{expr.base}.{expr.member}'", expr.location))
                return "unknown"
            # Valid members for score stage: affinity, score, rmsd
            if expr.member in ("affinity", "score", "rmsd", "energy"):
                return "number"
            self.errors.append(SemanticError(f"Invalid member '{expr.member}' on stage '{expr.base}'", expr.location))
            return "unknown"

        if isinstance(expr, UnaryOpNode):
            operand_type = self._eval_expr_type(expr.operand)
            if expr.op == "not":
                if operand_type not in ("boolean", "unknown"):
                    self.errors.append(SemanticError(f"Operator 'not' requires boolean operand, got '{operand_type}'", expr.location))
                return "boolean"
            elif expr.op == "-":
                if operand_type not in ("number", "unknown"):
                    self.errors.append(SemanticError(f"Unary minus requires number operand, got '{operand_type}'", expr.location))
                return "number"

        if isinstance(expr, BinaryOpNode):
            t_left = self._eval_expr_type(expr.left)
            t_right = self._eval_expr_type(expr.right)

            # Logical operators: and, or
            if expr.op in ("and", "or"):
                if t_left not in ("boolean", "unknown") or t_right not in ("boolean", "unknown"):
                    self.errors.append(SemanticError(f"Logical '{expr.op}' requires boolean operands, got '{t_left}' and '{t_right}'", expr.location))
                return "boolean"

            # Comparison: ==, !=
            if expr.op in ("==", "!="):
                if t_left != "unknown" and t_right != "unknown" and t_left != t_right:
                    self.errors.append(SemanticError(f"Cannot compare incompatible types '{t_left}' and '{t_right}' with '{expr.op}'", expr.location))
                return "boolean"

            # Relational: <, <=, >, >=
            if expr.op in ("<", "<=", ">", ">="):
                if t_left not in ("number", "unknown") or t_right not in ("number", "unknown"):
                    self.errors.append(SemanticError(f"Comparison '{expr.op}' requires number operands, got '{t_left}' and '{t_right}'", expr.location))
                return "boolean"

            # Arithmetic: +, -, *, /
            if expr.op in ("+", "-", "*", "/"):
                if t_left not in ("number", "unknown") or t_right not in ("number", "unknown"):
                    self.errors.append(SemanticError(f"Arithmetic '{expr.op}' requires number operands, got '{t_left}' and '{t_right}'", expr.location))
                return "number"

        return "unknown"
