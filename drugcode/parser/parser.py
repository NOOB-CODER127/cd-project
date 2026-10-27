"""
DrugCode DSL - Parser
Recursive-descent syntactic analyzer producing a typed Abstract Syntax Tree (AST).
"""

from typing import List, Dict, Any, Optional
from drugcode.lexer.tokens import Token, TokenType, SourceLocation
from drugcode.lexer.lexer import Lexer
from .ast_nodes import (
    ProgramNode, TargetDeclNode, PipelineDeclNode, AnnotationNode,
    StatementNode, DatasetStmtNode, FilterStmtNode, ScoreStmtNode,
    RankStmtNode, ExportStmtNode, ExpressionNode, BinaryOpNode,
    UnaryOpNode, LiteralNode, IdentifierNode, QualifiedIdentifierNode,
    ListNode
)


class ParserError(Exception):
    def __init__(self, message: str, location: SourceLocation):
        super().__init__(f"Parser error at {location}: {message}")
        self.message = message
        self.location = location


class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    @classmethod
    def from_source(cls, source: str, file_path: Optional[str] = None) -> "Parser":
        lexer = Lexer(source, file_path)
        tokens = lexer.tokenize()
        return cls(tokens)

    def _current(self) -> Token:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return self.tokens[-1]

    def _peek(self, offset: int = 1) -> Token:
        idx = self.pos + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return self.tokens[-1]

    def _match(self, *expected_types: TokenType) -> bool:
        if self._current().type in expected_types:
            self.pos += 1
            return True
        return False

    def _check(self, token_type: TokenType) -> bool:
        return self._current().type == token_type

    def _consume(self, expected_type: TokenType, error_message: str) -> Token:
        cur = self._current()
        if cur.type == expected_type:
            self.pos += 1
            return cur
        raise ParserError(f"{error_message} (found {cur.type.name} '{cur.lexeme}')", cur.location)

    # =========================================================================
    # Grammar Entry Point
    # =========================================================================

    def parse(self) -> ProgramNode:
        targets: List[TargetDeclNode] = []
        pipelines: List[PipelineDeclNode] = []
        loc = self._current().location

        while not self._check(TokenType.EOF):
            # Annotations can precede pipeline declarations
            annotations = self._parse_annotations()

            if self._check(TokenType.KW_TARGET):
                if annotations:
                    raise ParserError("Annotations cannot be applied to target declarations", annotations[0].location)
                targets.append(self._parse_target_decl())
            elif self._check(TokenType.KW_PIPELINE):
                pipelines.append(self._parse_pipeline_decl(annotations))
            else:
                tok = self._current()
                raise ParserError(f"Expected 'target' or 'pipeline' declaration, got '{tok.lexeme}'", tok.location)

        return ProgramNode(targets=targets, pipelines=pipelines, location=loc)

    def _parse_annotations(self) -> List[AnnotationNode]:
        annotations = []
        while self._match(TokenType.AT):
            loc = self.tokens[self.pos - 1].location
            cur = self._current()
            if cur.type in (TokenType.IDENTIFIER, TokenType.KW_OPTIMIZE, TokenType.KW_PARALLEL):
                self.pos += 1
                name_tok = cur
            else:
                raise ParserError(f"Expected annotation name after '@', found {cur.type.name} '{cur.lexeme}'", cur.location)
            args: Dict[str, ExpressionNode] = {}
            if self._match(TokenType.LPAREN):
                if not self._check(TokenType.RPAREN):
                    while True:
                        cur_arg = self._current()
                        if cur_arg.type in (TokenType.IDENTIFIER, TokenType.KW_OPTIMIZE, TokenType.KW_PARALLEL):
                            self.pos += 1
                            arg_key = cur_arg.lexeme
                        else:
                            raise ParserError("Expected argument name in annotation", cur_arg.location)
                        if self._match(TokenType.ASSIGN) or self._match(TokenType.COLON):
                            arg_val = self._parse_expression()
                        else:
                            arg_val = LiteralNode(True, "boolean", self._current().location)
                        args[arg_key] = arg_val
                        if not self._match(TokenType.COMMA):
                            break
                self._consume(TokenType.RPAREN, "Expected ')' closing annotation arguments")
            annotations.append(AnnotationNode(name=name_tok.lexeme, arguments=args, location=loc))
        return annotations

    def _parse_target_decl(self) -> TargetDeclNode:
        kw_tok = self._consume(TokenType.KW_TARGET, "Expected 'target'")
        name_tok = self._consume(TokenType.IDENTIFIER, "Expected target identifier name")
        self._consume(TokenType.LBRACE, "Expected '{' opening target body")

        props: Dict[str, ExpressionNode] = {}
        while not self._check(TokenType.RBRACE) and not self._check(TokenType.EOF):
            prop_key = self._consume(TokenType.IDENTIFIER, "Expected target property key").lexeme
            self._consume(TokenType.COLON, "Expected ':' after target property key")
            prop_val = self._parse_expression()
            props[prop_key] = prop_val
            self._match(TokenType.COMMA, TokenType.SEMICOLON)

        self._consume(TokenType.RBRACE, "Expected '}' closing target body")
        return TargetDeclNode(name=name_tok.lexeme, properties=props, location=kw_tok.location)

    def _parse_pipeline_decl(self, annotations: List[AnnotationNode]) -> PipelineDeclNode:
        kw_tok = self._consume(TokenType.KW_PIPELINE, "Expected 'pipeline'")
        name_tok = self._consume(TokenType.IDENTIFIER, "Expected pipeline identifier name")
        self._consume(TokenType.LBRACE, "Expected '{' opening pipeline body")

        statements: List[StatementNode] = []
        while not self._check(TokenType.RBRACE) and not self._check(TokenType.EOF):
            statements.append(self._parse_statement())

        self._consume(TokenType.RBRACE, "Expected '}' closing pipeline body")
        return PipelineDeclNode(
            name=name_tok.lexeme,
            annotations=annotations,
            statements=statements,
            location=kw_tok.location
        )

    def _parse_statement(self) -> StatementNode:
        tok = self._current()
        if tok.type == TokenType.KW_DATASET:
            return self._parse_dataset_stmt()
        elif tok.type == TokenType.KW_FILTER:
            return self._parse_filter_stmt()
        elif tok.type == TokenType.KW_SCORE:
            return self._parse_score_stmt()
        elif tok.type == TokenType.KW_RANK:
            return self._parse_rank_stmt()
        elif tok.type == TokenType.KW_EXPORT:
            return self._parse_export_stmt()
        else:
            raise ParserError(f"Unexpected token in pipeline: '{tok.lexeme}'", tok.location)

    def _parse_dataset_stmt(self) -> DatasetStmtNode:
        kw_tok = self._consume(TokenType.KW_DATASET, "Expected 'dataset'")
        name_tok = self._consume(TokenType.IDENTIFIER, "Expected dataset variable name")
        self._consume(TokenType.ASSIGN, "Expected '=' in dataset assignment")

        loader_tok = self._current()
        if loader_tok.type == TokenType.KW_LOAD_SMILES:
            self.pos += 1
            src_type = "smiles"
        elif loader_tok.type == TokenType.KW_LOAD_SDF:
            self.pos += 1
            src_type = "sdf"
        else:
            raise ParserError("Expected 'load_smiles' or 'load_sdf' loader function", loader_tok.location)

        self._consume(TokenType.LPAREN, "Expected '(' after dataset loader")
        path_expr = self._parse_expression()
        self._consume(TokenType.RPAREN, "Expected ')' closing dataset loader")
        self._consume(TokenType.SEMICOLON, "Expected ';' ending dataset statement")

        return DatasetStmtNode(
            name=name_tok.lexeme,
            source_type=src_type,
            path_expr=path_expr,
            location=kw_tok.location
        )

    def _parse_filter_stmt(self) -> FilterStmtNode:
        kw_tok = self._consume(TokenType.KW_FILTER, "Expected 'filter'")
        name_tok = self._consume(TokenType.IDENTIFIER, "Expected filter stage name")
        self._consume(TokenType.LPAREN, "Expected '(' after filter stage name")
        in_ds = self._consume(TokenType.IDENTIFIER, "Expected input dataset name").lexeme
        self._consume(TokenType.RPAREN, "Expected ')' after input dataset name")
        self._consume(TokenType.LBRACE, "Expected '{' opening filter body")

        predicates: List[ExpressionNode] = []
        while not self._check(TokenType.RBRACE) and not self._check(TokenType.EOF):
            expr = self._parse_expression()
            predicates.append(expr)
            self._consume(TokenType.SEMICOLON, "Expected ';' after filter predicate")

        self._consume(TokenType.RBRACE, "Expected '}' closing filter body")
        return FilterStmtNode(
            name=name_tok.lexeme,
            input_dataset=in_ds,
            predicates=predicates,
            location=kw_tok.location
        )

    def _parse_score_stmt(self) -> ScoreStmtNode:
        kw_tok = self._consume(TokenType.KW_SCORE, "Expected 'score'")
        name_tok = self._consume(TokenType.IDENTIFIER, "Expected score stage name")
        self._consume(TokenType.LPAREN, "Expected '(' after score stage name")
        in_ds = self._consume(TokenType.IDENTIFIER, "Expected input dataset name").lexeme
        self._consume(TokenType.RPAREN, "Expected ')' after input dataset name")
        self._consume(TokenType.KW_AGAINST, "Expected 'against' target specification")
        target_name = self._consume(TokenType.IDENTIFIER, "Expected target identifier").lexeme
        self._consume(TokenType.LBRACE, "Expected '{' opening score body")

        params: Dict[str, ExpressionNode] = {}
        while not self._check(TokenType.RBRACE) and not self._check(TokenType.EOF):
            param_key = self._consume(TokenType.IDENTIFIER, "Expected score parameter key").lexeme
            self._consume(TokenType.COLON, "Expected ':' after score parameter key")
            param_val = self._parse_expression()
            params[param_key] = param_val
            self._match(TokenType.COMMA, TokenType.SEMICOLON)

        self._consume(TokenType.RBRACE, "Expected '}' closing score body")
        return ScoreStmtNode(
            name=name_tok.lexeme,
            input_dataset=in_ds,
            target_name=target_name,
            params=params,
            location=kw_tok.location
        )

    def _parse_rank_stmt(self) -> RankStmtNode:
        kw_tok = self._consume(TokenType.KW_RANK, "Expected 'rank'")
        name_tok = self._consume(TokenType.IDENTIFIER, "Expected rank stage name")
        self._consume(TokenType.LPAREN, "Expected '(' after rank stage name")
        in_ds = self._consume(TokenType.IDENTIFIER, "Expected input dataset name").lexeme
        self._consume(TokenType.RPAREN, "Expected ')' after input dataset name")
        self._consume(TokenType.LBRACE, "Expected '{' opening rank body")

        order_by_expr: Optional[ExpressionNode] = None
        direction: str = "asc"
        top_k: Optional[int] = None

        while not self._check(TokenType.RBRACE) and not self._check(TokenType.EOF):
            tok = self._current()
            if tok.type == TokenType.KW_ORDER_BY:
                self.pos += 1
                self._consume(TokenType.COLON, "Expected ':' after 'order_by'")
                order_by_expr = self._parse_expression()
                if self._match(TokenType.KW_ASC):
                    direction = "asc"
                elif self._match(TokenType.KW_DESC):
                    direction = "desc"
                self._match(TokenType.COMMA, TokenType.SEMICOLON)
            elif tok.type == TokenType.KW_TOP_K:
                self.pos += 1
                self._consume(TokenType.COLON, "Expected ':' after 'top_k'")
                num_tok = self._consume(TokenType.NUMBER, "Expected integer after top_k:")
                top_k = int(num_tok.value)
                self._match(TokenType.COMMA, TokenType.SEMICOLON)
            else:
                raise ParserError(f"Unexpected property in rank stage: '{tok.lexeme}'", tok.location)

        self._consume(TokenType.RBRACE, "Expected '}' closing rank body")

        if order_by_expr is None:
            raise ParserError("Rank stage must specify 'order_by:' property", kw_tok.location)

        return RankStmtNode(
            name=name_tok.lexeme,
            input_dataset=in_ds,
            order_by_field=order_by_expr,
            direction=direction,
            top_k=top_k,
            location=kw_tok.location
        )

    def _parse_export_stmt(self) -> ExportStmtNode:
        kw_tok = self._consume(TokenType.KW_EXPORT, "Expected 'export'")
        src_ds = self._consume(TokenType.IDENTIFIER, "Expected source dataset name in export").lexeme
        self._consume(TokenType.KW_TO, "Expected 'to' keyword in export statement")
        dest_path = self._parse_expression()
        self._consume(TokenType.SEMICOLON, "Expected ';' ending export statement")
        return ExportStmtNode(
            source_dataset=src_ds,
            destination_path=dest_path,
            location=kw_tok.location
        )

    # =========================================================================
    # Expression Parsing (Precedence Climbing)
    # =========================================================================

    def _parse_expression(self) -> ExpressionNode:
        return self._parse_or()

    def _parse_or(self) -> ExpressionNode:
        expr = self._parse_and()
        while self._match(TokenType.KW_OR):
            op_tok = self.tokens[self.pos - 1]
            right = self._parse_and()
            expr = BinaryOpNode(expr, "or", right, op_tok.location)
        return expr

    def _parse_and(self) -> ExpressionNode:
        expr = self._parse_equality()
        while self._match(TokenType.KW_AND):
            op_tok = self.tokens[self.pos - 1]
            right = self._parse_equality()
            expr = BinaryOpNode(expr, "and", right, op_tok.location)
        return expr

    def _parse_equality(self) -> ExpressionNode:
        expr = self._parse_relational()
        while self._check(TokenType.EQ) or self._check(TokenType.NEQ):
            op_tok = self._current()
            self.pos += 1
            right = self._parse_relational()
            expr = BinaryOpNode(expr, op_tok.lexeme, right, op_tok.location)
        return expr

    def _parse_relational(self) -> ExpressionNode:
        expr = self._parse_additive()
        while self._check(TokenType.LT) or self._check(TokenType.LE) or self._check(TokenType.GT) or self._check(TokenType.GE):
            op_tok = self._current()
            self.pos += 1
            right = self._parse_additive()
            expr = BinaryOpNode(expr, op_tok.lexeme, right, op_tok.location)
        return expr

    def _parse_additive(self) -> ExpressionNode:
        expr = self._parse_multiplicative()
        while self._check(TokenType.PLUS) or self._check(TokenType.MINUS):
            op_tok = self._current()
            self.pos += 1
            right = self._parse_multiplicative()
            expr = BinaryOpNode(expr, op_tok.lexeme, right, op_tok.location)
        return expr

    def _parse_multiplicative(self) -> ExpressionNode:
        expr = self._parse_unary()
        while self._check(TokenType.STAR) or self._check(TokenType.SLASH):
            op_tok = self._current()
            self.pos += 1
            right = self._parse_unary()
            expr = BinaryOpNode(expr, op_tok.lexeme, right, op_tok.location)
        return expr

    def _parse_unary(self) -> ExpressionNode:
        if self._check(TokenType.KW_NOT) or self._check(TokenType.MINUS):
            op_tok = self._current()
            self.pos += 1
            operand = self._parse_unary()
            return UnaryOpNode(op_tok.lexeme, operand, op_tok.location)
        return self._parse_primary()

    def _parse_primary(self) -> ExpressionNode:
        tok = self._current()

        # Literals
        if tok.type == TokenType.NUMBER:
            self.pos += 1
            return LiteralNode(tok.value, "number", tok.location)
        if tok.type == TokenType.STRING:
            self.pos += 1
            return LiteralNode(tok.value, "string", tok.location)
        if tok.type == TokenType.BOOLEAN:
            self.pos += 1
            return LiteralNode(tok.value, "boolean", tok.location)

        # List literal [elem, elem, ...]
        if tok.type == TokenType.LBRACKET:
            self.pos += 1
            elements: List[ExpressionNode] = []
            if not self._check(TokenType.RBRACKET):
                while True:
                    elements.append(self._parse_expression())
                    if not self._match(TokenType.COMMA):
                        break
            self._consume(TokenType.RBRACKET, "Expected ']' closing list literal")
            return ListNode(elements, tok.location)

        # Parenthesized expression (expr)
        if tok.type == TokenType.LPAREN:
            self.pos += 1
            expr = self._parse_expression()
            self._consume(TokenType.RPAREN, "Expected ')' closing parenthesized expression")
            return expr

        # Identifier or qualified identifier (e.g. Stage.affinity)
        if tok.type == TokenType.IDENTIFIER:
            self.pos += 1
            if self._match(TokenType.DOT):
                member_tok = self._consume(TokenType.IDENTIFIER, "Expected member identifier after '.'")
                return QualifiedIdentifierNode(base=tok.lexeme, member=member_tok.lexeme, location=tok.location)
            return IdentifierNode(tok.lexeme, tok.location)

        raise ParserError(f"Unexpected token in expression: '{tok.lexeme}'", tok.location)
