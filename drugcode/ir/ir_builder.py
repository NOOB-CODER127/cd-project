"""
DrugCode DSL - IR Builder
Lowers the typed Abstract Syntax Tree into the Intermediate Representation (IR).
"""

from typing import List, Dict, Any, Optional
from drugcode.parser.ast_nodes import (
    ProgramNode, TargetDeclNode, PipelineDeclNode, StatementNode,
    DatasetStmtNode, FilterStmtNode, ScoreStmtNode, RankStmtNode,
    ExportStmtNode, ExpressionNode, BinaryOpNode, UnaryOpNode,
    LiteralNode, IdentifierNode, QualifiedIdentifierNode, ListNode
)
from .ir_nodes import (
    IRProgram, IRPipeline, IRTarget, IRNode, IROpType,
    IRDatasetLoad, IRFilter, IRPredicate, IRScore, IRRank, IRExport
)


class IRBuilder:
    def __init__(self):
        pass

    def build(self, program_ast: ProgramNode) -> IRProgram:
        ir_prog = IRProgram()

        # Lower targets
        for t_ast in program_ast.targets:
            ir_target = self._build_target(t_ast)
            ir_prog.targets[ir_target.name] = ir_target

        # Lower pipelines
        for p_ast in program_ast.pipelines:
            ir_pipeline = self._build_pipeline(p_ast)
            ir_prog.pipelines[ir_pipeline.name] = ir_pipeline

        return ir_prog

    def _build_target(self, node: TargetDeclNode) -> IRTarget:
        props = node.properties

        pdb_id = self._eval_const_literal(props.get("pdb_id")) if "pdb_id" in props else None

        pocket_residues = []
        if "pocket_residues" in props:
            res_val = self._eval_const_literal(props["pocket_residues"])
            if isinstance(res_val, list):
                pocket_residues = [str(r) for r in res_val]

        grid_center = []
        if "grid_center" in props:
            gc_val = self._eval_const_literal(props["grid_center"])
            if isinstance(gc_val, list):
                grid_center = [float(x) for x in gc_val]

        grid_size = [20.0, 20.0, 20.0]
        if "grid_size" in props:
            gs_val = self._eval_const_literal(props["grid_size"])
            if isinstance(gs_val, list):
                grid_size = [float(x) for x in gs_val]

        return IRTarget(
            name=node.name,
            pdb_id=pdb_id,
            pocket_residues=pocket_residues,
            grid_center=grid_center,
            grid_size=grid_size
        )

    def _build_pipeline(self, node: PipelineDeclNode) -> IRPipeline:
        annotations: Dict[str, Any] = {}
        for ann in node.annotations:
            ann_args = {}
            for k, v in ann.arguments.items():
                ann_args[k] = self._eval_const_literal(v)
            annotations[ann.name] = ann_args

        ir_pipeline = IRPipeline(name=node.name, annotations=annotations)

        # Lower each statement
        for stmt in node.statements:
            if isinstance(stmt, DatasetStmtNode):
                ir_node = IRDatasetLoad(
                    id=stmt.name,
                    op_type=IROpType.LOAD,
                    inputs=[],
                    outputs=[stmt.name],
                    format=stmt.source_type,
                    filepath=str(self._eval_const_literal(stmt.path_expr)),
                    estimated_cost_ms=0.01,
                    estimated_selectivity=1.0
                )
                ir_pipeline.add_node(ir_node)

            elif isinstance(stmt, FilterStmtNode):
                predicates: List[IRPredicate] = []
                req_props: List[str] = []

                for p_expr in stmt.predicates:
                    preds = self._extract_predicates(p_expr)
                    for p in preds:
                        predicates.append(p)
                        if p.property_name not in req_props:
                            req_props.append(p.property_name)

                ir_node = IRFilter(
                    id=stmt.name,
                    op_type=IROpType.FILTER,
                    inputs=[stmt.input_dataset],
                    outputs=[stmt.name],
                    predicates=predicates,
                    required_properties=req_props,
                    estimated_cost_ms=0.05 * len(req_props),  # Fast property computation
                    estimated_selectivity=0.45                # Prior default selectivity
                )
                ir_pipeline.add_node(ir_node)

            elif isinstance(stmt, ScoreStmtNode):
                params = {k: self._eval_const_literal(v) for k, v in stmt.params.items()}
                scoring_fn = params.get("scoring_function", "vina_empirical")
                exhaust = int(params.get("exhaustiveness", 8))
                cutoff = float(params.get("affinity_threshold")) if "affinity_threshold" in params else None

                ir_node = IRScore(
                    id=stmt.name,
                    op_type=IROpType.SCORE,
                    inputs=[stmt.input_dataset],
                    outputs=[stmt.name],
                    target_name=stmt.target_name,
                    scoring_function=scoring_fn,
                    exhaustiveness=exhaust,
                    affinity_threshold=cutoff,
                    estimated_cost_ms=25.0 * (exhaust / 8.0),  # Computationally expensive!
                    estimated_selectivity=0.15
                )
                ir_pipeline.add_node(ir_node)

            elif isinstance(stmt, RankStmtNode):
                order_field = self._expr_to_str(stmt.order_by_field)
                ir_node = IRRank(
                    id=stmt.name,
                    op_type=IROpType.RANK,
                    inputs=[stmt.input_dataset],
                    outputs=[stmt.name],
                    order_by_field=order_field,
                    direction=stmt.direction,
                    top_k=stmt.top_k,
                    estimated_cost_ms=0.02,
                    estimated_selectivity=1.0
                )
                ir_pipeline.add_node(ir_node)

            elif isinstance(stmt, ExportStmtNode):
                dest = str(self._eval_const_literal(stmt.destination_path))
                export_id = f"export_{stmt.source_dataset}"
                ir_node = IRExport(
                    id=export_id,
                    op_type=IROpType.EXPORT,
                    inputs=[stmt.source_dataset],
                    outputs=[],
                    destination_path=dest,
                    format="csv" if dest.endswith(".csv") else "tsv",
                    estimated_cost_ms=0.01,
                    estimated_selectivity=1.0
                )
                ir_pipeline.add_node(ir_node)

        return ir_pipeline

    def _extract_predicates(self, expr: ExpressionNode) -> List[IRPredicate]:
        # Handle compound "and" expressions: e.g. A and B
        if isinstance(expr, BinaryOpNode) and expr.op == "and":
            return self._extract_predicates(expr.left) + self._extract_predicates(expr.right)

        # Single binary comparison: e.g. MW <= 500.0 or hERG == false
        if isinstance(expr, BinaryOpNode) and expr.op in ("<", "<=", ">", ">=", "==", "!="):
            if isinstance(expr.left, IdentifierNode):
                prop_name = expr.left.name
                val = self._eval_const_literal(expr.right)
                return [IRPredicate(
                    property_name=prop_name,
                    operator=expr.op,
                    threshold=val,
                    raw_expr_str=f"{prop_name} {expr.op} {val}"
                )]
            elif isinstance(expr.right, IdentifierNode):
                # Flipped: 500.0 >= MW
                prop_name = expr.right.name
                val = self._eval_const_literal(expr.left)
                flip_map = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "==": "==", "!=": "!="}
                return [IRPredicate(
                    property_name=prop_name,
                    operator=flip_map[expr.op],
                    threshold=val,
                    raw_expr_str=f"{prop_name} {flip_map[expr.op]} {val}"
                )]

        return [IRPredicate(
            property_name="custom",
            operator="eval",
            threshold=None,
            raw_expr_str=self._expr_to_str(expr)
        )]

    def _eval_const_literal(self, expr: Optional[ExpressionNode]) -> Any:
        if expr is None:
            return None
        if isinstance(expr, LiteralNode):
            return expr.value
        if isinstance(expr, ListNode):
            return [self._eval_const_literal(e) for e in expr.elements]
        if isinstance(expr, IdentifierNode):
            return expr.name
        if isinstance(expr, UnaryOpNode) and expr.op == "-":
            val = self._eval_const_literal(expr.operand)
            return -val if isinstance(val, (int, float)) else val
        return None

    def _expr_to_str(self, expr: ExpressionNode) -> str:
        if isinstance(expr, LiteralNode):
            return repr(expr.value)
        if isinstance(expr, IdentifierNode):
            return expr.name
        if isinstance(expr, QualifiedIdentifierNode):
            return f"{expr.base}.{expr.member}"
        if isinstance(expr, BinaryOpNode):
            return f"({self._expr_to_str(expr.left)} {expr.op} {self._expr_to_str(expr.right)})"
        if isinstance(expr, UnaryOpNode):
            return f"{expr.op}({self._expr_to_str(expr.operand)})"
        if isinstance(expr, ListNode):
            return "[" + ", ".join(self._expr_to_str(e) for e in expr.elements) + "]"
        return str(expr)
