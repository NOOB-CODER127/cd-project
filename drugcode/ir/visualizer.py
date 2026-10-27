"""
DrugCode DSL - IR Visualizer & 3-Address Code Printer
Converts the IR graph into Mermaid diagrams, Graphviz DOT, and readable 3AC listings.
"""

from typing import List
from .ir_nodes import (
    IRProgram, IRPipeline, IRNode, IROpType,
    IRDatasetLoad, IRFilter, IRScore, IRRank, IRExport
)


class IRVisualizer:
    @staticmethod
    def to_3ac(ir_program: IRProgram) -> str:
        lines: List[str] = [";; ==================================================",
                            ";; DrugCode 3-Address Code (3AC) Intermediate Representation",
                            ";; ==================================================\n"]

        for t_name, target in ir_program.targets.items():
            lines.append(f"TARGET_DECL @{t_name}:")
            lines.append(f"    pdb_id          = {target.pdb_id}")
            lines.append(f"    pocket_residues = {target.pocket_residues}")
            lines.append(f"    grid_center     = {target.grid_center}")
            lines.append(f"    grid_size       = {target.grid_size}\n")

        for p_name, pipe in ir_program.pipelines.items():
            lines.append(f"PIPELINE @{p_name}:")
            for node_id in pipe.execution_order:
                node = pipe.nodes[node_id]
                cost_str = f"[cost={node.estimated_cost_ms:.2f}ms, sel={node.estimated_selectivity:.2f}]"
                if isinstance(node, IRDatasetLoad):
                    lines.append(f"    %{node.id} = LOAD_{node.format.upper()} path=\"{node.filepath}\"  {cost_str}")
                elif isinstance(node, IRFilter):
                    pred_strs = [p.raw_expr_str for p in node.predicates]
                    lines.append(f"    %{node.id} = FILTER in=%{node.inputs[0]} predicates=[{', '.join(pred_strs)}]  {cost_str}")
                elif isinstance(node, IRScore):
                    lines.append(f"    %{node.id} = DOCK_SCORE in=%{node.inputs[0]} target=@{node.target_name} cutoff={node.affinity_threshold} exhaust={node.exhaustiveness}  {cost_str}")
                elif isinstance(node, IRRank):
                    lines.append(f"    %{node.id} = RANK in=%{node.inputs[0]} order_by={node.order_by_field} dir={node.direction} top_k={node.top_k}  {cost_str}")
                elif isinstance(node, IRExport):
                    lines.append(f"    EXPORT in=%{node.inputs[0]} dest=\"{node.destination_path}\"  {cost_str}")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def to_mermaid(ir_pipeline: IRPipeline) -> str:
        lines = ["flowchart TD", "    classDef load fill:#dbeafe,stroke:#1d4ed8,stroke-width:2px;",
                 "    classDef filter fill:#dcfce7,stroke:#15803d,stroke-width:2px;",
                 "    classDef score fill:#fef3c7,stroke:#b45309,stroke-width:2px;",
                 "    classDef rank fill:#f3e8ff,stroke:#7e22ce,stroke-width:2px;",
                 "    classDef export fill:#ffe4e6,stroke:#be123c,stroke-width:2px;"]

        for node_id, node in ir_pipeline.nodes.items():
            if isinstance(node, IRDatasetLoad):
                label = f"\"{node.id}\\n[Load {node.format.upper()}]\""
                style = "load"
            elif isinstance(node, IRFilter):
                preds = "<br/>".join(p.raw_expr_str for p in node.predicates[:3])
                if len(node.predicates) > 3:
                    preds += f"<br/>(+{len(node.predicates)-3} more)"
                label = f"\"{node.id}\\n[Filter: {preds}]\\nCost: {node.estimated_cost_ms:.2f}ms\""
                style = "filter"
            elif isinstance(node, IRScore):
                label = f"\"{node.id}\\n[Dock against {node.target_name}]\\nAffinity <= {node.affinity_threshold} kcal/mol\\nCost: {node.estimated_cost_ms:.1f}ms\""
                style = "score"
            elif isinstance(node, IRRank):
                label = f"\"{node.id}\\n[Rank: top {node.top_k} by {node.order_by_field}]\""
                style = "rank"
            elif isinstance(node, IRExport):
                label = f"\"{node.id}\\n[Export to {node.destination_path}]\""
                style = "export"
            else:
                label = f"\"{node.id}\""
                style = "filter"

            lines.append(f"    {node.id}[{label}]:::{style}")

        # Edges
        for node_id, node in ir_pipeline.nodes.items():
            for in_id in node.inputs:
                if in_id in ir_pipeline.nodes:
                    lines.append(f"    {in_id} --> {node_id}")

        return "\n".join(lines)

    @staticmethod
    def to_dot(ir_pipeline: IRPipeline) -> str:
        lines = ["digraph G {", "    rankdir=TD;", "    node [fontname=\"Helvetica\", shape=box, style=rounded];"]

        color_map = {
            IROpType.LOAD: "lightblue",
            IROpType.FILTER: "lightgreen",
            IROpType.SCORE: "gold",
            IROpType.RANK: "plum",
            IROpType.EXPORT: "lightcoral"
        }

        for node_id, node in ir_pipeline.nodes.items():
            color = color_map.get(node.op_type, "white")
            lines.append(f'    "{node_id}" [fillcolor="{color}", style="filled,rounded", label="{node_id}\\n{node.op_type.name}"];')

        for node_id, node in ir_pipeline.nodes.items():
            for in_id in node.inputs:
                if in_id in ir_pipeline.nodes:
                    lines.append(f'    "{in_id}" -> "{node_id}";')

        lines.append("}")
        return "\n".join(lines)
