"""
DrugCode DSL - Classical Optimization Passes
Implements Dead Stage Elimination, Constant Folding, and Filter Fusion.
"""

from typing import List, Dict, Set, Tuple
from drugcode.ir.ir_nodes import (
    IRPipeline, IRNode, IROpType, IRFilter, IRScore, IRRank, IRExport, IRDatasetLoad
)


class PassResult:
    def __init__(self, modified: bool, message: str = ""):
        self.modified = modified
        self.message = message


class DeadStageEliminationPass:
    """
    Eliminates stages whose results are never consumed by an export or active stage.
    """
    def run(self, pipeline: IRPipeline) -> PassResult:
        # 1. Identify all sinks (e.g. Export nodes)
        sinks = [nid for nid, node in pipeline.nodes.items() if node.op_type == IROpType.EXPORT]
        if not sinks:
            # If no export node, keep the last node in execution order
            if pipeline.execution_order:
                sinks = [pipeline.execution_order[-1]]

        # 2. Backward reachability from sinks
        reachable: Set[str] = set()
        queue = list(sinks)

        while queue:
            cur_id = queue.pop(0)
            if cur_id in reachable:
                continue
            reachable.add(cur_id)
            node = pipeline.nodes.get(cur_id)
            if node:
                for in_id in node.inputs:
                    if in_id not in reachable and in_id in pipeline.nodes:
                        queue.append(in_id)

        # 3. Prune unreachable nodes
        all_nodes = list(pipeline.nodes.keys())
        dead_nodes = [nid for nid in all_nodes if nid not in reachable]

        if not dead_nodes:
            return PassResult(False, "No dead stages found")

        for nid in dead_nodes:
            del pipeline.nodes[nid]

        pipeline.execution_order = [nid for nid in pipeline.execution_order if nid in reachable]
        return PassResult(True, f"Eliminated dead stages: {', '.join(dead_nodes)}")


class FilterFusionPass:
    """
    Fuses consecutive filter stages (e.g. Filter1 -> Filter2) into a single composite filter
    to eliminate intermediate buffer copying and enable vectorized multi-property computation.
    """
    def run(self, pipeline: IRPipeline) -> PassResult:
        modified = False
        fused_names = []

        # Find consecutive filter nodes: node_a -> node_b where both are IRFilter
        # and node_a has only node_b as its single consumer
        changed = True
        while changed:
            changed = False
            for i in range(len(pipeline.execution_order) - 1):
                id_a = pipeline.execution_order[i]
                id_b = pipeline.execution_order[i + 1]

                node_a = pipeline.nodes.get(id_a)
                node_b = pipeline.nodes.get(id_b)

                if isinstance(node_a, IRFilter) and isinstance(node_b, IRFilter):
                    # Check if node_b's only input is node_a
                    if node_b.inputs == [id_a]:
                        # Check if any other node in pipeline takes id_a as input
                        other_consumers = [
                            nid for nid, n in pipeline.nodes.items()
                            if nid != id_b and id_a in n.inputs
                        ]
                        if not other_consumers:
                            # Safe to fuse!
                            # Merge predicates and required properties
                            node_b.predicates = node_a.predicates + node_b.predicates
                            for p in node_a.required_properties:
                                if p not in node_b.required_properties:
                                    node_b.required_properties.append(p)

                            # Re-wire inputs
                            node_b.inputs = list(node_a.inputs)
                            node_b.estimated_cost_ms += node_a.estimated_cost_ms
                            node_b.estimated_selectivity *= node_a.estimated_selectivity

                            # Remove node_a
                            del pipeline.nodes[id_a]
                            pipeline.execution_order.remove(id_a)

                            fused_names.append(f"{id_a}+{id_b}")
                            modified = True
                            changed = True
                            break

        if modified:
            return PassResult(True, f"Fused filters: {', '.join(fused_names)}")
        return PassResult(False, "No fusible filters found")
