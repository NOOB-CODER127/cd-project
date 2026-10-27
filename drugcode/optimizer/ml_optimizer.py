"""
DrugCode DSL - Machine Learning-Guided Pipeline Optimizer
Uses ML cost/selectivity models to perform learned stage reordering,
cascade optimization, and adaptive parallel scheduling.
"""

from typing import List, Dict, Set, Tuple, Optional
import os
import itertools
from drugcode.ir.ir_nodes import (
    IRPipeline, IRNode, IROpType, IRFilter, IRScore, IRRank, IRExport, IRDatasetLoad
)
from .cost_model import MLCostModel, DatasetProfile


class MLOptimizationReport:
    def __init__(self):
        self.reordered_stages: List[str] = []
        self.unoptimized_cost_ms: float = 0.0
        self.optimized_cost_ms: float = 0.0
        self.estimated_speedup: float = 1.0
        self.worker_allocation: int = 1
        self.batch_size: int = 256
        self.notes: List[str] = []

    def __repr__(self) -> str:
        return (f"MLOptimizationReport(speedup={self.estimated_speedup:.2f}x, "
                f"unopt={self.unoptimized_cost_ms:.1f}ms, opt={self.optimized_cost_ms:.1f}ms, "
                f"workers={self.worker_allocation}, batch={self.batch_size})")


class MLGuidedOptimizerPass:
    """
    ML-guided cascade optimizer. Estimates stage cost and pass rate,
    solves for the optimal screening stage ordering, and rewires the IR DAG.
    """
    def __init__(self, cost_model: Optional[MLCostModel] = None):
        self.cost_model = cost_model or MLCostModel()

    def run(self, pipeline: IRPipeline, profile: Optional[DatasetProfile] = None) -> MLOptimizationReport:
        report = MLOptimizationReport()
        profile = profile or DatasetProfile(total_molecules=10000)

        # 1. Update ML estimates for all nodes in pipeline
        self._calibrate_ir_costs(pipeline, profile)

        # 2. Compute unoptimized pipeline expected cost per molecule
        report.unoptimized_cost_ms = self._calculate_pipeline_expected_cost(pipeline, pipeline.execution_order)

        # 3. Identify reorderable screening stages (Stages between initial Load and final Rank/Export)
        # Check dependency boundaries:
        # A filter node cannot be placed after a node it explicitly depends on for computed values.
        # But a filter based solely on intrinsic molecular properties (MW, LogP, etc.)
        # has NO data dependency on docking scores!
        reorderable_chain = self._extract_reorderable_screening_chain(pipeline)

        if len(reorderable_chain) > 1:
            # Find optimal permutation of reorderable chain
            best_order, min_cost = self._find_optimal_chain_order(pipeline, reorderable_chain)

            if best_order != reorderable_chain:
                # Rewire pipeline
                self._apply_reordered_chain(pipeline, reorderable_chain, best_order)
                report.reordered_stages = best_order
                report.notes.append(f"Reordered stages from {[n.id for n in reorderable_chain]} to {[n.id for n in best_order]}")

        # 4. Compute optimized cost and estimated speedup
        report.optimized_cost_ms = self._calculate_pipeline_expected_cost(pipeline, pipeline.execution_order)
        if report.optimized_cost_ms > 0:
            report.estimated_speedup = max(1.0, report.unoptimized_cost_ms / report.optimized_cost_ms)

        # 5. ML-guided Worker Allocation & Chunk Sizing
        total_mols = profile.total_molecules
        cpu_cores = os.cpu_count() or 4
        # Heavy pipelines benefit from higher concurrency
        is_compute_heavy = any(isinstance(n, IRScore) for n in pipeline.nodes.values())
        if is_compute_heavy:
            report.worker_allocation = min(cpu_cores, max(1, total_mols // 500))
            report.batch_size = max(32, min(512, total_mols // (report.worker_allocation * 8)))
        else:
            report.worker_allocation = min(cpu_cores // 2 or 1, max(1, total_mols // 2000))
            report.batch_size = max(128, min(2048, total_mols // (report.worker_allocation * 4)))

        pipeline.metadata["ml_optimization"] = {
            "speedup": report.estimated_speedup,
            "workers": report.worker_allocation,
            "batch_size": report.batch_size
        }

        return report

    def _calibrate_ir_costs(self, pipeline: IRPipeline, profile: DatasetProfile):
        for nid, node in pipeline.nodes.items():
            if isinstance(node, IRFilter):
                node.estimated_cost_ms = self.cost_model.estimate_stage_cost("FILTER", node.required_properties)
                node.estimated_selectivity = self.cost_model.predict_filter_selectivity(node.required_properties, profile)
            elif isinstance(node, IRScore):
                node.estimated_cost_ms = self.cost_model.estimate_stage_cost("SCORE", [], node.exhaustiveness)
                node.estimated_selectivity = self.cost_model.predict_docking_selectivity(node.affinity_threshold, profile)
            elif isinstance(node, IRRank):
                node.estimated_cost_ms = self.cost_model.estimate_stage_cost("RANK", [])
                node.estimated_selectivity = 1.0
            elif isinstance(node, IRExport):
                node.estimated_cost_ms = self.cost_model.estimate_stage_cost("EXPORT", [])
                node.estimated_selectivity = 1.0

    def _extract_reorderable_screening_chain(self, pipeline: IRPipeline) -> List[IRNode]:
        """
        Extracts a linear chain of filter and score nodes that can be safely reordered.
        """
        chain: List[IRNode] = []
        for nid in pipeline.execution_order:
            node = pipeline.nodes[nid]
            if isinstance(node, (IRFilter, IRScore)):
                # Check if this filter depends on docking score (e.g. predicates referencing Docking.affinity)
                depends_on_score = False
                if isinstance(node, IRFilter):
                    for p in node.predicates:
                        if "." in p.property_name or p.property_name in ("affinity", "score"):
                            depends_on_score = True
                            break
                if not depends_on_score:
                    chain.append(node)
                else:
                    # Chain must end before score-dependent filters
                    break
        return chain

    def _find_optimal_chain_order(self, pipeline: IRPipeline, chain: List[IRNode]) -> Tuple[List[IRNode], float]:
        """
        Evaluates permutations of the screening chain to find the schedule
        minimizing the expected cost cascade: sum(c_i * prod(s_j for j < i)).
        """
        best_order = chain
        best_cost = self._eval_chain_cost(chain)

        # For small chains (<= 7 stages), exhaustive permutation search is instantaneous (O(k!))
        # For larger chains, sorting by Smith's Rule (Cost / (1 - Selectivity)) is provably optimal.
        if len(chain) <= 6:
            for perm in itertools.permutations(chain):
                cost = self._eval_chain_cost(list(perm))
                if cost < best_cost:
                    best_cost = cost
                    best_order = list(perm)
        else:
            # Smith's Rule / Cost-Selectivity index: sort ascending by c_i / (1 - s_i)
            def score_key(node: IRNode) -> float:
                rejection = max(0.01, 1.0 - node.estimated_selectivity)
                return node.estimated_cost_ms / rejection

            sorted_chain = sorted(chain, key=score_key)
            cost = self._eval_chain_cost(sorted_chain)
            if cost < best_cost:
                best_cost = cost
                best_order = sorted_chain

        return best_order, best_cost

    def _eval_chain_cost(self, chain: List[IRNode]) -> float:
        total_cost = 0.0
        cumulative_surviving = 1.0
        for node in chain:
            total_cost += cumulative_surviving * node.estimated_cost_ms
            cumulative_surviving *= node.estimated_selectivity
        return total_cost

    def _calculate_pipeline_expected_cost(self, pipeline: IRPipeline, order: List[str]) -> float:
        total_cost = 0.0
        fraction = 1.0
        for nid in order:
            node = pipeline.nodes.get(nid)
            if not node:
                continue
            total_cost += fraction * node.estimated_cost_ms
            fraction *= node.estimated_selectivity
        return total_cost

    def _apply_reordered_chain(self, pipeline: IRPipeline, old_chain: List[IRNode], new_chain: List[IRNode]):
        """
        Rewires the pipeline edges and updates execution order to match new_chain.
        """
        old_ids = [n.id for n in old_chain]
        new_ids = [n.id for n in new_chain]

        # Get parent of old chain start
        first_old = old_chain[0]
        chain_input = first_old.inputs[0]

        # Rewire inputs through new chain
        prev_id = chain_input
        for node in new_chain:
            node.inputs = [prev_id]
            prev_id = node.id

        # Rewire any consumers that took the output of the last old node
        last_old_id = old_chain[-1].id
        last_new_id = new_chain[-1].id

        for nid, node in pipeline.nodes.items():
            if nid not in new_ids:
                if last_old_id in node.inputs:
                    idx = node.inputs.index(last_old_id)
                    node.inputs[idx] = last_new_id

        # Update execution order
        start_idx = pipeline.execution_order.index(old_ids[0])
        end_idx = start_idx + len(old_ids)
        pipeline.execution_order[start_idx:end_idx] = new_ids
