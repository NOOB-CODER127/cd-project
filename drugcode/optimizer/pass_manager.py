"""
DrugCode DSL - Optimization Pass Manager
Orchestrates classical and ML-guided compiler passes.
"""

from typing import List, Dict, Any, Optional
from drugcode.ir.ir_nodes import IRProgram, IRPipeline
from .classical_passes import DeadStageEliminationPass, FilterFusionPass, PassResult
from .ml_optimizer import MLGuidedOptimizerPass, MLOptimizationReport
from .cost_model import DatasetProfile, MLCostModel


class PassManager:
    def __init__(self, opt_level: int = 2):
        self.opt_level = opt_level
        self.dse_pass = DeadStageEliminationPass()
        self.fusion_pass = FilterFusionPass()
        self.ml_pass = MLGuidedOptimizerPass()

    def run(self, ir_prog: IRProgram, profile: Optional[DatasetProfile] = None) -> Dict[str, Any]:
        results = {
            "opt_level": self.opt_level,
            "pipelines": {}
        }

        for pipe_name, pipeline in ir_prog.pipelines.items():
            pipe_log = []

            if self.opt_level >= 1:
                # Run Dead Stage Elimination
                dse_res = self.dse_pass.run(pipeline)
                if dse_res.modified:
                    pipe_log.append(f"DSE: {dse_res.message}")

            if self.opt_level >= 2:
                # Run ML-Guided Optimization
                ml_report = self.ml_pass.run(pipeline, profile)
                pipe_log.append(f"ML-Opt: Estimated speedup = {ml_report.estimated_speedup:.2f}x")
                if ml_report.reordered_stages:
                    pipe_log.append(f"ML-Opt: Stages reordered -> {[s.id for s in ml_report.reordered_stages]}")
                results["pipelines"][pipe_name] = {
                    "report": ml_report,
                    "log": pipe_log
                }

            if self.opt_level >= 1:
                # Run Filter Fusion after reordering
                fusion_res = self.fusion_pass.run(pipeline)
                if fusion_res.modified:
                    pipe_log.append(f"Fusion: {fusion_res.message}")

            if pipe_name not in results["pipelines"]:
                results["pipelines"][pipe_name] = {"log": pipe_log}

        return results
