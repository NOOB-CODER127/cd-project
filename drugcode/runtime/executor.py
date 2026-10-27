"""
DrugCode DSL - Runtime Pipeline Executor
Executes compiled Intermediate Representation graphs with profiling and metrics.
"""

import os
import csv
import time
from typing import List, Dict, Any, Optional
from drugcode.ir.ir_nodes import (
    IRProgram, IRPipeline, IRNode, IROpType,
    IRDatasetLoad, IRFilter, IRScore, IRRank, IRExport, IRTarget
)
from .chemo_kernel import ChemoKernel, Molecule
from .docking_surrogate import DockingSurrogate


class StageMetric:
    def __init__(self, stage_id: str, op_type: str):
        self.stage_id = stage_id
        self.op_type = op_type
        self.input_count: int = 0
        self.output_count: int = 0
        self.elapsed_sec: float = 0.0

    @property
    def selectivity(self) -> float:
        return (self.output_count / self.input_count) if self.input_count > 0 else 0.0


class ExecutionReport:
    def __init__(self, pipeline_name: str):
        self.pipeline_name = pipeline_name
        self.total_input_molecules: int = 0
        self.final_hits_count: int = 0
        self.total_elapsed_sec: float = 0.0
        self.stage_metrics: List[StageMetric] = []
        self.exported_files: List[str] = []

    def summary(self) -> str:
        lines = [
            f"===========================================================",
            f" DrugCode Execution Report: {self.pipeline_name}",
            f"===========================================================",
            f" Total Input Compounds:  {self.total_input_molecules:,}",
            f" Final Surviving Hits:   {self.final_hits_count:,}",
            f" Total Wall-Clock Time:  {self.total_elapsed_sec:.4f} seconds",
            f" Throughput:             {self.total_input_molecules / max(1e-5, self.total_elapsed_sec):,.1f} mols/sec",
            f"-----------------------------------------------------------",
            f" Stage Breakdown:"
        ]
        for m in self.stage_metrics:
            lines.append(f"  * {m.stage_id:<18} [{m.op_type:<6}]: in={m.input_count:<5} out={m.output_count:<5} (pass={m.selectivity*100.0:5.1f}%) time={m.elapsed_sec*1000.0:7.2f}ms")
        lines.append(f"===========================================================")
        return "\n".join(lines)


class PipelineExecutor:
    def __init__(self, program: IRProgram):
        self.program = program

    def execute_pipeline(self, pipeline_name: str, verbose: bool = False) -> ExecutionReport:
        if pipeline_name not in self.program.pipelines:
            raise ValueError(f"Pipeline '{pipeline_name}' not found in program")

        pipeline = self.program.pipelines[pipeline_name]
        report = ExecutionReport(pipeline_name)
        start_time = time.perf_counter()

        # Stream table mapping node_id -> List[Molecule]
        streams: Dict[str, List[Molecule]] = {}

        for node_id in pipeline.execution_order:
            node = pipeline.nodes[node_id]
            s_metric = StageMetric(node.id, node.op_type.name)
            t0 = time.perf_counter()

            if isinstance(node, IRDatasetLoad):
                mols = self._execute_load(node)
                streams[node.id] = mols
                s_metric.input_count = len(mols)
                s_metric.output_count = len(mols)
                report.total_input_molecules = len(mols)

            elif isinstance(node, IRFilter):
                in_mols = streams.get(node.inputs[0], [])
                s_metric.input_count = len(in_mols)
                out_mols = self._execute_filter(node, in_mols)
                streams[node.id] = out_mols
                s_metric.output_count = len(out_mols)

            elif isinstance(node, IRScore):
                in_mols = streams.get(node.inputs[0], [])
                s_metric.input_count = len(in_mols)
                out_mols = self._execute_score(node, in_mols)
                streams[node.id] = out_mols
                s_metric.output_count = len(out_mols)

            elif isinstance(node, IRRank):
                in_mols = streams.get(node.inputs[0], [])
                s_metric.input_count = len(in_mols)
                out_mols = self._execute_rank(node, in_mols)
                streams[node.id] = out_mols
                s_metric.output_count = len(out_mols)

            elif isinstance(node, IRExport):
                in_mols = streams.get(node.inputs[0], [])
                s_metric.input_count = len(in_mols)
                self._execute_export(node, in_mols)
                streams[node.id] = in_mols
                s_metric.output_count = len(in_mols)
                report.exported_files.append(node.destination_path)
                report.final_hits_count = len(in_mols)

            s_metric.elapsed_sec = time.perf_counter() - t0
            report.stage_metrics.append(s_metric)

        report.total_elapsed_sec = time.perf_counter() - start_time
        if not report.final_hits_count and pipeline.execution_order:
            last_id = pipeline.execution_order[-1]
            report.final_hits_count = len(streams.get(last_id, []))

        return report

    def _execute_load(self, node: IRDatasetLoad) -> List[Molecule]:
        mols: List[Molecule] = []
        if not os.path.exists(node.filepath):
            raise FileNotFoundError(f"Dataset file not found: {node.filepath}")

        with open(node.filepath, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                smiles = parts[0]
                mol_id = parts[1] if len(parts) > 1 else f"MOL_{idx+1}"
                mol_name = parts[2] if len(parts) > 2 else mol_id
                mol = ChemoKernel.parse_smiles_to_molecule(smiles, mol_id=mol_id, name=mol_name)
                mols.append(mol)
        return mols

    def _execute_filter(self, node: IRFilter, mols: List[Molecule]) -> List[Molecule]:
        survivors = []
        for mol in mols:
            passed = True
            for pred in node.predicates:
                val = mol.get(pred.property_name)
                if val is None:
                    passed = False
                    break
                op = pred.operator
                th = pred.threshold
                if op == "<=" and not (val <= th):
                    passed = False; break
                elif op == "<" and not (val < th):
                    passed = False; break
                elif op == ">=" and not (val >= th):
                    passed = False; break
                elif op == ">" and not (val > th):
                    passed = False; break
                elif op == "==" and not (val == th):
                    passed = False; break
                elif op == "!=" and not (val != th):
                    passed = False; break
            if passed:
                survivors.append(mol)
        return survivors

    def _execute_score(self, node: IRScore, mols: List[Molecule]) -> List[Molecule]:
        target = self.program.targets.get(node.target_name)
        pocket_residues = target.pocket_residues if target else []

        survivors = []
        for mol in mols:
            # Score docking binding energy
            dock_res = DockingSurrogate.score_ligand(
                mol=mol,
                target_name=node.target_name,
                pocket_residues=pocket_residues,
                exhaustiveness=node.exhaustiveness
            )
            # Store docking results on molecule
            mol.properties[f"{node.id}.affinity"] = dock_res["affinity"]
            mol.properties[f"{node.id}.score"] = dock_res["score"]
            mol.properties[f"{node.id}.rmsd"] = dock_res["rmsd"]
            mol.properties["docking_affinity"] = dock_res["affinity"]

            # Filter by threshold if defined
            if node.affinity_threshold is not None:
                # More negative affinity is stronger binding
                if dock_res["affinity"] <= node.affinity_threshold:
                    survivors.append(mol)
            else:
                survivors.append(mol)

        return survivors

    def _execute_rank(self, node: IRRank, mols: List[Molecule]) -> List[Molecule]:
        field = node.order_by_field
        # Remove qualifiers if present
        def get_sort_key(m: Molecule):
            v = m.get(field)
            if v is None and "." in field:
                v = m.get(field.split(".")[-1])
            return v if v is not None else 9999.0

        reverse = (node.direction == "desc")
        sorted_mols = sorted(mols, key=get_sort_key, reverse=reverse)

        if node.top_k is not None:
            sorted_mols = sorted_mols[:node.top_k]

        return sorted_mols

    def _execute_export(self, node: IRExport, mols: List[Molecule]):
        dest = node.destination_path
        os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)

        if not mols:
            # Write empty file with header
            with open(dest, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["id", "name", "smiles"])
            return

        # Collect all property keys
        prop_keys = set()
        for m in mols:
            prop_keys.update(m.properties.keys())
        sorted_props = sorted(list(prop_keys))

        header = ["id", "name", "smiles"] + sorted_props
        with open(dest, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for m in mols:
                row = [m.id, m.name or m.id, m.smiles] + [m.properties.get(k, "") for k in sorted_props]
                writer.writerow(row)
