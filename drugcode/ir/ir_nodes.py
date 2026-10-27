"""
DrugCode DSL - Intermediate Representation (IR) Nodes
Represents high-level workflow DAG and low-level execution instructions.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum, auto


class IROpType(Enum):
    LOAD = auto()
    FILTER = auto()
    SCORE = auto()
    RANK = auto()
    EXPORT = auto()


@dataclass
class IRTarget:
    name: str
    pdb_id: Optional[str] = None
    pocket_residues: List[str] = field(default_factory=list)
    grid_center: List[float] = field(default_factory=list)
    grid_size: List[float] = field(default_factory=list)


@dataclass
class IRNode:
    id: str
    op_type: IROpType
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    # Cost & selectivity metadata for ML-guided optimizer
    estimated_cost_ms: float = 1.0       # Expected execution latency per molecule in milliseconds
    estimated_selectivity: float = 1.0   # Expected fraction of molecules that pass [0.0, 1.0]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        return f"{self.id}: {self.op_type.name}(in={self.inputs}, out={self.outputs})"


@dataclass
class IRDatasetLoad(IRNode):
    format: str = "smiles"
    filepath: str = ""


@dataclass
class IRPredicate:
    property_name: str
    operator: str
    threshold: Any
    raw_expr_str: str


@dataclass
class IRFilter(IRNode):
    predicates: List[IRPredicate] = field(default_factory=list)
    required_properties: List[str] = field(default_factory=list)


@dataclass
class IRScore(IRNode):
    target_name: str = ""
    scoring_function: str = "vina_empirical"
    exhaustiveness: int = 8
    affinity_threshold: Optional[float] = None


@dataclass
class IRRank(IRNode):
    order_by_field: str = ""
    direction: str = "asc"
    top_k: Optional[int] = None


@dataclass
class IRExport(IRNode):
    destination_path: str = ""
    format: str = "csv"


@dataclass
class IRPipeline:
    name: str
    annotations: Dict[str, Any] = field(default_factory=dict)
    nodes: Dict[str, IRNode] = field(default_factory=dict)
    execution_order: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: IRNode):
        self.nodes[node.id] = node
        if node.id not in self.execution_order:
            self.execution_order.append(node.id)

    def get_node(self, node_id: str) -> Optional[IRNode]:
        return self.nodes.get(node_id)


@dataclass
class IRProgram:
    targets: Dict[str, IRTarget] = field(default_factory=dict)
    pipelines: Dict[str, IRPipeline] = field(default_factory=dict)
