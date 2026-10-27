"""
Unit tests for Classical and ML-Guided Optimizers
"""

import unittest
from drugcode.parser import Parser
from drugcode.ir import IRBuilder
from drugcode.optimizer import PassManager, DeadStageEliminationPass, FilterFusionPass


class TestOptimizer(unittest.TestCase):
    def test_dead_stage_elimination(self):
        code = """
        pipeline DeadPipe {
            dataset ds = load_smiles("data.smi");
            filter Active(ds) {
                MW <= 500.0;
            }
            filter Dead(ds) {
                LogP <= 2.0;
            }
            export Active to "out.csv";
        }
        """
        ast = Parser.from_source(code).parse()
        ir = IRBuilder().build(ast)
        pipe = ir.pipelines["DeadPipe"]
        self.assertIn("Dead", pipe.nodes)

        res = DeadStageEliminationPass().run(pipe)
        self.assertTrue(res.modified)
        self.assertNotIn("Dead", pipe.nodes)
        self.assertIn("Active", pipe.nodes)

    def test_filter_fusion(self):
        code = """
        pipeline FusePipe {
            dataset ds = load_smiles("data.smi");
            filter Filter1(ds) {
                MW <= 500.0;
            }
            filter Filter2(Filter1) {
                LogP <= 5.0;
            }
            export Filter2 to "out.csv";
        }
        """
        ast = Parser.from_source(code).parse()
        ir = IRBuilder().build(ast)
        pipe = ir.pipelines["FusePipe"]

        res = FilterFusionPass().run(pipe)
        self.assertTrue(res.modified)
        self.assertNotIn("Filter1", pipe.nodes)
        self.assertIn("Filter2", pipe.nodes)
        # Verify predicates merged
        self.assertEqual(len(pipe.nodes["Filter2"].predicates), 2)

    def test_ml_reordering(self):
        code = """
        target TargetX {
            pdb_id: "6LU7",
            grid_center: [0.0, 0.0, 0.0]
        }
        pipeline SlowPipe {
            dataset ds = load_smiles("data.smi");
            score HeavyDock(ds) against TargetX {
                exhaustiveness: 8,
                affinity_threshold: -7.0
            }
            filter FastFilt(HeavyDock) {
                MW <= 500.0;
            }
            export FastFilt to "out.csv";
        }
        """
        ast = Parser.from_source(code).parse()
        ir = IRBuilder().build(ast)
        pm = PassManager(opt_level=2)
        res = pm.run(ir)
        pipe = ir.pipelines["SlowPipe"]

        # FastFilt should now be scheduled BEFORE HeavyDock
        order = pipe.execution_order
        idx_fast = order.index("FastFilt")
        idx_dock = order.index("HeavyDock")
        self.assertLess(idx_fast, idx_dock)


if __name__ == "__main__":
    unittest.main()
