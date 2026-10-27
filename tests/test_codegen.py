"""
Unit tests for Python and C++ Code Generators
"""

import unittest
from drugcode.parser import Parser
from drugcode.ir import IRBuilder
from drugcode.optimizer import PassManager
from drugcode.codegen import PythonCodeGen, CppCodeGen


class TestCodeGen(unittest.TestCase):
    def setUp(self):
        code = """
        target TargetA {
            pdb_id: "1M17",
            grid_center: [0.0, 0.0, 0.0]
        }
        pipeline GenPipe {
            dataset lib = load_smiles("data/sample_compounds.smi");
            filter F1(lib) {
                MW <= 500.0;
            }
            score D1(F1) against TargetA {
                exhaustiveness: 8,
                affinity_threshold: -7.0
            }
            rank R1(D1) {
                order_by: D1.affinity asc,
                top_k: 5
            }
            export R1 to "output/test_gen.csv";
        }
        """
        ast = Parser.from_source(code).parse()
        self.ir = IRBuilder().build(ast)
        PassManager(opt_level=2).run(self.ir)

    def test_python_codegen(self):
        py_code = PythonCodeGen().generate(self.ir)
        self.assertIn("class MoleculeRecord", py_code)
        self.assertIn("def run_pipeline_GenPipe", py_code)
        self.assertIn("score_docking", py_code)

    def test_cpp_codegen(self):
        cpp_code = CppCodeGen().generate(self.ir)
        self.assertIn("#pragma omp parallel for", cpp_code)
        self.assertIn("struct Molecule", cpp_code)
        self.assertIn("int main()", cpp_code)


if __name__ == "__main__":
    unittest.main()
