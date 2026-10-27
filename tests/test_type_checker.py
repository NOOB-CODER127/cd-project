"""
Unit tests for DrugCode Semantic Analysis and Type Checking
"""

import unittest
from drugcode.parser import Parser
from drugcode.semantics import TypeChecker, SemanticError


class TestTypeChecker(unittest.TestCase):
    def test_valid_program(self):
        code = """
        target Receptor {
            pdb_id: "1M17",
            grid_center: [1.0, 2.0, 3.0]
        }
        pipeline ValidPipe {
            dataset ds = load_smiles("data.smi");
            filter F(ds) {
                MW <= 500.0;
                LogP >= -0.4;
                hERG_Inhibition == false;
            }
            score S(F) against Receptor {
                exhaustiveness: 8
            }
            rank R(S) {
                order_by: S.affinity asc,
                top_k: 10
            }
            export R to "out.csv";
        }
        """
        ast = Parser.from_source(code).parse()
        self.assertTrue(TypeChecker().check(ast))

    def test_undefined_target_error(self):
        code = """
        pipeline UndefTargetPipe {
            dataset ds = load_smiles("data.smi");
            score S(ds) against NonExistentTarget {
                exhaustiveness: 8
            }
        }
        """
        ast = Parser.from_source(code).parse()
        with self.assertRaises(SemanticError):
            TypeChecker().check(ast)

    def test_type_mismatch_error(self):
        code = """
        pipeline MismatchPipe {
            dataset ds = load_smiles("data.smi");
            filter F(ds) {
                MW <= "five_hundred";
            }
        }
        """
        ast = Parser.from_source(code).parse()
        with self.assertRaises(SemanticError):
            TypeChecker().check(ast)


if __name__ == "__main__":
    unittest.main()
