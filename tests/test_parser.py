"""
Unit tests for DrugCode Parser
"""

import unittest
from drugcode.parser import (
    Parser, ParserError, TargetDeclNode, PipelineDeclNode,
    FilterStmtNode, ScoreStmtNode
)


class TestDrugCodeParser(unittest.TestCase):
    def test_parse_target(self):
        code = """
        target CovidTarget {
            pdb_id: "6LU7",
            grid_center: [0.0, 10.0, -5.0]
        }
        """
        ast = Parser.from_source(code).parse()
        self.assertEqual(len(ast.targets), 1)
        self.assertEqual(ast.targets[0].name, "CovidTarget")
        self.assertIn("pdb_id", ast.targets[0].properties)

    def test_parse_pipeline(self):
        code = """
        @optimize(ml_guided=true)
        pipeline ScreenPipeline {
            dataset lib = load_smiles("data.smi");
            filter Ro5(lib) {
                MW <= 500.0;
                LogP <= 5.0;
            }
            score Dock(Ro5) against CovidTarget {
                exhaustiveness: 8
            }
            export Dock to "out.csv";
        }
        """
        ast = Parser.from_source(code).parse()
        self.assertEqual(len(ast.pipelines), 1)
        p = ast.pipelines[0]
        self.assertEqual(p.name, "ScreenPipeline")
        self.assertEqual(len(p.statements), 4)
        self.assertIsInstance(p.statements[1], FilterStmtNode)
        self.assertIsInstance(p.statements[2], ScoreStmtNode)

    def test_syntax_error(self):
        bad_code = "pipeline Bad { dataset = ; }"
        with self.assertRaises(ParserError):
            Parser.from_source(bad_code).parse()


if __name__ == "__main__":
    unittest.main()
