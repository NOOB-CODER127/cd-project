"""
Unit tests for DrugCode Lexer
"""

import unittest
from drugcode.lexer import Lexer, TokenType, LexerError


class TestDrugCodeLexer(unittest.TestCase):
    def test_keywords_and_symbols(self):
        code = "target pipeline dataset filter score rank export against to"
        tokens = Lexer(code).tokenize()
        types = [t.type for t in tokens[:-1]]
        expected = [
            TokenType.KW_TARGET, TokenType.KW_PIPELINE, TokenType.KW_DATASET,
            TokenType.KW_FILTER, TokenType.KW_SCORE, TokenType.KW_RANK,
            TokenType.KW_EXPORT, TokenType.KW_AGAINST, TokenType.KW_TO
        ]
        self.assertEqual(types, expected)

    def test_numbers_and_signs(self):
        code = "[-10.5, 500, 500.5, 1e-4]"
        tokens = Lexer(code).tokenize()
        values = [t.value for t in tokens if t.type == TokenType.NUMBER]
        self.assertEqual(values, [-10.5, 500, 500.5, 0.0001])

    def test_comments_and_whitespace(self):
        code = """
        // Single line comment
        # Another comment
        /* Multi-line
           comment block */
        MW <= 500.0;
        """
        tokens = Lexer(code).tokenize()
        types = [t.type for t in tokens[:-1]]
        self.assertEqual(types, [TokenType.IDENTIFIER, TokenType.LE, TokenType.NUMBER, TokenType.SEMICOLON])

    def test_unterminated_string_error(self):
        with self.assertRaises(LexerError):
            Lexer('"unterminated string').tokenize()


if __name__ == "__main__":
    unittest.main()
