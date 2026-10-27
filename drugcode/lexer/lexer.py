"""
DrugCode DSL - Lexical Analyzer (Lexer)
Scans source code characters and produces a stream of typed tokens.
"""

from typing import List, Optional
from .tokens import Token, TokenType, KEYWORDS, SourceLocation


class LexerError(Exception):
    def __init__(self, message: str, location: SourceLocation):
        super().__init__(f"Lexer error at {location}: {message}")
        self.message = message
        self.location = location


class Lexer:
    def __init__(self, source: str, file_path: Optional[str] = None):
        self.source = source
        self.file_path = file_path
        self.length = len(source)
        self.pos = 0
        self.line = 1
        self.column = 1

    def _current_loc(self) -> SourceLocation:
        return SourceLocation(line=self.line, column=self.column, file_path=self.file_path)

    def _peek(self, offset: int = 0) -> str:
        idx = self.pos + offset
        if idx < self.length:
            return self.source[idx]
        return '\0'

    def _advance(self) -> str:
        if self.pos >= self.length:
            return '\0'
        ch = self.source[self.pos]
        self.pos += 1
        if ch == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return ch

    def _skip_whitespace_and_comments(self):
        while self.pos < self.length:
            ch = self._peek()

            # Whitespace
            if ch in (' ', '\t', '\r', '\n'):
                self._advance()
                continue

            # Single-line comment: // or #
            if (ch == '/' and self._peek(1) == '/') or ch == '#':
                while self.pos < self.length and self._peek() != '\n':
                    self._advance()
                continue

            # Multi-line comment: /* ... */
            if ch == '/' and self._peek(1) == '*':
                self._advance() # /
                self._advance() # *
                start_loc = self._current_loc()
                closed = False
                while self.pos < self.length:
                    if self._peek() == '*' and self._peek(1) == '/':
                        self._advance() # *
                        self._advance() # /
                        closed = True
                        break
                    self._advance()
                if not closed:
                    raise LexerError("Unterminated multi-line comment", start_loc)
                continue

            break

    def _read_string(self) -> Token:
        start_loc = self._current_loc()
        quote_char = self._advance() # " or '
        chars = []

        while self.pos < self.length:
            ch = self._peek()
            if ch == quote_char:
                self._advance()
                value = ''.join(chars)
                return Token(TokenType.STRING, value, f'"{value}"', start_loc)
            if ch == '\\':
                self._advance()
                esc = self._advance()
                if esc == 'n':
                    chars.append('\n')
                elif esc == 't':
                    chars.append('\t')
                elif esc == 'r':
                    chars.append('\r')
                elif esc == '\\':
                    chars.append('\\')
                elif esc == quote_char:
                    chars.append(quote_char)
                else:
                    chars.append(esc)
                continue
            if ch == '\n':
                raise LexerError("Unterminated string literal (newline before closing quote)", start_loc)
            chars.append(self._advance())

        raise LexerError("Unterminated string literal (reached EOF)", start_loc)

    def _read_number(self) -> Token:
        start_loc = self._current_loc()
        start_pos = self.pos
        is_float = False

        if self._peek() == '-':
            self._advance()

        while self.pos < self.length and self._peek().isdigit():
            self._advance()

        # Check for decimal part
        if self._peek() == '.' and self._peek(1).isdigit():
            is_float = True
            self._advance() # '.'
            while self.pos < self.length and self._peek().isdigit():
                self._advance()

        # Check for scientific notation: e.g. 1e-4, 2.5E+2
        if self._peek() in ('e', 'E'):
            is_float = True
            self._advance()
            if self._peek() in ('+', '-'):
                self._advance()
            while self.pos < self.length and self._peek().isdigit():
                self._advance()

        lexeme = self.source[start_pos:self.pos]
        val = float(lexeme) if is_float else int(lexeme)
        return Token(TokenType.NUMBER, val, lexeme, start_loc)

    def _read_identifier_or_keyword(self) -> Token:
        start_loc = self._current_loc()
        start_pos = self.pos

        while self.pos < self.length and (self._peek().isalnum() or self._peek() == '_'):
            self._advance()

        lexeme = self.source[start_pos:self.pos]

        # Check keyword map
        token_type = KEYWORDS.get(lexeme, TokenType.IDENTIFIER)
        if token_type == TokenType.KW_TRUE:
            return Token(TokenType.BOOLEAN, True, lexeme, start_loc)
        elif token_type == TokenType.KW_FALSE:
            return Token(TokenType.BOOLEAN, False, lexeme, start_loc)

        return Token(token_type, lexeme, lexeme, start_loc)

    def tokenize(self) -> List[Token]:
        tokens: List[Token] = []

        while self.pos < self.length:
            self._skip_whitespace_and_comments()
            if self.pos >= self.length:
                break

            ch = self._peek()
            loc = self._current_loc()

            # String literal
            if ch in ('"', "'"):
                tokens.append(self._read_string())
                continue

            # Number literal (or negative number when preceded by sign)
            if ch.isdigit():
                tokens.append(self._read_number())
                continue

            # Negative number check: '-' followed by digit, when previous token was an operator or delimiter
            if ch == '-' and self._peek(1).isdigit():
                # Check if this minus is unary (at start or after operator/comma/lparen/bracket)
                if not tokens or tokens[-1].type in (
                    TokenType.ASSIGN, TokenType.COMMA, TokenType.LPAREN, TokenType.LBRACKET,
                    TokenType.LBRACE, TokenType.COLON, TokenType.EQ, TokenType.NEQ,
                    TokenType.LT, TokenType.LE, TokenType.GT, TokenType.GE,
                    TokenType.KW_AND, TokenType.KW_OR
                ):
                    tokens.append(self._read_number())
                    continue

            # Identifiers and keywords
            if ch.isalpha() or ch == '_':
                tokens.append(self._read_identifier_or_keyword())
                continue

            # Two-character operators
            if ch == '=':
                self._advance()
                if self._peek() == '=':
                    self._advance()
                    tokens.append(Token(TokenType.EQ, "==", "==", loc))
                else:
                    tokens.append(Token(TokenType.ASSIGN, "=", "=", loc))
                continue

            if ch == '!':
                self._advance()
                if self._peek() == '=':
                    self._advance()
                    tokens.append(Token(TokenType.NEQ, "!=", "!=", loc))
                else:
                    tokens.append(Token(TokenType.KW_NOT, "not", "!", loc))
                continue

            if ch == '<':
                self._advance()
                if self._peek() == '=':
                    self._advance()
                    tokens.append(Token(TokenType.LE, "<=", "<=", loc))
                else:
                    tokens.append(Token(TokenType.LT, "<", "<", loc))
                continue

            if ch == '>':
                self._advance()
                if self._peek() == '=':
                    self._advance()
                    tokens.append(Token(TokenType.GE, ">=", ">=", loc))
                else:
                    tokens.append(Token(TokenType.GT, ">", ">", loc))
                continue

            # Single-character tokens
            single_tokens = {
                '(': TokenType.LPAREN,
                ')': TokenType.RPAREN,
                '{': TokenType.LBRACE,
                '}': TokenType.RBRACE,
                '[': TokenType.LBRACKET,
                ']': TokenType.RBRACKET,
                ',': TokenType.COMMA,
                ':': TokenType.COLON,
                ';': TokenType.SEMICOLON,
                '.': TokenType.DOT,
                '@': TokenType.AT,
                '+': TokenType.PLUS,
                '-': TokenType.MINUS,
                '*': TokenType.STAR,
                '/': TokenType.SLASH,
            }

            if ch in single_tokens:
                self._advance()
                tokens.append(Token(single_tokens[ch], ch, ch, loc))
                continue

            # Unexpected character
            self._advance()
            raise LexerError(f"Unexpected character: {ch!r}", loc)

        tokens.append(Token(TokenType.EOF, None, "", self._current_loc()))
        return tokens
