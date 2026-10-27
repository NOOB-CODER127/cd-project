# DrugCode: Formal Language & Compiler Specification

**DrugCode** is a domain-specific compiler and programming language tailored for high-throughput virtual screening and computational drug discovery workflows.

---

## 1. Formal Grammar (Extended Backus-Naur Form - EBNF)

```ebnf
Program         ::= ( TargetDecl | PipelineDecl )* EOF ;

TargetDecl      ::= "target" IDENTIFIER "{" TargetProperty* "}" ;
TargetProperty  ::= IDENTIFIER ":" Expression ( "," | ";" )? ;

Annotation      ::= "@" IDENTIFIER ( "(" ( IDENTIFIER "=" Expression ( "," IDENTIFIER "=" Expression )* )? ")" )? ;

PipelineDecl    ::= Annotation* "pipeline" IDENTIFIER "{" Statement* "}" ;

Statement       ::= DatasetStmt
                  | FilterStmt
                  | ScoreStmt
                  | RankStmt
                  | ExportStmt ;

DatasetStmt     ::= "dataset" IDENTIFIER "=" ( "load_smiles" | "load_sdf" ) "(" Expression ")" ";" ;

FilterStmt      ::= "filter" IDENTIFIER "(" IDENTIFIER ")" "{" ( Expression ";" )* "}" ;

ScoreStmt       ::= "score" IDENTIFIER "(" IDENTIFIER ")" "against" IDENTIFIER "{" ScoreProperty* "}" ;
ScoreProperty   ::= IDENTIFIER ":" Expression ( "," | ";" )? ;

RankStmt        ::= "rank" IDENTIFIER "(" IDENTIFIER ")" "{" RankProperty* "}" ;
RankProperty    ::= ( "order_by" ":" Expression ( "asc" | "desc" )? ( "," | ";" )? )
                  | ( "top_k" ":" INTEGER ( "," | ";" )? ) ;

ExportStmt      ::= "export" IDENTIFIER "to" Expression ";" ;

Expression      ::= OrExpr ;
OrExpr          ::= AndExpr ( "or" AndExpr )* ;
AndExpr         ::= EqualityExpr ( "and" EqualityExpr )* ;
EqualityExpr    ::= RelationalExpr ( ( "==" | "!=" ) RelationalExpr )* ;
RelationalExpr  ::= AdditiveExpr ( ( "<" | "<=" | ">" | ">=" ) AdditiveExpr )* ;
AdditiveExpr    ::= MultiplicativeExpr ( ( "+" | "-" ) MultiplicativeExpr )* ;
MultiplicativeExpr ::= UnaryExpr ( ( "*" | "/" ) UnaryExpr )* ;
UnaryExpr       ::= ( "not" | "-" ) UnaryExpr | PrimaryExpr ;
PrimaryExpr     ::= NUMBER
                  | STRING
                  | BOOLEAN
                  | "[" ( Expression ( "," Expression )* )? "]"
                  | "(" Expression ")"
                  | IDENTIFIER ( "." IDENTIFIER )? ;
```

---

## 2. Type System & Semantics

DrugCode features static type checking and domain-specific typing:

### Primitive Types
- `number`: 64-bit IEEE floating point value (e.g., `MW`, `LogP`, `affinity`).
- `boolean`: Logical `true` / `false` flag (e.g., `hERG_Inhibition`, `BBB_Permeability`).
- `string`: File paths, receptor identifiers, scoring engine names.
- `list[T]`: Coordinate vectors (`[x, y, z]`) and residue lists (`["HIS41", "CYS145"]`).

### Domain Compound Types
- `Target`: Macromolecular protein receptor target with binding pocket definition.
- `MoleculeStream`: Ordered collection of molecular candidate records flowing between stages.
- `Property`: Chemoinformatics descriptor bound to a molecule record.

---

## 3. Intermediate Representation (IR)

The compiler lowers the AST into a dual-level Intermediate Representation:
1. **Workflow DAG**: Nodes represent compute kernels (`IRDatasetLoad`, `IRFilter`, `IRScore`, `IRRank`, `IRExport`), connected by directed dataflow edges.
2. **3-Address Code (3AC)**: Linear instruction sequence annotated with compiler cost metrics $[cost, selectivity]$.

---

## 4. ML-Guided Optimization Formulation

### Theoretical Model
Let a screening pipeline contain $N$ screening stages $S_1, S_2, \dots, S_N$.
- Each stage $i$ has latency cost $c_i > 0$ (ms/molecule) and pass-rate selectivity $s_i \in (0, 1]$.
- For an initial library of $M$ molecules evaluated in stage order $\pi = (\pi_1, \pi_2, \dots, \pi_N)$, the expected total runtime is:

$$T(\pi) = M \cdot \left[ c_{\pi_1} + \sum_{k=2}^{N} \left( \prod_{j=1}^{k-1} s_{\pi_j} \right) c_{\pi_k} \right]$$

### Cascade Reordering Theorem
For independent screening filters and scoring operators whose predicates are commutative conjunctions, $T(\pi)$ is minimized when stages are sorted in ascending order of their **Cost-to-Rejection Ratio**:

$$\text{RankMetric}(i) = \frac{c_i}{1 - s_i}$$

### ML Selectivity Surrogate
Because true selectivity $s_i$ depends on the chemical distribution of the input library, DrugCode employs a learned regression surrogate:

$$\hat{s}_i = \sigma\left( \mathbf{W}_i^\top \mathbf{x}_{\text{dataset}} + b_i \right)$$

where $\mathbf{x}_{\text{dataset}} = [\overline{\text{MW}}, \overline{\text{LogP}}, \overline{\text{HBD}}, \overline{\text{HBA}}, \overline{\text{RotB}}, \overline{\text{TPSA}}, \overline{\text{Arom}}, \text{FracDrugLike}]$.
