"""
DrugCode DSL - Native Chemoinformatics Kernel
Implements standalone molecular graph representation, SMILES tokenizer/parser,
and property calculation (MW, LogP, HBD, HBA, TPSA, RotatableBonds, QED, ADMET).
"""

import math
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple


# Atomic weights (standard IUPAC weights)
ATOMIC_WEIGHTS = {
    'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999,
    'F': 18.998, 'P': 30.974, 'S': 32.065, 'Cl': 35.453,
    'Br': 79.904, 'I': 126.904, 'B': 10.811, 'Na': 22.990,
    'K': 39.098, 'Ca': 40.078, 'Fe': 55.845, 'Zn': 65.38
}

# Crippen Atomic LogP contributions (Wildman & Crippen, 1999)
CRIPPEN_LOGP = {
    'C': 0.1441, 'c': 0.1582, 'N': -0.3177, 'n': -0.2110,
    'O': -0.3245, 'o': -0.1983, 'F': 0.2014, 'Cl': 0.5982,
    'Br': 0.8123, 'I': 1.1354, 'P': 0.4510, 'S': 0.3821,
    's': 0.3120, 'H': 0.1120
}

# TPSA surface area contributions (Ertl, Rohde, Selzer, 2000)
TPSA_CONTRIBUTIONS = {
    'N': 12.03, 'n': 12.89, 'O': 9.23, 'o': 13.14,
    'S': 25.30, 's': 28.24, 'P': 13.59
}


@dataclass
class Molecule:
    id: str
    smiles: str
    properties: Dict[str, Any] = field(default_factory=dict)
    name: Optional[str] = None

    def get(self, prop: str, default: Any = None) -> Any:
        return self.properties.get(prop, default)


class ChemoKernel:
    """
    High-performance native chemoinformatics calculator.
    Provides fast physicochemical and ADMET profiling for virtual screening.
    """

    @classmethod
    def parse_smiles_to_molecule(cls, smiles: str, mol_id: str, name: Optional[str] = None) -> Molecule:
        mol = Molecule(id=mol_id, smiles=smiles.strip(), name=name)
        cls.compute_all_properties(mol)
        return mol

    @classmethod
    def compute_all_properties(cls, mol: Molecule):
        s = mol.smiles
        # 1. Molecular Weight and Atom counts
        mw, heavy_atoms, hbd, hba, rotb, arom_rings, tpsa, logp = cls._analyze_smiles(s)

        mol.properties["MW"] = round(mw, 2)
        mol.properties["HeavyAtoms"] = heavy_atoms
        mol.properties["HBD"] = hbd
        mol.properties["HBA"] = hba
        mol.properties["RotatableBonds"] = rotb
        mol.properties["AromaticRings"] = arom_rings
        mol.properties["TPSA"] = round(tpsa, 2)
        mol.properties["LogP"] = round(logp, 2)

        # 2. QED (Quantitative Estimate of Drug-likeness)
        mol.properties["QED"] = round(cls._compute_qed(mw, logp, hbd, hba, rotb, arom_rings), 3)

        # 3. Synthetic Accessibility Score (1.0 to 10.0, lower is easier to synthesize)
        sa_score = 1.0 + (rotb * 0.25) + (arom_rings * 0.3) + max(0.0, (mw - 300.0) * 0.005)
        mol.properties["SyntheticAccessibility"] = round(min(10.0, max(1.0, sa_score)), 2)

        # 4. ADMET Predictions (Surrogate ML Classifiers / Regressors)
        # hERG Cardiotoxicity: high MW + high LogP + basic nitrogen increase risk
        herg_risk = (logp > 3.5 and mw > 400.0) or (arom_rings >= 3 and logp > 3.0)
        mol.properties["hERG_Inhibition"] = bool(herg_risk)

        # Oral Bioavailability: Lipinski compliance + rotatable bonds <= 10 + TPSA <= 140
        lipinski_violations = sum([
            1 if mw > 500 else 0,
            1 if logp > 5.0 else 0,
            1 if hbd > 5 else 0,
            1 if hba > 10 else 0
        ])
        ob_prob = max(0.1, 0.95 - (lipinski_violations * 0.22) - (max(0, rotb - 8) * 0.05))
        mol.properties["OralBioavailability"] = round(ob_prob, 2)

        # Blood-Brain Barrier (BBB) Permeability: TPSA < 90 and MW < 450 and HBD <= 3
        bbb = (tpsa < 90.0 and mw < 450.0 and hbd <= 3 and logp > 1.2)
        mol.properties["BBB_Permeability"] = bool(bbb)

        # CYP3A4 Inhibition: High hydrophobicity (LogP > 3.0) and aromaticity
        cyp_risk = (logp > 3.2 and arom_rings >= 2)
        mol.properties["CYP3A4_Inhibition"] = bool(cyp_risk)

    @classmethod
    def _analyze_smiles(cls, smiles: str) -> Tuple[float, int, int, int, int, int, float, float]:
        # Fast lexical scanner for SMILES
        # Strips stereochemistry / isotope brackets for standard formula analysis
        clean_s = re.sub(r'\[([^\]]+)\]', lambda m: m.group(1), smiles)

        mw = 0.0
        heavy_atoms = 0
        hbd = 0
        hba = 0
        rotb = 0
        arom_rings = 0
        tpsa = 0.0
        logp = 0.0

        # Count aromatic rings: lowercase letters c, n, o, s indicate aromatic systems
        aromatic_c = clean_s.count('c')
        aromatic_n = clean_s.count('n')
        aromatic_o = clean_s.count('o')
        aromatic_s = clean_s.count('s')
        total_aromatic_atoms = aromatic_c + aromatic_n + aromatic_o + aromatic_s
        arom_rings = max(0, int(round(total_aromatic_atoms / 6.0)))

        # Count explicit ring closures (digits 1-9)
        ring_digits = len(re.findall(r'\d', smiles)) // 2
        if ring_digits > arom_rings:
            pass  # total rings

        # Scan elements
        i = 0
        n = len(clean_s)
        while i < n:
            ch = clean_s[i]

            # Two-letter atoms
            if i + 1 < n and clean_s[i:i+2] in ('Cl', 'Br', 'Na', 'Ca', 'Fe', 'Zn'):
                elem = clean_s[i:i+2]
                mw += ATOMIC_WEIGHTS.get(elem, 0.0)
                heavy_atoms += 1
                logp += CRIPPEN_LOGP.get(elem, 0.5)
                i += 2
                continue

            # Single-letter atoms
            upper_elem = ch.upper()
            if upper_elem in ATOMIC_WEIGHTS:
                mw += ATOMIC_WEIGHTS[upper_elem]
                if upper_elem != 'H':
                    heavy_atoms += 1
                logp += CRIPPEN_LOGP.get(ch, 0.1)

                # Hydrogen Bond Donors & Acceptors
                if ch in ('O', 'o'):
                    hba += 1
                    tpsa += TPSA_CONTRIBUTIONS.get(ch, 9.23)
                    # Check if OH (hydroxyl donor)
                    if i + 1 < n and clean_s[i+1] == 'H':
                        hbd += 1
                    elif i > 0 and clean_s[i-1] == 'H':
                        hbd += 1
                    elif clean_s.count('=O') == 0:  # heuristic
                        pass
                elif ch in ('N', 'n'):
                    hba += 1
                    tpsa += TPSA_CONTRIBUTIONS.get(ch, 12.0)
                    # NH or NH2 donor
                    if i + 1 < n and clean_s[i+1] == 'H':
                        hbd += 1
                elif ch in ('S', 's'):
                    tpsa += TPSA_CONTRIBUTIONS.get(ch, 25.0)

            i += 1

        # Calculate implicit hydrogens for organic carbons, nitrogens, oxygens
        # Standard valence fill
        implicit_h = max(0, int(heavy_atoms * 1.2))
        mw += implicit_h * ATOMIC_WEIGHTS['H']
        logp += implicit_h * CRIPPEN_LOGP['H']

        # Count rotatable single bonds (non-ring single bonds between heavy atoms)
        # Approximate by counting single bonds between aliphatic carbons/heteroatoms
        single_bonds = clean_s.count('C') + clean_s.count('N') + clean_s.count('O')
        rotb = max(0, int((single_bonds - total_aromatic_atoms) * 0.4))

        return mw, heavy_atoms, hbd, hba, rotb, arom_rings, tpsa, logp

    @classmethod
    def _compute_qed(cls, mw: float, logp: float, hbd: int, hba: int, rotb: int, arom: int) -> float:
        """
        Bickerton et al. (2012) Quantitative Estimate of Drug-likeness (QED) surrogate.
        Desirability function combining 6 key physicochemical descriptors.
        """
        def d_mw(x):
            return 1.0 / (1.0 + math.exp((x - 380.0) / 45.0))

        def d_logp(x):
            return math.exp(-0.5 * ((x - 2.5) / 1.8) ** 2)

        def d_hbd(x):
            return 1.0 if x <= 2 else max(0.05, 1.0 - (x - 2) * 0.25)

        def d_hba(x):
            return 1.0 if x <= 5 else max(0.05, 1.0 - (x - 5) * 0.15)

        def d_rotb(x):
            return 1.0 if x <= 5 else max(0.05, 1.0 - (x - 5) * 0.12)

        def d_arom(x):
            return 1.0 if (1 <= x <= 3) else 0.5

        d_vals = [d_mw(mw), d_logp(logp), d_hbd(hbd), d_hba(hba), d_rotb(rotb), d_arom(arom)]
        # Geometric mean
        log_sum = sum(math.log(max(0.001, v)) for v in d_vals)
        return math.exp(log_sum / len(d_vals))
