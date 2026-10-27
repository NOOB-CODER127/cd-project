#!/usr/bin/env python3
import sys
import os

# Ensure project root is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drugcode.cli import main

if __name__ == "__main__":
    main()
