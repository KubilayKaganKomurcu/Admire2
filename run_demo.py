"""
Quick Demo Script for AdMIRe 2.0
================================
Run this to test the system with mock data.

Usage:
    python run_demo.py
"""

import os
import sys

# Ensure we're in the right directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import run_demo

if __name__ == "__main__":
    print("\n🚀 Starting AdMIRe 2.0 Demo...\n")
    run_demo()


