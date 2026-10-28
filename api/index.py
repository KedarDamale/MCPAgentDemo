"""Serverless function entrypoint for Vercel deployment."""

import sys
import os

# Ensure src/ is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from mcpagentdemo.app import app
