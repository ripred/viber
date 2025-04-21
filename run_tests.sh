#!/usr/bin/env bash
set -e
echo "\n--- pytest output ---"
pytest test_viber.py --disable-warnings -v | tee pytest_output.txt
