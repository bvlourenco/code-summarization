#!/bin/bash

# Python dataset
python3 dataset.py --language python --code_snippet_file ../../data/python/test_originalcode.txt --summary_file ../../data/python/test_summary.txt --type test --pre_processing True
python3 dataset.py --language python --code_snippet_file ../../data/python/validation_originalcode.txt --summary_file ../../data/python/validation_summary.txt --type validation --pre_processing True
python3 dataset.py --language python --code_snippet_file ../../data/python/train_originalcode.txt --summary_file ../../data/python/train_summary.txt --type train --pre_processing True

# Java dataset
python3 dataset.py --language java --code_summary_file ../../data/java/test.json --type test
python3 dataset.py --language java --code_summary_file ../../data/java/valid.json --type validation
python3 dataset.py --language java --code_summary_file ../../data/java/train.json --type train



# CodeSearchNet
python3 dataset.py --language python --code_summary_file ../../data/CodeSearchNet/dataset/python/test.jsonl --type test --codesearchnet True
python3 dataset.py --language python --code_summary_file ../../data/CodeSearchNet/dataset/python/valid.jsonl --type validation --codesearchnet True
python3 dataset.py --language python --code_summary_file ../../data/CodeSearchNet/dataset/python/train.jsonl --type train --codesearchnet True

python3 dataset.py --language java --code_summary_file ../../data/CodeSearchNet/dataset/java/test.jsonl --type test --codesearchnet True
python3 dataset.py --language java --code_summary_file ../../data/CodeSearchNet/dataset/java/valid.jsonl --type validation --codesearchnet True
python3 dataset.py --language java --code_summary_file ../../data/CodeSearchNet/dataset/java/train.jsonl --type train --codesearchnet True

python3 dataset.py --language go --code_summary_file ../../data/CodeSearchNet/dataset/go/test.jsonl --type test --codesearchnet True
python3 dataset.py --language go --code_summary_file ../../data/CodeSearchNet/dataset/go/valid.jsonl --type validation --codesearchnet True
python3 dataset.py --language go --code_summary_file ../../data/CodeSearchNet/dataset/go/train.jsonl --type train --codesearchnet True

python3 dataset.py --language javascript --code_summary_file ../../data/CodeSearchNet/dataset/javascript/test.jsonl --type test --codesearchnet True
python3 dataset.py --language javascript --code_summary_file ../../data/CodeSearchNet/dataset/javascript/valid.jsonl --type validation --codesearchnet True
python3 dataset.py --language javascript --code_summary_file ../../data/CodeSearchNet/dataset/javascript/train.jsonl --type train --codesearchnet True

python3 dataset.py --language ruby --code_summary_file ../../data/CodeSearchNet/dataset/ruby/test.jsonl --type test --codesearchnet True
python3 dataset.py --language ruby --code_summary_file ../../data/CodeSearchNet/dataset/ruby/valid.jsonl --type validation --codesearchnet True
python3 dataset.py --language ruby --code_summary_file ../../data/CodeSearchNet/dataset/ruby/train.jsonl --type train --codesearchnet True

python3 dataset.py --language php --code_summary_file ../../data/CodeSearchNet/dataset/php/test.jsonl --type test --codesearchnet True
python3 dataset.py --language php --code_summary_file ../../data/CodeSearchNet/dataset/php/valid.jsonl --type validation --codesearchnet True
python3 dataset.py --language php --code_summary_file ../../data/CodeSearchNet/dataset/php/train.jsonl --type train --codesearchnet True