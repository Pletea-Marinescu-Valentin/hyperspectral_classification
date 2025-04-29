# Makefile for common tasks

.PHONY: test clean report

test:
	python -m unittest discover -s tests

clean:
	find . -type f -name '*.pyc' -delete
	required_dirs="results/"; for dir in $$required_dirs; do rm -rf $$dir/*; done

report:
	python scripts/report_generator.py