# make file for pynlme class

.PHONY: clean, tests

build: setup.py
	python setup.py build

install: setup.py
	python check_requirements.py
	python setup.py install

sdist: setup.py
	python setup.py sdist

tests:
	python tests/check_utils.py
	python tests/check_limetr.py

clean:
	find . -name "*.so*" | xargs rm -rf
	find . -name "*.pyc" | xargs rm -rf
	find . -name "__pycache__" | xargs rm -rf
	find . -name "build" | xargs rm -rf
	find . -name "dist" | xargs rm -rf
	find . -name "MANIFEST" | xargs rm -rf
	rm -rf ./lib

uninstall:
	find $(CONDA_PREFIX)/lib/ -name "*limetr*" | xargs rm -rf
