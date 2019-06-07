# make file for pynlme class


build: setup.py
	python3 setup.py build

install: setup.py
	python3 setup.py install

sdist: setup.py
	python3 setup.py sdist

test:
	python3 tests/check_utils.py
	python3 tests/check_limetr.py

clean:
	find . -name "*.so*" | xargs rm -rf
	find . -name "*.pyc" | xargs rm -rf
	find . -name "__pycache__" | xargs rm -rf
	find . -name "build" | xargs rm -rf
