# make file for the fortran module

.PHONY: clean, tests

special_mat: special_mat.f90
	f2py -c -m special_mat special_mat.f90 -L/usr/local/lib -llapack -lblas

tests: tests/check_utils.py
	python tests/check_utils.py

clean:
	find . -name '__pycache__' | xargs rm -rf
	find . -name '*.so' | xargs rm -rf
