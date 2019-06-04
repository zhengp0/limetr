# make file for pynlme class

clean:
	find . -name "__pycache__" -exec rm -rf "{}" \;
	find . -name "*.so" -delete