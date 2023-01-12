cythonize:
	cython -3 --no-docstrings --fast-fail --output-file csrc/scalar.c pysrc/scalar.py
binlib:
	clang -shared -fpic $(shell python3.8-config --cflags) -c csrc/scalar.c -o bin/scalar.cpython-39-darwin.so