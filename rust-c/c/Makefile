.PHONY: static
static:
	clang -c libscalarprod.c -o static/libscalarprod.o

.PHONY: dynamic
dynamic:
	clang -shared -fpic libscalarprod.c -o dynamic/libscalarprod.dylib

compile: static dynamic
	clang main.c static/libscalarprod.o -o static-linked-main.out
	clang main.c dynamic/libscalarprod.dylib -o dynamic-linked-main.out