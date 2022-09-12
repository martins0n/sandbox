#include "libscalarprod.h"
#include <stdio.h>
#include <assert.h>


int main() {
    float a[3] = {1, 2, 3};
    float b[3] = {4, 5, 6};
    int c = scalarprod(a, b, 3);

    printf("%d", c);
    assert(c == 32);
    return 0;
}