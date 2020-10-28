#include "MatrixMulLib.h"
int main()
{
    int length = 512;
    int **Atemp = new int *[length];
    int **Btemp = new int *[length];
    int **Ctemp = new int *[length];
    for (int i = 0; i < length; ++i)
    {
        Atemp[i] = new int[length];
        Btemp[i] = new int[length];
        Ctemp[i] = new int[length];
    }
    for (int i = 0; i < length; ++i)
        for (int j = 0; j < length; ++j)
            Atemp[i][j] = 1;
    for (int i = 0; i < length; ++i)
        for (int j = 0; j < length; ++j)
            Btemp[i][j] = 2;
    matrix_multiply(Atemp, Btemp, Ctemp, length, length, length);
}