#include <iostream>
#include <immintrin.h>

int main()
{
    __m256 a;
    a = __m256(0);
    std::cout << a[0] << std::endl;
//    std::cout << sizeof(__m256) << std::endl << sizeof(__m256d) << std::endl << sizeof(float) << std::endl << sizeof(double) << std::endl;
}