#include <iostream>
#include <vector>
#include <complex>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <windows.h>
extern "C" {
#include <cblas.h>
} // Я использовал openBLAS (а он на C вроде как написан), ибо скачать Intel-овскую версию не удалось, не устанавливалась часть нужных файлов

using namespace std;

using complexf = complex<float>;

const int N = 1024;
const int BLOCK = 64;

// доступ к элементу плоской матрицы
inline complexf& m(vector<complexf>& a, int i, int j) {
    return a[i * N + j];
}

inline const complexf& m(const vector<complexf>& a, int i, int j) {
    return a[i * N + j];
}

// заполнение случайными числами
void genMatrix(vector<complexf>& a) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            m(a, i, j) = complexf(
                float(rand()) / RAND_MAX,
                float(rand()) / RAND_MAX
            );
        }
    }
}

// обычное умножение матриц
void multiplySimple(const vector<complexf>& a,
                    const vector<complexf>& b,
                    vector<complexf>& c)
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {

            complexf sum = 0;

            for (int k = 0; k < N; k++) {
                sum += m(a, i, k) * m(b, k, j);
            }

            m(c, i, j) = sum;
        }
    }
}

// BLAS
void multiplyBLAS(const vector<complexf>& a,
                  const vector<complexf>& b,
                  vector<complexf>& c)
{
    complexf alpha(1.0f, 0.0f);
    complexf beta(0.0f, 0.0f);

    cblas_cgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        N, N, N,
        &alpha,
        a.data(), N,
        b.data(), N,
        &beta,
        c.data(), N
    );
}

// своя оптимизация (спойлер: 30% не добил)
void multiplyOptimized(const vector<complexf>& a,
                       const vector<complexf>& b,
                       vector<complexf>& c)
{
    fill(c.begin(), c.end(), complexf(0.0f, 0.0f));

    // блочное перемножение
    for (int ii = 0; ii < N; ii += BLOCK) {
        for (int kk = 0; kk < N; kk += BLOCK) {
            for (int jj = 0; jj < N; jj += BLOCK) {
                for (int i = ii; i < min(ii + BLOCK, N); i++) {
                    for (int k = kk; k < min(kk + BLOCK, N); k++) {
                        complexf temp = m(a, i, k);
                        for (int j = jj; j < min(jj + BLOCK, N); j++) {

                            m(c, i, j) += temp * m(b, k, j);
                        }
                    }
                }
            }
        }
    }
}

// измерение времени
template<typename Func>
double measure(Func f) {

    auto start = chrono::high_resolution_clock::now();

    f();

    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double> diff = end - start;

    return diff.count();
}

int main() {

    SetConsoleOutputCP(65001);

    vector<complexf> A(N * N);
    vector<complexf> B(N * N);

    vector<complexf> C1(N * N);
    vector<complexf> C2(N * N);
    vector<complexf> C3(N * N);

    srand(time(0)); 

    genMatrix(A);
    genMatrix(B);

    double complexity = 2.0 * N * N * N;

    cout << "Complexity: " << complexity << endl << endl;

    double t1 = measure([&]() {
        multiplySimple(A, B, C1);
    });

    double t2 = measure([&]() {
        multiplyBLAS(A, B, C2);
    });

    double t3 = measure([&]() {
        multiplyOptimized(A, B, C3);
    });

    double p1 = (complexity / t1) * 1e-6;
    double p2 = (complexity / t2) * 1e-6;
    double p3 = (complexity / t3) * 1e-6;

    cout << "Simple: " << t1 << " sec; "
         << p1 << " MFlops" << endl;

    cout << "BLAS: " << t2 << " sec; "
         << p2 << " MFlops" << endl;

    cout << "Optimized: " << t3 << " sec; "
         << p3 << " MFlops" << endl;

    cout << "\nPress Enter...";
    cin.get();

    return 0;
}