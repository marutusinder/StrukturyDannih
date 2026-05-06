#include <iostream>
#include <vector>
#include <complex>
#include <chrono>
#include <random>
#include <windows.h>
#include <immintrin.h>
extern "C" {
#include <cblas.h>
}

using namespace std;
using complexf = complex<float>;

const int N = 1024;

void generateMatrix(vector<complexf>& A){
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N * N; i++){
        A[i] = complexf(dist(gen), dist(gen));
    }
}

void multiplyNaive(const vector<complexf>& A, const vector<complexf>& B, vector<complexf>& C){
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            complexf sum = 0;
            for (int k = 0; k < N; k++){
                sum += A[i * N + k] * B[j * N + k];
            }
            C[i * N + j] = sum;
        }
    }
}

void multiplyOptimized(const vector<complexf>& A,
                       const vector<complexf>& B,
                       vector<complexf>& C)
{
    for (int i = 0; i < N * N; i++)
        C[i] = 0;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {

            complexf sum = 0;

            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[j * N + k];
            }

            C[i * N + j] = sum;
        }
    }
}

template<typename Func>
double measure(Func f){
    auto start = chrono::high_resolution_clock::now();
    f();
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = end - start;
    return diff.count();
}

int main(){
    SetConsoleOutputCP(65001);
    SetConsoleCP(65001);
    vector<complexf> A(N*N), B(N*N), C(N*N);

    generateMatrix(A);
    generateMatrix(B);

    double c = 2.0 * N * N * N;
    cout << "Сложность алгоритма c: " << c << endl;
    
    double t1, t2, t3;

    t1 = measure([&](){
        multiplyNaive(A, B, C);
    });

    t2 = measure([&](){
        complexf alpha(1.0f, 0.0f);
        complexf beta(0.0f, 0.0f);

        cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, &alpha, A.data(), N, B.data(), N, &beta, C.data(), N);
    });

    t3 = measure([&](){
        multiplyOptimized(A, B, C);
    });

    double p1 = (c / t1) * 0.000001;
    double p2 = (c / t2) * 0.000001;
    double p3 = (c / t3) * 0.000001;

    cout << "Линейное время: " << t1 << " секунд, MFlops: " << p1 << endl;
    cout << "Время BLAS: " << t2 << " секунд, MFlops: " << p2 << endl;
    cout << "Оптимизированное: " << t3 << " секунд, MFlops: " << p3 << endl;

    cout<<"Нажмите Enter, чтобы выйти";
    cin.get();
    return 0;
}