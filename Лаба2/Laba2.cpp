#include <iostream>
#include <vector>
#include <complex>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <immintrin.h>
extern "C" {
#include <cblas.h>
} // Я использовал openBLAS, ибо скачать Intel-овскую версию не удалось, не устанавливалась часть нужных файлов

using namespace std;

using complexf = complex<float>; // complexf вместо complex<float>, float, т.к. в С++ это число одинарной точности

const int N = 1024; // Размер матрицы N*N

// доступ к элементу плоской матрицы
inline complexf& m(vector<complexf>& a, int i, int j) { // inline — просьба компилятору встроить вызов прямо в код
    // complexf& — возвращаем ссылку, значит можно присваивать: m(a, i, j) = ...
    return a[i * N + j]; // строка i, столбец j => плоский индекс в одномерном векторе
}

inline const complexf& m(const vector<complexf>& a, int i, int j) { // const-версия для чтения (когда матрица передана как const)
    return a[i * N + j];
}

// заполнение случайными числами
void genMatrix(vector<complexf>& a) { // принимаем по ссылке — меняем на месте
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            m(a, i, j) = complexf(
                float(rand()) / RAND_MAX, // вещественная часть: случайное [0,1)
                float(rand()) / RAND_MAX // мнимая часть: случайное [0,1)
            );
        }
    }
}

// обычное умножение матриц
void multiplySimple(const vector<complexf>& a,
                    const vector<complexf>& b,
                    vector<complexf>& c)
{
    for (int i = 0; i < N; i++) { // Строки А
        for (int j = 0; j < N; j++) { // Столбцы В

            complexf sum = 0; // Накопитель суммы

            for (int k = 0; k < N; k++) {
                sum += m(a, i, k) * m(b, k, j); // Скалярное произведение строки i и столбца j
            }

            m(c, i, j) = sum; // Записываем результат
        }
    }
}

// BLAS
void multiplyBLAS(const vector<complexf>& a,
                  const vector<complexf>& b,
                  vector<complexf>& c)
{
    // Вычисляет: C = alpha * A * B + beta * C
    // При наших значениях: C = 1*A*B + 0*C = A*B

    complexf alpha(1.0f, 0.0f); // коэффициент при A*B — единица (реальная)
    complexf beta(0.0f, 0.0f); // коэффициент при C — ноль (обнуляем C)

    cblas_cgemm(
        CblasRowMajor, // данные хранятся по строкам
        CblasNoTrans, // матрицу A не транспонировать
        CblasNoTrans, // матрицу B не транспонировать
        N, N, N, // M, N, K: размеры. C(M×N) = A(M×K) × B(K×N)
        &alpha, // указатель на alpha
        a.data(), N, // данные A, ведущая размерность
        b.data(), N, // данные B, ведущая размерность
        &beta, // указатель на beta
        c.data(), N // данные C (результат), ведущая размерность
    );
}

// используем матрицу в SOA-формате
// Это позволяет грузить 8 вещественных частей одним _mm256_loadu_ps
struct MatrixSOA {
    vector<float> re;  // вещественные части всех N*N элементов
    vector<float> im;  // мнимые части всех N*N элементов

    MatrixSOA() : re(N * N, 0.f), im(N * N, 0.f) {} // оба массива длиной N*N, заполнены нулями

    // Конвертируем из vector<complexf> в SOA
    MatrixSOA(const vector<complexf>& src) : re(N * N), im(N * N) {
        for (int i = 0; i < N * N; i++) {
            re[i] = src[i].real(); // .real() — метод complex-ов, возвращает вещественную часть
            im[i] = src[i].imag(); // .imag() — возвращает мнимую часть
        }
    }

    // Конвертируем обратно в vector<complexf> для проверки корректности
    void toComplex(vector<complexf>& dst) const {
        dst.resize(N * N); // resize — устанавливает размер вектора
        for (int i = 0; i < N * N; i++) 
            dst[i] = complexf(re[i], im[i]);
    }
};

// Оптимизированное перемножение
void multiplyOptimized(const vector<complexf>& a,
                       const vector<complexf>& b,
                       vector<complexf>& c)
{
    MatrixSOA A(a), B(b); // Конвертируем в SOA
    MatrixSOA C;  // результат, изначально нули

    // Порядок i→k→j по заветам агентов
    #pragma omp parallel for
    for (int i = 0; i < N; i++) { // перебираем строки A и C
        for (int k = 0; k < N; k++) { // перебираем столбцы A = строки B

            // A[i][k] — скалярные значения, грузим один раз в регистр
            float ar = A.re[i * N + k]; // вещественная часть
            float ai = A.im[i * N + k]; // мнимая часть

            // Транслируем скаляр в вектор
            // __m256 — тип для 256-битного регистра (8 раз по float)
            __m256 va_re = _mm256_set1_ps(ar);  // [ar, ar, ar, ar, ar, ar, ar, ar]
            __m256 va_im = _mm256_set1_ps(ai);  // [ai, ai, ai, ai, ai, ai, ai, ai]

            // Указатели на начало строк k матриц B и C
            const float* br = &B.re[k * N]; // адрес начала строки k в массиве вещ. частей B
            const float* bi = &B.im[k * N]; // адрес начала строки k в массиве мнимых частей B
            float*       cr = &C.re[i * N]; // адрес начала строки i в результате C
            float*       ci = &C.im[i * N];

            // Основной цикл: шаг 8 элементов за итерацию (один AVX2-регистр)
            for (int j = 0; j <= N - 8; j += 8) { // N-8 — чтобы не выйти за границу при загрузке 8 элементов
                // Грузим 8 вещественных и 8 мнимых частей строки B[k]
                __m256 vb_re = _mm256_loadu_ps(br + j); // _mm256_loadu_ps(ptr) — загружает 8 последовательных float из памяти
                __m256 vb_im = _mm256_loadu_ps(bi + j);

                // Грузим текущие накопленные значения C[i]
                __m256 vc_re = _mm256_loadu_ps(cr + j);
                __m256 vc_im = _mm256_loadu_ps(ci + j);

                // вещественная часть произведения
                // _mm256_fmadd_ps(a, b, c) = a * b + c — слитое умножение-добавление, без промежуточного округления, что точнее, и быстрее за счёт того, что это одна операция
                vc_re = _mm256_fmadd_ps(va_re, vb_re, vc_re); // vc_re += ar * br
                // _mm256_fnmadd_ps(a, b, c) = -(a * b) + c — слитое отклонение-умножение-добавление
                vc_re = _mm256_fnmadd_ps(va_im, vb_im, vc_re); // vc_re -= ai * bi

                // Мнимая часть произведения
                vc_im = _mm256_fmadd_ps(va_re, vb_im, vc_im);   // vc_im += ar * bi
                vc_im = _mm256_fmadd_ps(va_im, vb_re, vc_im);   // vc_im += ai * br

                // Записываем обратно в память
                _mm256_storeu_ps(cr + j, vc_re); // _mm256_storeu_ps(ptr, vec) — записывает 8 float из регистра в память
                _mm256_storeu_ps(ci + j, vc_im);
            }
        }
    }

    // Конвертируем результат обратно в vector<complexf>
    C.toComplex(c);
}

// измерение времени
template<typename Func> // передаем Func — любой вызываемый объект
double measure(Func f) {

    // high_resolution_clock выбран, так как BLAS слишком быстрый для не таких точных часов

    auto start = chrono::high_resolution_clock::now(); // время до начала

    f(); // вызываем переданную

    auto end = chrono::high_resolution_clock::now(); // время после

    chrono::duration<double> diff = end - start; // Разность в секндах = прошедшему времени

    return diff.count(); // возвращаем прошедшее время
}

int main() {
    // Объявляем матрицы как плоские векторы длиной N*N
    vector<complexf> A(N * N);
    vector<complexf> B(N * N);

    vector<complexf> C1(N * N); // результат Simple
    vector<complexf> C2(N * N); // результат BLAS
    vector<complexf> C3(N * N); // результат Optimized

    srand(time(0)); // инициализируем рандом текущим временем

    genMatrix(A); // заполняем A случайными комплексными числами
    genMatrix(B); // и заполняем В случайными комплексными числами

    double complexity = 2.0 * N * N * N;

    cout << "Complexity: " << complexity << endl << endl;

    // Замеряем каждый метод, передавая лямбду в measure()
    double t1 = measure([&]() {
       multiplySimple(A, B, C1);
    });

    double t2 = measure([&]() {
        multiplyBLAS(A, B, C2);
    });

    double t3 = measure([&]() {
        multiplyOptimized(A, B, C3);
    });

    // complexity = операций/сек, делим на 1e6, получаем мегафлопсы
    double p1 = (complexity / t1) * 1e-6;
    double p2 = (complexity / t2) * 1e-6;
    double p3 = (complexity / t3) * 1e-6;

    // Выводим время и производительность каждого метода
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
