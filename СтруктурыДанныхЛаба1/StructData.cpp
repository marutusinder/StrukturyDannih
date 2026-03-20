#include <iostream>
#include <map>
#include <algorithm>
using namespace std;

void getPrimeFactors(int num, map<int,int>& link4Nums){// ищем простые множители числа и их количество (Prime factor - простой множитель с англ.)
    for(int i = 2; i * i <= num; i++){
        if(num % i == 0){
            int iQuantity = 0;
            while(num % i == 0){
                num /= i;
                iQuantity++;
            }
            link4Nums[i] += iQuantity;
        }
    }
    if(num > 1){
        link4Nums[num]++;
    }
}

int main(){
    long long A, B, n=0;
    cout<<"A: ";
    cin>>A;
    cout<<"B: ";
    cin>>B;
    if(A<2 || A>2000000000 || B<2 || B>2000000000){
        cout<<"invalid nums!";
        return 0;
    }
    map <int, int> primeFactorsA;
    getPrimeFactors(A, primeFactorsA);
    for(auto [primeFactor, primeFactorQuantity] : primeFactorsA){
        int temp = B;
        int b = 0;
        while (temp % primeFactor == 0) {
            temp /= primeFactor;
            b++;
        }
        if(b==0){
            cout << -1;
            return 0;
        }
        long long need = (primeFactorQuantity + b - 1) / b;
        n = max(n, need);
    }
    cout<<"n: "<< n;
}
