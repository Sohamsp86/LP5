#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <ctime>
#include <chrono>
using namespace std;

// Bubble Sort Sequential
void bubbleSortSeq(int arr[], int n) {
    for(int i=0; i<n-1; i++) {
        for(int j=0; j<n-i-1; j++) {
            if(arr[j] > arr[j+1])
                swap(arr[j], arr[j+1]);
        }
    }
}

// Bubble Sort Parallel
void bubbleSortPar(int arr[], int n) {
    for(int i=0; i<n; i++) {
        #pragma omp parallel for
        for(int j=i%2; j<n-1; j+=2) {
            if(arr[j] > arr[j+1])
                swap(arr[j], arr[j+1]);
        }
    }
}   

// Merge function
void merge(int arr[], int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;
    int *L = new int[n1];
    int *R = new int[n2];

    for(int i=0; i<n1; i++) L[i] = arr[l + i];
    for(int i=0; i<n2; i++) R[i] = arr[m + 1 + i];

    int i = 0, j = 0, k = l;
    while(i < n1 && j < n2) 
        arr[k++] = (L[i] < R[j]) ? L[i++] : R[j++];

    while(i < n1) arr[k++] = L[i++];
    while(j < n2) arr[k++] = R[j++];

    delete[] L;
    delete[] R;
}

// Merge Sort Sequential
void mergeSortSeq(int arr[], int l, int r) {
    if(l < r) {
        int m = (l + r) / 2;
        mergeSortSeq(arr, l, m);
        mergeSortSeq(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}

// Merge Sort Parallel
void mergeSortPar(int arr[], int l, int r) {
    if(l < r) {
        int m = (l + r) / 2;
        #pragma omp parallel sections
        {
            #pragma omp section
            mergeSortPar(arr, l, m);

            #pragma omp section
            mergeSortPar(arr, m + 1, r);
        } 
        merge(arr, l, m, r);
    }
}

int main() {
    int n;
    cout << "Enter the number of elements: ";
    cin >> n;

    int *arr1 = new int[n];
    int *arr2 = new int[n];
    int *arr3 = new int[n];
    int *arr4 = new int[n];

    srand(time(0));
    for(int i=0; i<n; i++) {
        int val = rand() % 10000;
        arr1[i] = arr2[i] = arr3[i] = arr4[i] = val;
    }

    // Sequential Bubble Sort
    auto start = chrono::high_resolution_clock::now();
    bubbleSortSeq(arr1, n);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> bubbleSeqTime = end - start;
    cout << "Sequential Bubble Sort Time: " << bubbleSeqTime.count() << " sec" << endl;

    // Parallel Bubble Sort
    start = chrono::high_resolution_clock::now();
    bubbleSortPar(arr2, n);
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> bubbleParTime = end - start;
    cout << "Parallel Bubble Sort Time: " << bubbleParTime.count() << " sec" << endl;

    if (bubbleParTime.count() > 0)
        cout << "Bubble Sort Speedup: " << bubbleSeqTime.count() / bubbleParTime.count() << "x" << endl;

    // Sequential Merge Sort
    start = chrono::high_resolution_clock::now();
    mergeSortSeq(arr3, 0, n-1);
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> mergeSeqTime = end - start;
    cout << "Sequential Merge Sort Time: " << mergeSeqTime.count() << " sec" << endl;

    // Parallel Merge Sort
    start = chrono::high_resolution_clock::now();
    mergeSortPar(arr4, 0, n-1);
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> mergeParTime = end - start;
    cout << "Parallel Merge Sort Time: " << mergeParTime.count() << " sec" << endl;

    if (mergeParTime.count() > 0)
        cout << "Merge Sort Speedup: " << mergeSeqTime.count() / mergeParTime.count() << "x" << endl;

    delete[] arr1;
    delete[] arr2;
    delete[] arr3;
    delete[] arr4;

    return 0;
}