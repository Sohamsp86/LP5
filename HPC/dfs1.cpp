#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

using namespace std;

int visited[1000];
int visit[1000];
int qu[1000], front = 0, rear = 0; 
int stk[1000], top = -1;

// Sequential BFS
void bfs_sequential(int cost[1000][1000], int n, int start_vertex) {
    fill(visited, visited + n, 0);  // Reset visited array
    fill(visit, visit + n, 0);      // Reset visit array
    cout << "Sequential BFS starting from vertex " << start_vertex << ": ";
    visited[start_vertex] = 1;
    cout << start_vertex << " ";
    qu[rear++] = start_vertex;

    while (front < rear) {
        int v = qu[front++];

        for (int j = 0; j < n; j++) {
            if (cost[v][j] != 0 && !visited[j] && !visit[j]) {
                visit[j] = 1;
                qu[rear++] = j;
                cout << j << " ";
                visited[j] = 1;
            }
        }
    }
    cout << endl;
}

// Parallel BFS
void bfs_parallel(int cost[1000][1000], int n, int start_vertex) {
    fill(visited, visited + n, 0);  // Reset visited array
    fill(visit, visit + n, 0);      // Reset visit array
    cout << "Parallel BFS starting from vertex " << start_vertex << ": ";
    visited[start_vertex] = 1;
    cout << start_vertex << " ";
    qu[rear++] = start_vertex;

    while (front < rear) {
        int v = qu[front++];

        #pragma omp parallel for
        for (int j = 0; j < n; j++) {
            if (cost[v][j] != 0 && !visited[j] && !visit[j]) {
                #pragma omp critical
                {
                    visit[j] = 1;
                    qu[rear++] = j;
                    cout << j << " ";
                    visited[j] = 1;
                }
            }
        }
    }
    cout << endl;
}

// Sequential DFS
void dfs_sequential(int cost[1000][1000], int n, int start_vertex) {
    fill(visited, visited + n, 0);  
    fill(visit, visit + n, 0);     
    cout << "Sequential DFS starting from vertex " << start_vertex << ": ";
    visited[start_vertex] = 1;
    cout << start_vertex << " ";
    stk[++top] = start_vertex;

    while (top >= 0) {
        int v = stk[top--];

        for (int j = n - 1; j >= 0; j--) {
            if (cost[v][j] != 0 && !visited[j] && !visit[j]) {
                visit[j] = 1;
                stk[++top] = j;
                cout << j << " ";
                visited[j] = 1;
            }
        }
    }
    cout << endl;
}

// Parallel DFS
void dfs_parallel(int cost[1000][1000], int n, int start_vertex) {
    fill(visited, visited + n, 0);  
    fill(visit, visit + n, 0);      
    cout << "Parallel DFS starting from vertex " << start_vertex << ": ";
    visited[start_vertex] = 1;
    cout << start_vertex << " ";
    stk[++top] = start_vertex;

    while (top >= 0) {
        int v = stk[top--];

        #pragma omp parallel for
        for (int j = n - 1; j >= 0; j--) {
            if (cost[v][j] != 0 && !visited[j] && !visit[j]) {
                #pragma omp critical
                {
                    visit[j] = 1;
                    stk[++top] = j;
                    cout << j << " ";
                    visited[j] = 1;
                }
            }
        }
    }
    cout << endl;
}

int main() {
    int n = 1000;
     int cost[1000][1000]; 
       for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {  
            int edge = rand() % 2;  
            cost[i][j] = edge;
            cost[j][i] = edge;  
        }
        }

    int start_vertex;
    
    // BFS
    cout << "Enter initial vertex for Sequential BFS: ";
    cin >> start_vertex;
    auto start_time_bfs_seq = chrono::high_resolution_clock::now();
    bfs_sequential(cost, n, start_vertex);
    auto end_time_bfs_seq = chrono::high_resolution_clock::now();
    cout << "Time taken for Sequential BFS: " 
         << chrono::duration_cast<chrono::microseconds>(end_time_bfs_seq - start_time_bfs_seq).count() 
         << " microseconds" << endl;

    cout << "Enter initial vertex for Parallel BFS: ";
    cin >> start_vertex;
    auto start_time_bfs_par = chrono::high_resolution_clock::now();
    bfs_parallel(cost, n, start_vertex);
    auto end_time_bfs_par = chrono::high_resolution_clock::now();
    cout << "Time taken for Parallel BFS: " 
         << chrono::duration_cast<chrono::microseconds>(end_time_bfs_par - start_time_bfs_par).count() 
         << " microseconds" << endl;
cout<<"speed up "<<(chrono::duration_cast<chrono::microseconds>(end_time_bfs_seq - start_time_bfs_seq).count())/(chrono::duration_cast<chrono::microseconds>(end_time_bfs_par - start_time_bfs_par).count() )<<endl;
    // DFS
    cout << "Enter initial vertex for Sequential DFS: ";
    cin >> start_vertex;
    auto start_time_dfs_seq = chrono::high_resolution_clock::now();
    dfs_sequential(cost, n, start_vertex);
    auto end_time_dfs_seq = chrono::high_resolution_clock::now();
    cout << "Time taken for Sequential DFS: " 
         << chrono::duration_cast<chrono::microseconds>(end_time_dfs_seq - start_time_dfs_seq).count() 
         << " microseconds" << endl;

    cout << "Enter initial vertex for Parallel DFS: ";
    cin >> start_vertex;
    auto start_time_dfs_par = chrono::high_resolution_clock::now();
    dfs_parallel(cost, n, start_vertex);
    auto end_time_dfs_par = chrono::high_resolution_clock::now();
    cout << "Time taken for Parallel DFS: " << chrono::duration_cast<chrono::microseconds>(end_time_dfs_par - start_time_dfs_par).count() << " microseconds" << endl;
double speed_up_bfs = static_cast<double>(chrono::duration_cast<chrono::microseconds>(end_time_bfs_seq - start_time_bfs_seq).count()) / 
                      chrono::duration_cast<chrono::microseconds>(end_time_bfs_par - start_time_bfs_par).count();
cout << "Speed up for BFS: " << speed_up_bfs << endl;

double speed_up_dfs = static_cast<double>(chrono::duration_cast<chrono::microseconds>(end_time_dfs_seq - start_time_dfs_seq).count()) / 
                      chrono::duration_cast<chrono::microseconds>(end_time_dfs_par - start_time_dfs_par).count();
cout << "Speed up for DFS: " << speed_up_dfs << endl;
    return 0;
}


