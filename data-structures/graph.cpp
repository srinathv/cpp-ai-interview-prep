#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <unordered_set>

class Graph {
private:
    int V;
    std::vector<std::vector<int>> adj;
    
public:
    Graph(int vertices) : V(vertices), adj(vertices) {}
    
    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u); // Undirected graph
    }
    
    // BFS traversal
    void BFS(int start) {
        std::vector<bool> visited(V, false);
        std::queue<int> q;
        
        visited[start] = true;
        q.push(start);
        
        std::cout << "BFS: ";
        while (!q.empty()) {
            int v = q.front();
            q.pop();
            std::cout << v << " ";
            
            for (int neighbor : adj[v]) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    q.push(neighbor);
                }
            }
        }
        std::cout << std::endl;
    }
    
    // DFS helper
    void DFSUtil(int v, std::vector<bool>& visited) {
        visited[v] = true;
        std::cout << v << " ";
        
        for (int neighbor : adj[v]) {
            if (!visited[neighbor]) {
                DFSUtil(neighbor, visited);
            }
        }
    }
    
    // DFS traversal
    void DFS(int start) {
        std::vector<bool> visited(V, false);
        std::cout << "DFS: ";
        DFSUtil(start, visited);
        std::cout << std::endl;
    }
    
    // Check if path exists
    bool hasPath(int src, int dest) {
        if (src == dest) return true;
        std::vector<bool> visited(V, false);
        std::queue<int> q;
        
        visited[src] = true;
        q.push(src);
        
        while (!q.empty()) {
            int v = q.front();
            q.pop();
            
            for (int neighbor : adj[v]) {
                if (neighbor == dest) return true;
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    q.push(neighbor);
                }
            }
        }
        return false;
    }
};

// Interview: Number of islands (2D grid DFS)
void dfs(std::vector<std::vector<char>>& grid, int i, int j) {
    if (i < 0 || i >= grid.size() || j < 0 || j >= grid[0].size() || grid[i][j] == '0')
        return;
    grid[i][j] = '0';
    dfs(grid, i+1, j);
    dfs(grid, i-1, j);
    dfs(grid, i, j+1);
    dfs(grid, i, j-1);
}

int numIslands(std::vector<std::vector<char>>& grid) {
    int count = 0;
    for (size_t i = 0; i < grid.size(); ++i) {
        for (size_t j = 0; j < grid[0].size(); ++j) {
            if (grid[i][j] == '1') {
                count++;
                dfs(grid, i, j);
            }
        }
    }
    return count;
}

int main() {
    Graph g(6);
    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(1, 3);
    g.addEdge(2, 4);
    g.addEdge(3, 5);
    g.addEdge(4, 5);
    
    g.BFS(0);
    g.DFS(0);
    
    std::cout << "Path from 0 to 5: " << (g.hasPath(0, 5) ? "Yes" : "No") << std::endl;
    
    std::vector<std::vector<char>> grid = {
        {'1','1','0','0','0'},
        {'1','1','0','0','0'},
        {'0','0','1','0','0'},
        {'0','0','0','1','1'}
    };
    std::cout << "Number of islands: " << numIslands(grid) << std::endl;
    
    return 0;
}
