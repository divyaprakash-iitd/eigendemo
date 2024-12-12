#include <Eigen/Sparse>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>

using namespace Eigen;

// Function to read vector from file
template<typename T>
std::vector<T> readVectorFromFile(const std::string& filename) {
    std::vector<T> data;
    std::ifstream file(filename);
    T value;
    
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(1);
    }
    
    while (file >> value) {
        data.push_back(value);
    }
    
    file.close();
    return data;
}

int main() {
    // Read matrix dimensions
    int rows, cols;
    std::ifstream dimFile("dimensions.txt");
    if (!dimFile.is_open()) {
        std::cerr << "Failed to open dimensions file" << std::endl;
        return 1;
    }
    dimFile >> rows >> cols;
    dimFile.close();
    
    // Read CRS format data from files
    std::vector<double> values = readVectorFromFile<double>("values.txt");
    std::vector<int> inner_indices = readVectorFromFile<int>("inner_indices.txt");
    std::vector<int> outer_starts = readVectorFromFile<int>("outer_starts.txt");
    std::vector<double> b = readVectorFromFile<double>("rhs.txt");
    
    // Print some info about the matrix
    std::cout << "Matrix size: " << rows << " x " << cols << std::endl;
    std::cout << "Number of non-zero elements: " << values.size() << std::endl;
    
    // Create matrix using triplet list
    typedef Triplet<double> T;
    std::vector<T> tripletList;
    tripletList.reserve(values.size());
    
    // Convert CRS format to triplets
    for(int i = 0; i < rows; i++) {
        for(int j = outer_starts[i]; j < outer_starts[i+1]; j++) {
            tripletList.push_back(T(i, inner_indices[j], values[j]));
        }
    }

    // Create sparse matrix
    SparseMatrix<double> A(rows, cols);
    A.setFromTriplets(tripletList.begin(), tripletList.end());
    A.makeCompressed();

    // Create right hand side vector
    VectorXd b_eigen = Map<VectorXd>(b.data(), b.size());

    // Create BiCGSTAB solver
    BiCGSTAB<SparseMatrix<double>> solver;
    
    // Set solver parameters
    solver.setTolerance(1e-6);
    solver.setMaxIterations(1000);

    // Compute and solve
    solver.compute(A);
    VectorXd x = solver.solve(b_eigen);

    // Check if solve was successful
    if(solver.info() != Success) {
        std::cerr << "Solving failed!" << std::endl;
        return -1;
    }

    // Print solution and solver statistics
    std::cout << "Number of iterations: " << solver.iterations() << std::endl;
    std::cout << "Solution x = \n" << x << std::endl;
    std::cout << "Estimated error: " << solver.error() << std::endl;

    // Save solution to file
    std::ofstream solFile("solution.txt");
    if (solFile.is_open()) {
        solFile.precision(12);  // Set high precision output
        for(int i = 0; i < x.size(); i++) {
            solFile << x[i] << "\n";
        }
        solFile.close();
    }

    // Compute and print residual
    double residual = (A * x - b_eigen).norm();
    std::cout << "Residual: " << residual << std::endl;

    return 0;
}

