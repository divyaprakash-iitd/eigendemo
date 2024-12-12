#include <Eigen/Sparse>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>

using namespace Eigen;

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
    std::vector<int> column_indices = readVectorFromFile<int>("column_indices.txt");
    std::vector<int> row_pointers = readVectorFromFile<int>("row_pointers.txt");
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
        for(int j = row_pointers[i]; j < row_pointers[i+1]; j++) {
            tripletList.push_back(T(i, column_indices[j], values[j]));
        }
    }

    // Rest of the code remains the same...
    SparseMatrix<double> A(rows, cols);
    A.setFromTriplets(tripletList.begin(), tripletList.end());
    A.makeCompressed();

    VectorXd b_eigen = Map<VectorXd>(b.data(), b.size());
    BiCGSTAB<SparseMatrix<double>> solver;
    solver.setTolerance(1e-6);
    solver.setMaxIterations(1000);
    solver.compute(A);
    VectorXd x = solver.solve(b_eigen);

    if(solver.info() != Success) {
        std::cerr << "Solving failed!" << std::endl;
        return -1;
    }

    std::cout << "Number of iterations: " << solver.iterations() << std::endl;
    std::cout << "Estimated error: " << solver.error() << std::endl;

    std::ofstream solFile("solution.txt");
    if (solFile.is_open()) {
        solFile.precision(12);
        for(int i = 0; i < x.size(); i++) {
            solFile << x[i] << "\n";
        }
        solFile.close();
    }

    double residual = (A * x - b_eigen).norm();
    std::cout << "Residual: " << residual << std::endl;

    return 0;
}

