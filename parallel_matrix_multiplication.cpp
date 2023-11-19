#include <sycl/sycl.hpp>
#include <random>
#include <chrono>

constexpr size_t MATRIX_SIZE = 1024;
constexpr size_t BLOCK_SIZE = 16;

// 随机生成输入矩阵
void generateRandomMatrix(float* matrix, size_t rows, size_t cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (size_t i = 0; i < rows * cols; ++i) {
        matrix[i] = dis(gen);
    }
}

// 常规串行计算作为baseline
void matrixMultiplyScalar(float* result, const float* matrixA, const float* matrixB) {
    for (size_t i = 0; i < MATRIX_SIZE; ++i) {
        for (size_t j = 0; j < MATRIX_SIZE; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < MATRIX_SIZE; ++k) {
                sum += matrixA[i * MATRIX_SIZE + k] * matrixB[k * MATRIX_SIZE + j];
            }
            result[i * MATRIX_SIZE + j] = sum;
        }
    }
}

class MatrixMultiplyKernel {
public:
    MatrixMultiplyKernel(sycl::queue& queue, float* result, const float* matrixA, const float* matrixB)
        : queue_(queue), result_(result), matrixA_(matrixA), matrixB_(matrixB) {}

    void operator()() {
        sycl::buffer<float, 2> resultBuffer(result_, sycl::range<2>(MATRIX_SIZE, MATRIX_SIZE));
        sycl::buffer<const float, 2> matrixABuffer(matrixA_, sycl::range<2>(MATRIX_SIZE, MATRIX_SIZE));
        sycl::buffer<const float, 2> matrixBBuffer(matrixB_, sycl::range<2>(MATRIX_SIZE, MATRIX_SIZE));

        queue_.submit([&](sycl::handler& cgh) {
            auto resultAcc = resultBuffer.get_access<sycl::access::mode::write>(cgh);
            auto matrixAAcc = matrixABuffer.get_access<sycl::access::mode::read>(cgh);
            auto matrixBAcc = matrixBBuffer.get_access<sycl::access::mode::read>(cgh);

            cgh.parallel_for<class BlockMatrixMultiplyKernel>(
                sycl::nd_range<2>(sycl::range<2>(MATRIX_SIZE, MATRIX_SIZE), sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE)),
                [=](sycl::nd_item<2> item) {
                    size_t globalRow = item.get_global_id(0);
                    size_t globalCol = item.get_global_id(1);
                    
                    float sum = 0.0f;

                    // 循环计算块乘法
                    for (size_t k = 0; k < MATRIX_SIZE; k++) {
                        sum += matrixAAcc[globalRow][k] * matrixBAcc[k][globalCol];
                    }

                    // 将局部结果写回全局内存
                    resultAcc[item.get_global_id()] = sum;
                });
        });
    }

private:
    sycl::queue& queue_;
    float* result_;
    const float* matrixA_;
    const float* matrixB_;
};

bool compare(const float* matrixA, const float* matrixB, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        if (fabs(matrixA[i] - matrixB[i]) >= 1e-3f) {
            return false;
        }
    }
    return true;
}

int main() {
    sycl::queue queue(sycl::gpu_selector_v);

    // 使用USM分配内存
    float* matrixA = sycl::malloc_shared<float>(MATRIX_SIZE * MATRIX_SIZE, queue);
    float* matrixB = sycl::malloc_shared<float>(MATRIX_SIZE * MATRIX_SIZE, queue);
    float* matrixC = sycl::malloc_shared<float>(MATRIX_SIZE * MATRIX_SIZE, queue);
    float* resultScalar = sycl::malloc_shared<float>(MATRIX_SIZE * MATRIX_SIZE, queue);
    float* resultParallel = sycl::malloc_shared<float>(MATRIX_SIZE * MATRIX_SIZE, queue);

    generateRandomMatrix(matrixA, MATRIX_SIZE, MATRIX_SIZE);
    generateRandomMatrix(matrixB, MATRIX_SIZE, MATRIX_SIZE);

    auto startSerial = std::chrono::high_resolution_clock::now();
    matrixMultiplyScalar(resultScalar, matrixA, matrixB);
    auto endSerial = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationSerial = endSerial - startSerial;

    std::cout << "Time taken for scalar matrix multiplication:: " << durationSerial.count() << " seconds" << std::endl;

    MatrixMultiplyKernel matrixMultiplyParallel(queue, resultParallel, matrixA, matrixB);

    auto startParallel = std::chrono::high_resolution_clock::now();
    matrixMultiplyParallel();
    auto endParallel = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationParallel = endParallel - startParallel;
    
    std::cout << "Time taken for parallel matrix multiplication:: " << durationParallel.count() << " seconds" << std::endl;
    
    bool match = compare(resultParallel, resultScalar, MATRIX_SIZE * MATRIX_SIZE);
    if (match) {
        std::cout << "Matrix multiplication successfully run on the device"
              << "\n";
    } else {
        std::cout
            << "*********************************************Verification Failed. Results are "
               "not matched**************************"
            << "\n";
    }
    
    free(matrixA, queue);
    free(matrixB, queue);
    free(matrixC, queue);
    free(resultScalar, queue);
    free(resultParallel, queue);

    return 0;
}