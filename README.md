# 作业一: 并行矩阵乘法
## 问题陈述
编写⼀个基于oneAPI的C++/SYCL程序来执行矩阵乘法操作。
## 项目简介
本项目基于Intel提供的oneAPI工具集，编写C++/SYCL程序利用基于SYCL的编程模型在GPU上实现矩阵乘法的计算。
## 技术框架
### Intel oneAPI

## 具体实现
```cpp
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
            // 分块
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
```

## 运行结果
随机生成1024*1024的两个矩阵，进行乘法运算，并将并行运算的结果与常规串行运算结果相比较，输出如下：

```
Job has been submitted to Intel(R) DevCloud and will execute soon.

Job ID                    Name             User            Time Use S Queue
------------------------- ---------------- --------------- -------- - -----
2432120.v-qsvr-1           ...ub-singleuser u207452         00:00:41 R jupyterhub     
2432161.v-qsvr-1           ...x_multiply.sh u207452                0 Q batch          

Waiting for Output ████████████████████ Done⬇

########################################################################
#      Date:           Sat 18 Nov 2023 10:13:11 PM PST
#    Job ID:           2432161.v-qsvr-1.aidevcloud
#      User:           u207452
# Resources:           cput=75:00:00,neednodes=1:gpu:ppn=2,nodes=1:gpu:ppn=2,walltime=06:00:00
########################################################################

## u207452 is compiling SYCL_Essentials Module2 -- SYCL Program Structure sample - 7 of 7 matrix_multiply.cpp
Time taken for scalar matrix multiplication:: 1.77183 seconds
Time taken for parallel matrix multiplication:: 0.148906 seconds
Matrix multiplication successfully run on the device

########################################################################
# End of output for job 2432161.v-qsvr-1.aidevcloud
# Date: Sat 18 Nov 2023 10:13:23 PM PST
########################################################################

Job Completed in 20 seconds.
```

可以看到，并行运算与串行运算结果一致，且具有很高的执行效率。

## 学到了什么
通过本次实验，我初次接触了高性能计算以及Intel提供的高性能计算平台，利用oneAPI这一开发工具套件，上手实操完成了对矩阵乘法的并行优化。此外，我切实感受了SYCL编程模型及相应库函数的强大之处，对我未来的工作大有裨益。
