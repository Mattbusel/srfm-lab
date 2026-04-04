#pragma once
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <initializer_list>
#include <algorithm>
#include <cmath>
#include <vector>
#include <memory>
#include <cassert>
#include <new>

namespace linalg {

// Aligned allocation (for SIMD)
static constexpr size_t kAlign = 32; // AVX2 requires 32-byte alignment

template <typename T>
T* aligned_alloc_t(size_t n) {
    if (n == 0) return nullptr;
    void* ptr = nullptr;
#ifdef _WIN32
    ptr = _aligned_malloc(sizeof(T) * n, kAlign);
#else
    if (posix_memalign(&ptr, kAlign, sizeof(T) * n) != 0) ptr = nullptr;
#endif
    if (!ptr) throw std::bad_alloc();
    return static_cast<T*>(ptr);
}

template <typename T>
void aligned_free_t(T* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// Row-major heap-allocated matrix
template <typename T>
class Matrix {
public:
    using value_type = T;

    // Constructors
    Matrix() : rows_(0), cols_(0), data_(nullptr) {}

    Matrix(size_t rows, size_t cols)
        : rows_(rows), cols_(cols), data_(nullptr)
    {
        if (rows * cols > 0) {
            data_ = aligned_alloc_t<T>(rows * cols);
            std::memset(data_, 0, sizeof(T) * rows * cols);
        }
    }

    Matrix(size_t rows, size_t cols, T fill)
        : rows_(rows), cols_(cols), data_(nullptr)
    {
        if (rows * cols > 0) {
            data_ = aligned_alloc_t<T>(rows * cols);
            std::fill(data_, data_ + rows * cols, fill);
        }
    }

    // Init from initializer list (row-major)
    Matrix(size_t rows, size_t cols, std::initializer_list<T> il)
        : rows_(rows), cols_(cols), data_(nullptr)
    {
        size_t n = rows * cols;
        if (n > 0) {
            data_ = aligned_alloc_t<T>(n);
            size_t i = 0;
            for (T v : il) { if (i < n) data_[i++] = v; }
            while (i < n) data_[i++] = T{};
        }
    }

    // Copy
    Matrix(const Matrix& o)
        : rows_(o.rows_), cols_(o.cols_), data_(nullptr)
    {
        size_t n = rows_ * cols_;
        if (n > 0) {
            data_ = aligned_alloc_t<T>(n);
            std::memcpy(data_, o.data_, sizeof(T) * n);
        }
    }

    // Move
    Matrix(Matrix&& o) noexcept
        : rows_(o.rows_), cols_(o.cols_), data_(o.data_)
    { o.rows_ = o.cols_ = 0; o.data_ = nullptr; }

    Matrix& operator=(const Matrix& o) {
        if (this == &o) return *this;
        if (rows_ * cols_ != o.rows_ * o.cols_) {
            aligned_free_t(data_);
            data_ = (o.rows_ * o.cols_ > 0) ? aligned_alloc_t<T>(o.rows_ * o.cols_) : nullptr;
        }
        rows_ = o.rows_; cols_ = o.cols_;
        if (data_) std::memcpy(data_, o.data_, sizeof(T) * rows_ * cols_);
        return *this;
    }

    Matrix& operator=(Matrix&& o) noexcept {
        aligned_free_t(data_);
        rows_ = o.rows_; cols_ = o.cols_; data_ = o.data_;
        o.rows_ = o.cols_ = 0; o.data_ = nullptr;
        return *this;
    }

    ~Matrix() { aligned_free_t(data_); }

    // Element access
    T&       at(size_t r, size_t c)       { check_bounds(r, c); return data_[r * cols_ + c]; }
    const T& at(size_t r, size_t c) const { check_bounds(r, c); return data_[r * cols_ + c]; }

    T&       operator()(size_t r, size_t c)       noexcept { return data_[r * cols_ + c]; }
    const T& operator()(size_t r, size_t c) const noexcept { return data_[r * cols_ + c]; }

    T*       row(size_t r)       noexcept { return data_ + r * cols_; }
    const T* row(size_t r) const noexcept { return data_ + r * cols_; }

    T*       data()       noexcept { return data_; }
    const T* data() const noexcept { return data_; }

    size_t rows() const noexcept { return rows_; }
    size_t cols() const noexcept { return cols_; }
    size_t size() const noexcept { return rows_ * cols_; }
    bool   empty() const noexcept { return size() == 0; }

    // Shape validation
    bool is_square()     const noexcept { return rows_ == cols_; }
    bool is_vector()     const noexcept { return rows_ == 1 || cols_ == 1; }
    bool is_compatible(const Matrix& o) const noexcept { return rows_ == o.rows_ && cols_ == o.cols_; }

    // Identity matrix
    static Matrix identity(size_t n) {
        Matrix m(n, n, T{0});
        for (size_t i = 0; i < n; ++i) m(i, i) = T{1};
        return m;
    }

    // Diagonal from vector
    static Matrix diag(const std::vector<T>& d) {
        Matrix m(d.size(), d.size(), T{0});
        for (size_t i = 0; i < d.size(); ++i) m(i,i) = d[i];
        return m;
    }

    // Scalar ops
    Matrix& operator+=(T s) { for (size_t i=0; i<size(); ++i) data_[i]+=s; return *this; }
    Matrix& operator-=(T s) { for (size_t i=0; i<size(); ++i) data_[i]-=s; return *this; }
    Matrix& operator*=(T s) { for (size_t i=0; i<size(); ++i) data_[i]*=s; return *this; }
    Matrix& operator/=(T s) { T inv=T{1}/s; for (size_t i=0; i<size(); ++i) data_[i]*=inv; return *this; }

    // Naive matrix ops (SIMD versions in simd_ops.cpp)
    Matrix operator+(const Matrix& o) const {
        check_same_shape(o);
        Matrix r(rows_, cols_);
        for (size_t i=0; i<size(); ++i) r.data_[i] = data_[i] + o.data_[i];
        return r;
    }
    Matrix operator-(const Matrix& o) const {
        check_same_shape(o);
        Matrix r(rows_, cols_);
        for (size_t i=0; i<size(); ++i) r.data_[i] = data_[i] - o.data_[i];
        return r;
    }

    // Naive matmul (reference implementation)
    Matrix matmul_naive(const Matrix& B) const {
        if (cols_ != B.rows_)
            throw std::invalid_argument("matmul: incompatible shapes");
        Matrix C(rows_, B.cols_, T{0});
        for (size_t i = 0; i < rows_; ++i)
            for (size_t k = 0; k < cols_; ++k)
                for (size_t j = 0; j < B.cols_; ++j)
                    C(i,j) += (*this)(i,k) * B(k,j);
        return C;
    }

    // Transpose
    Matrix transpose() const {
        Matrix t(cols_, rows_);
        for (size_t i = 0; i < rows_; ++i)
            for (size_t j = 0; j < cols_; ++j)
                t(j,i) = (*this)(i,j);
        return t;
    }

    // Frobenius norm
    T norm_frobenius() const {
        T sum = T{0};
        for (size_t i=0; i<size(); ++i) sum += data_[i]*data_[i];
        return std::sqrt(sum);
    }

    // Trace
    T trace() const {
        if (!is_square()) throw std::invalid_argument("trace: non-square matrix");
        T sum = T{0};
        for (size_t i=0; i<rows_; ++i) sum += (*this)(i,i);
        return sum;
    }

    // Fill
    void fill(T v) { std::fill(data_, data_ + size(), v); }
    void zero()    { std::memset(data_, 0, sizeof(T) * size()); }

    // Submatrix view (copy)
    Matrix submatrix(size_t r0, size_t c0, size_t nr, size_t nc) const {
        Matrix m(nr, nc);
        for (size_t i=0; i<nr; ++i)
            for (size_t j=0; j<nc; ++j)
                m(i,j) = (*this)(r0+i, c0+j);
        return m;
    }

    // Column/row vector extraction
    std::vector<T> col_vec(size_t c) const {
        std::vector<T> v(rows_);
        for (size_t i=0; i<rows_; ++i) v[i] = (*this)(i,c);
        return v;
    }
    std::vector<T> row_vec(size_t r) const {
        return std::vector<T>(data_ + r*cols_, data_ + (r+1)*cols_);
    }

    void set_col(size_t c, const std::vector<T>& v) {
        for (size_t i=0; i<rows_ && i<v.size(); ++i) (*this)(i,c) = v[i];
    }

private:
    size_t rows_, cols_;
    T*     data_;

    void check_bounds(size_t r, size_t c) const {
        if (r >= rows_ || c >= cols_)
            throw std::out_of_range("Matrix: index out of range");
    }
    void check_same_shape(const Matrix& o) const {
        if (rows_ != o.rows_ || cols_ != o.cols_)
            throw std::invalid_argument("Matrix: shape mismatch");
    }
};

using MatrixF = Matrix<float>;
using MatrixD = Matrix<double>;

} // namespace linalg
