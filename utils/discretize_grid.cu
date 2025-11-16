#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>


__constant__ int const_num_gaussians;
__constant__ int const_grid_size_z = 96;
__constant__ int const_grid_size_x = 96;
__constant__ int const_grid_size_y = 96;
__constant__ int const_num_gaussians_sparse;


// CUDA kernel function to compute intensity
__global__ void compute_intensity_forward_kernel(
    const float* __restrict__ grid_points,
    const float* __restrict__ inv_covariances,
    const float* __restrict__ gaussian_centers,
    const float* __restrict__ intensities,
    const float* __restrict__ scalings,
    float* __restrict__ intensity_grid,
    int* work_queue,
    int* work_counter
    ) {

    __shared__ float inv_cov[256][9];
    __shared__ float centers[256][3];
    __shared__ float scales[256][3];


    int gaussian_idx;
    while ((gaussian_idx = atomicAdd(work_counter, 1)) < const_num_gaussians) {
        if (gaussian_idx >= const_num_gaussians) return;

        for (int i = 0; i < 9; ++i) {
            inv_cov[threadIdx.x][i] = inv_covariances[gaussian_idx * 9 + i];
        }

        centers[threadIdx.x][0] = gaussian_centers[gaussian_idx * 3];
        centers[threadIdx.x][1] = gaussian_centers[gaussian_idx * 3 + 1];
        centers[threadIdx.x][2] = gaussian_centers[gaussian_idx * 3 + 2];

        scales[threadIdx.x][0] = scalings[gaussian_idx * 3];
        scales[threadIdx.x][1] = scalings[gaussian_idx * 3 + 1];
        scales[threadIdx.x][2] = scalings[gaussian_idx * 3 + 2];
        __syncthreads();

        // Compute mean coordinates
        int center_idx = gaussian_idx * 3;
        float mean_z = centers[threadIdx.x][0] * (float)const_grid_size_z;
        float mean_x = centers[threadIdx.x][1] * (float)const_grid_size_x;
        float mean_y = centers[threadIdx.x][2] * (float)const_grid_size_y;

        // 
        float coeff = 2.0;

        float norm_expand_z = coeff * scales[threadIdx.x][0];
        float norm_expand_x = coeff * scales[threadIdx.x][1];
        float norm_expand_y = coeff * scales[threadIdx.x][2];

        float expand_z = norm_expand_z * const_grid_size_z;
        float expand_x = norm_expand_x * const_grid_size_x;
        float expand_y = norm_expand_y * const_grid_size_y;
        

        int z_start = max(0, (int)(mean_z - expand_z));
        int x_start = max(0, (int)(mean_x - expand_x));
        int y_start = max(0, (int)(mean_y - expand_y));

        int z_end = min(const_grid_size_z-1, (int)(mean_z + expand_z));
        int x_end = min(const_grid_size_x-1, (int)(mean_x + expand_x));
        int y_end = min(const_grid_size_y-1, (int)(mean_y + expand_y));


        int grid_xy = const_grid_size_x * const_grid_size_y;
        float intensity = intensities[gaussian_idx];

        for (int z = z_start; z <= z_end; ++z){
            for (int x = x_start; x <= x_end; ++x) {
                for (int y = y_start; y <= y_end; ++y){
                    // Compute distance from the grid point to the Gaussian center
                    int grid_idx = 3 * (z * grid_xy + x * const_grid_size_y + y);
                    float dz = grid_points[grid_idx] - gaussian_centers[center_idx];
                    float dx = grid_points[grid_idx + 1] - gaussian_centers[center_idx + 1];
                    float dy = grid_points[grid_idx + 2] - gaussian_centers[center_idx + 2];

                    // Compute the expoential term
                    float power = -0.5f * (
                        dz * (inv_cov[threadIdx.x][0] * dz + inv_cov[threadIdx.x][1] * dx + inv_cov[threadIdx.x][2] * dy) +
                        dx * (inv_cov[threadIdx.x][3] * dz + inv_cov[threadIdx.x][4] * dx + inv_cov[threadIdx.x][5] * dy) +
                        dy * (inv_cov[threadIdx.x][6] * dz + inv_cov[threadIdx.x][7] * dx + inv_cov[threadIdx.x][8] * dy)
                    );
                    
                    // Compute the density value
                    float intensity_value = intensity * __expf(power);

                    // Atomic add
                    atomicAdd(&intensity_grid[z * grid_xy + x * const_grid_size_y + y], intensity_value);

                }
            }
        }
    }
}


torch::Tensor compute_intensity(
    torch::Tensor gaussian_centers,
    torch::Tensor grid_points, //
    torch::Tensor intensities,
    torch::Tensor inv_covariances,
    torch::Tensor scalings,
    torch::Tensor intensity_grid
    ){
    
    const int num_gaussians = gaussian_centers.size(0);
    const int grid_size_z = intensity_grid.size(1);
    const int grid_size_x = intensity_grid.size(2);
    const int grid_size_y = intensity_grid.size(3);


    cudaMemcpyToSymbol(const_num_gaussians, &num_gaussians, sizeof(int));
    cudaMemcpyToSymbol(const_grid_size_z, &grid_size_z, sizeof(int));
    cudaMemcpyToSymbol(const_grid_size_x, &grid_size_x, sizeof(int));
    cudaMemcpyToSymbol(const_grid_size_y, &grid_size_y, sizeof(int));

    const int threads_per_block = 256;
    const int num_blocks = (num_gaussians + threads_per_block - 1) / threads_per_block;

    // Initialize work queue and counter
    int* d_work_queue;
    int* d_work_counter;
    cudaMalloc(&d_work_queue, num_gaussians * sizeof(int));
    cudaMalloc(&d_work_counter, sizeof(int));
    cudaMemset(d_work_counter, 0, sizeof(int));

    // Fill the work queue
    thrust::sequence(thrust::device, d_work_queue, d_work_queue + num_gaussians);

    compute_intensity_forward_kernel<<<num_blocks, threads_per_block>>>(
        grid_points.data_ptr<float>(),
        inv_covariances.data_ptr<float>(),
        gaussian_centers.data_ptr<float>(),
        intensities.data_ptr<float>(),
        scalings.data_ptr<float>(),
        intensity_grid.data_ptr<float>(),
        d_work_queue,
        d_work_counter
    );

    cudaFree(d_work_queue);
    cudaFree(d_work_counter);


    return intensity_grid;
}


__forceinline__ __device__ void compute_grad_gaussian_center(
    float dz, float dx, float dy,
    const float* __restrict__ inv_cov,
    float intensity_value,
    float grad_output_val,
    float* __restrict__ grad_gaussian_center) {

    float common_term = 0.5f * intensity_value * grad_output_val;
    atomicAdd(&grad_gaussian_center[0], common_term * (2 * inv_cov[0] * dz + (inv_cov[1] + inv_cov[3]) * dx + (inv_cov[2] + inv_cov[6]) * dy));
    atomicAdd(&grad_gaussian_center[1], common_term * ((inv_cov[3] + inv_cov[1]) * dz + 2 * inv_cov[4] * dx + (inv_cov[5] + inv_cov[7]) * dy));
    atomicAdd(&grad_gaussian_center[2], common_term * ((inv_cov[6] + inv_cov[2]) * dz + (inv_cov[7] + inv_cov[5]) * dx + 2 * inv_cov[8] * dy));
}

__forceinline__ __device__ void compute_grad_inv_covariance(
    float dz, float dx, float dy,
    float intensity_value,
    float grad_output_val,
    float* __restrict__ grad_inv_covariance) {

    float grad_common = -0.5f * intensity_value * grad_output_val;
    atomicAdd(&grad_inv_covariance[0], grad_common * dz * dz);
    atomicAdd(&grad_inv_covariance[1], grad_common * dz * dx);
    atomicAdd(&grad_inv_covariance[2], grad_common * dz * dy);
    atomicAdd(&grad_inv_covariance[3], grad_common * dx * dz);
    atomicAdd(&grad_inv_covariance[4], grad_common * dx * dx);
    atomicAdd(&grad_inv_covariance[5], grad_common * dx * dy);
    atomicAdd(&grad_inv_covariance[6], grad_common * dy * dz);
    atomicAdd(&grad_inv_covariance[7], grad_common * dy * dx);
    atomicAdd(&grad_inv_covariance[8], grad_common * dy * dy);
}

__global__ void compute_intensity_backward_kernel(
    const float* __restrict__ grad_output, 
    const float* __restrict__ grid_points,
    const float* __restrict__ inv_covariances,
    const float* __restrict__ gaussian_centers,
    const float* __restrict__ intensities,
    const float* __restrict__ scalings,

    float* __restrict__ grad_gaussian_centers,
    float* __restrict__ grad_intensities, 
    float* __restrict__ grad_inv_covariances, 
    int* work_queue,
    int* work_counter
){

    __shared__ float inv_cov[256][9]; 
    __shared__ float centers[256][3];
    __shared__ float scales[256][3];

    int gaussian_idx;
    while ((gaussian_idx = atomicAdd(work_counter, 1)) < const_num_gaussians) {
        if (gaussian_idx >= const_num_gaussians) return;

        for (int i = 0; i < 9; ++i) {
            inv_cov[threadIdx.x][i] = inv_covariances[gaussian_idx * 9 + i];
        }

        centers[threadIdx.x][0] = gaussian_centers[gaussian_idx * 3];
        centers[threadIdx.x][1] = gaussian_centers[gaussian_idx * 3 + 1];
        centers[threadIdx.x][2] = gaussian_centers[gaussian_idx * 3 + 2];

        scales[threadIdx.x][0] = scalings[gaussian_idx * 3];
        scales[threadIdx.x][1] = scalings[gaussian_idx * 3 + 1];
        scales[threadIdx.x][2] = scalings[gaussian_idx * 3 + 2];
        __syncthreads();

        // Compute mean coordinates
        int center_idx = gaussian_idx * 3;
        float mean_z = centers[threadIdx.x][0] * (float)const_grid_size_z;
        float mean_x = centers[threadIdx.x][1] * (float)const_grid_size_x;
        float mean_y = centers[threadIdx.x][2] * (float)const_grid_size_y;

        // 
        float coeff = 2.0;
        float norm_expand_z = coeff * scales[threadIdx.x][0];
        float norm_expand_x = coeff * scales[threadIdx.x][1];
        float norm_expand_y = coeff * scales[threadIdx.x][2];

        float expand_z = norm_expand_z * const_grid_size_z;
        float expand_x = norm_expand_x * const_grid_size_x;
        float expand_y = norm_expand_y * const_grid_size_y;

        // float expand_z = 5.0;
        // float expand_x = 5.0;
        // float expand_y = 5.0;

        int z_start = max(0, (int)(mean_z - expand_z));
        int x_start = max(0, (int)(mean_x - expand_x));
        int y_start = max(0, (int)(mean_y - expand_y));

        int z_end = min(const_grid_size_z-1, (int)(mean_z + expand_z));
        int x_end = min(const_grid_size_x-1, (int)(mean_x + expand_x));
        int y_end = min(const_grid_size_y-1, (int)(mean_y + expand_y));

        int grid_xy = const_grid_size_x * const_grid_size_y;
        float intensity = intensities[gaussian_idx];

        for (int z = z_start; z <= z_end; ++z){
            for (int x = x_start; x <= x_end; ++x) {
                for (int y = y_start; y <= y_end; ++y){
                    // Compute distance from the grid point to the Gaussian center
                    int grid_idx = 3 * (z * grid_xy + x * const_grid_size_y + y);
                    float dz = grid_points[grid_idx] - gaussian_centers[center_idx];
                    float dx = grid_points[grid_idx + 1] - gaussian_centers[center_idx + 1];
                    float dy = grid_points[grid_idx + 2] - gaussian_centers[center_idx + 2];

                    // Compute the exponential term
                    float power = -0.5f * (
                        dz * (inv_cov[threadIdx.x][0] * dz + inv_cov[threadIdx.x][1] * dx + inv_cov[threadIdx.x][2] * dy) +
                        dx * (inv_cov[threadIdx.x][3] * dz + inv_cov[threadIdx.x][4] * dx + inv_cov[threadIdx.x][5] * dy) +
                        dy * (inv_cov[threadIdx.x][6] * dz + inv_cov[threadIdx.x][7] * dx + inv_cov[threadIdx.x][8] * dy)
                    );

                    float grad_output_val = grad_output[z * grid_xy + x * const_grid_size_y + y];
                    
                    // Compute the density value
                    float intensity_value = intensity * __expf(power);

                    // Compute gradient w.r.t. intensity
                    float grad_intensity = __expf(power) * grad_output_val;
                    atomicAdd(&grad_intensities[gaussian_idx], grad_intensity);

                    // Gradient w.r.t. gaussian centers
                    compute_grad_gaussian_center(
                        dz, dx, dy,
                        inv_cov[threadIdx.x],
                        intensity_value,
                        grad_output_val,
                        &grad_gaussian_centers[center_idx]
                    );

                    // Gradient w.r.t. inverse covariances
                    compute_grad_inv_covariance(
                        dz, dx, dy,
                        intensity_value,
                        grad_output_val,
                        &grad_inv_covariances[gaussian_idx * 9]
                    );

                }
            }
        }
    }
}

std::vector<torch::Tensor> compute_intensity_backward(
    torch::Tensor grad_output,
    torch::Tensor gaussian_centers,
    torch::Tensor grid_points, //
    torch::Tensor intensities,
    torch::Tensor inv_covariances,
    torch::Tensor scalings, //
    torch::Tensor intensity_grid
){
    const int num_gaussians = gaussian_centers.size(0);
    const int grid_size_z = intensity_grid.size(1);
    const int grid_size_x = intensity_grid.size(2);
    const int grid_size_y = intensity_grid.size(3);


    cudaMemcpyToSymbol(const_num_gaussians, &num_gaussians, sizeof(int));
    cudaMemcpyToSymbol(const_grid_size_z, &grid_size_z, sizeof(int));
    cudaMemcpyToSymbol(const_grid_size_x, &grid_size_x, sizeof(int));
    cudaMemcpyToSymbol(const_grid_size_y, &grid_size_y, sizeof(int));

    auto grad_gaussian_centers = torch::zeros_like(gaussian_centers);
    auto grad_intensities = torch::zeros_like(intensities);
    auto grad_inv_covariances = torch::zeros_like(inv_covariances);

    const int threads_per_block = 256;
    const int num_blocks = (num_gaussians + threads_per_block - 1) / threads_per_block;

    // Initialize work queue and counter
    int* d_work_queue;
    int* d_work_counter;
    cudaMalloc(&d_work_queue, num_gaussians * sizeof(int));
    cudaMalloc(&d_work_counter, sizeof(int));
    cudaMemset(d_work_counter, 0, sizeof(int));

    // Fill the work queue
    thrust::sequence(thrust::device, d_work_queue, d_work_queue + num_gaussians);

    compute_intensity_backward_kernel<<<num_blocks, threads_per_block>>>(
        grad_output.data_ptr<float>(),
        grid_points.data_ptr<float>(),
        inv_covariances.data_ptr<float>(),
        gaussian_centers.data_ptr<float>(),
        intensities.data_ptr<float>(),
        scalings.data_ptr<float>(),
        grad_gaussian_centers.data_ptr<float>(),
        grad_intensities.data_ptr<float>(),
        grad_inv_covariances.data_ptr<float>(),
        d_work_queue,
        d_work_counter
    );

    cudaFree(d_work_queue);
    cudaFree(d_work_counter);


    return {grad_gaussian_centers, grad_intensities, grad_inv_covariances, grad_output};
}


__global__ void compute_intensity_sparse_forward_kernel(
    const float* __restrict__ gaussian_centers,    // [N, 3]
    const float* __restrict__ sparse_grid_points,  // [M, 3]
    const float* __restrict__ intensities,         // [N, 1]
    const float* __restrict__ inv_covariances,     // [N, 9]
    const float* __restrict__ scalings,            // [N, 3]
    float* __restrict__ sparse_intensity_out,      // [M, 1] output
    const int M                                    // Number of sparse points
) {
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (point_idx >= M) {
        return;
    }


    float pz = sparse_grid_points[point_idx * 3 + 0];
    float px = sparse_grid_points[point_idx * 3 + 1];
    float py = sparse_grid_points[point_idx * 3 + 2];

    float total_intensity = 0.0f;


    for (int gaussian_idx = 0; gaussian_idx < const_num_gaussians_sparse; ++gaussian_idx) {
        int center_idx = gaussian_idx * 3;
        int cov_idx = gaussian_idx * 9;

        // Calculate distance to Gaussian center
        float dz = pz - gaussian_centers[center_idx + 0];
        float dx = px - gaussian_centers[center_idx + 1];
        float dy = py - gaussian_centers[center_idx + 2];

        // Apply 2 sigma limit
        float coeff = 2.0f;
        float norm_expand_z = coeff * scalings[gaussian_idx * 3 + 0];
        float norm_expand_x = coeff * scalings[gaussian_idx * 3 + 1];
        float norm_expand_y = coeff * scalings[gaussian_idx * 3 + 2];

        float expand_z = norm_expand_z * const_grid_size_z;
        float expand_x = norm_expand_x * const_grid_size_x;
        float expand_y = norm_expand_y * const_grid_size_y;

        // Calculate grid indices for current point
        int z_idx = (int)(round(pz * (const_grid_size_z-1)));
        int x_idx = (int)(round(px * (const_grid_size_x-1)));
        int y_idx = (int)(round(py * (const_grid_size_y-1)));

        // Calculate Gaussian center grid coordinates
        float mean_z = gaussian_centers[center_idx + 0] * const_grid_size_z;
        float mean_x = gaussian_centers[center_idx + 1] * const_grid_size_x;
        float mean_y = gaussian_centers[center_idx + 2] * const_grid_size_y;

        // Calculate bounding box
        int z_start = max(0, (int)(mean_z - expand_z));
        int x_start = max(0, (int)(mean_x - expand_x));
        int y_start = max(0, (int)(mean_y - expand_y));

        int z_end = min(const_grid_size_z-1, (int)(mean_z + expand_z));
        int x_end = min(const_grid_size_x-1, (int)(mean_x + expand_x));
        int y_end = min(const_grid_size_y-1, (int)(mean_y + expand_y));

        // Check if current point is within bounding box
        if (z_idx < z_start || z_idx > z_end ||
            x_idx < x_start || x_idx > x_end ||
            y_idx < y_start || y_idx > y_end) {
            continue;  // Skip this Gaussian if outside bounding box
        }

        // Load inverse covariance
        const float* inv_cov = &inv_covariances[cov_idx];

        // Calculate exponential term
        float power = -0.5f * (
            dz * (inv_cov[0] * dz + inv_cov[1] * dx + inv_cov[2] * dy) +
            dx * (inv_cov[3] * dz + inv_cov[4] * dx + inv_cov[5] * dy) +
            dy * (inv_cov[6] * dz + inv_cov[7] * dx + inv_cov[8] * dy)
        );

        // Calculate and accumulate intensity value
        total_intensity += intensities[gaussian_idx] * __expf(power);
    }

    // Write final result
    sparse_intensity_out[point_idx] = total_intensity;
}

// CUDA kernel: Sparse backward propagation
__global__ void compute_intensity_sparse_backward_kernel(
    const float* __restrict__ grad_output,          // [M, 1]
    const float* __restrict__ gaussian_centers,     // [N, 3]
    const float* __restrict__ sparse_grid_points,   // [M, 3]
    const float* __restrict__ intensities,          // [N, 1]
    const float* __restrict__ inv_covariances,      // [N, 9]
    const float* __restrict__ scalings,             // [N, 3]
    const float* __restrict__ computed_intensities, // [M, 1] (forward pass result)
    // Output gradients (accumulated atomically)
    float* __restrict__ grad_gaussian_centers,      // [N, 3]
    float* __restrict__ grad_intensities,           // [N, 1]
    float* __restrict__ grad_inv_covariances,       // [N, 9]
    const int M                                     // Number of sparse points
) {
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (point_idx >= M) {
        return;
    }

    // Get current sparse point information
    float pz = sparse_grid_points[point_idx * 3 + 0];
    float px = sparse_grid_points[point_idx * 3 + 1];
    float py = sparse_grid_points[point_idx * 3 + 2];
    float grad_out_val = grad_output[point_idx];

    // Skip if gradient is zero
    if (grad_out_val == 0.0f) {
        return;
    }

    // Loop through all Gaussians to calculate their gradient contributions
    for (int gaussian_idx = 0; gaussian_idx < const_num_gaussians_sparse; ++gaussian_idx) {
        int center_idx = gaussian_idx * 3;
        int cov_idx = gaussian_idx * 9;

        // Calculate distance
        float dz = pz - gaussian_centers[center_idx + 0];
        float dx = px - gaussian_centers[center_idx + 1];
        float dy = py - gaussian_centers[center_idx + 2];

        // Apply 2 sigma limit
        float coeff = 2.0f;
        float norm_expand_z = coeff * scalings[gaussian_idx * 3 + 0];
        float norm_expand_x = coeff * scalings[gaussian_idx * 3 + 1];
        float norm_expand_y = coeff * scalings[gaussian_idx * 3 + 2];
        float expand_z = norm_expand_z * const_grid_size_z;
        float expand_x = norm_expand_x * const_grid_size_x;
        float expand_y = norm_expand_y * const_grid_size_y;

        // Calculate grid indices for current point
        int z_idx = (int)(round(pz * (const_grid_size_z-1)));
        int x_idx = (int)(round(px * (const_grid_size_x-1)));
        int y_idx = (int)(round(py * (const_grid_size_y-1)));

        // Calculate Gaussian center grid coordinates
        float mean_z = gaussian_centers[center_idx + 0] * const_grid_size_z;
        float mean_x = gaussian_centers[center_idx + 1] * const_grid_size_x;
        float mean_y = gaussian_centers[center_idx + 2] * const_grid_size_y;

        // Calculate bounding box
        int z_start = max(0, (int)(mean_z - expand_z));
        int x_start = max(0, (int)(mean_x - expand_x));
        int y_start = max(0, (int)(mean_y - expand_y));

        int z_end = min(const_grid_size_z-1, (int)(mean_z + expand_z));
        int x_end = min(const_grid_size_x-1, (int)(mean_x + expand_x));
        int y_end = min(const_grid_size_y-1, (int)(mean_y + expand_y));

        // Check if current point is within bounding box
        if (z_idx < z_start || z_idx > z_end ||
            x_idx < x_start || x_idx > x_end ||
            y_idx < y_start || y_idx > y_end) {
            continue;  // Skip this Gaussian if outside bounding box
        }

        // Load inverse covariance
        const float* inv_cov = &inv_covariances[cov_idx];

        // Calculate exponential term
        float power = -0.5f * (
            dz * (inv_cov[0] * dz + inv_cov[1] * dx + inv_cov[2] * dy) +
            dx * (inv_cov[3] * dz + inv_cov[4] * dx + inv_cov[5] * dy) +
            dy * (inv_cov[6] * dz + inv_cov[7] * dx + inv_cov[8] * dy)
        );

        // Calculate Gaussian intensity contribution at this point
        float exp_val = __expf(power);
        float intensity_value = intensities[gaussian_idx] * exp_val;

        // Calculate gradient w.r.t. intensity parameter
        float grad_intensity_param = exp_val * grad_out_val;
        atomicAdd(&grad_intensities[gaussian_idx], grad_intensity_param);

        // Calculate gradient w.r.t. Gaussian centers
        float grad_center_common = 0.5f * intensity_value * grad_out_val;
        atomicAdd(&grad_gaussian_centers[center_idx + 0], grad_center_common * (2 * inv_cov[0] * dz + (inv_cov[1] + inv_cov[3]) * dx + (inv_cov[2] + inv_cov[6]) * dy));
        atomicAdd(&grad_gaussian_centers[center_idx + 1], grad_center_common * ((inv_cov[3] + inv_cov[1]) * dz + 2 * inv_cov[4] * dx + (inv_cov[5] + inv_cov[7]) * dy));
        atomicAdd(&grad_gaussian_centers[center_idx + 2], grad_center_common * ((inv_cov[6] + inv_cov[2]) * dz + (inv_cov[7] + inv_cov[5]) * dx + 2 * inv_cov[8] * dy));

        // Calculate gradient w.r.t. inverse covariance matrix elements
        float grad_inv_cov_common = -0.5f * intensity_value * grad_out_val;
        atomicAdd(&grad_inv_covariances[cov_idx + 0], grad_inv_cov_common * dz * dz);
        atomicAdd(&grad_inv_covariances[cov_idx + 1], grad_inv_cov_common * dz * dx);
        atomicAdd(&grad_inv_covariances[cov_idx + 2], grad_inv_cov_common * dz * dy);
        atomicAdd(&grad_inv_covariances[cov_idx + 3], grad_inv_cov_common * dx * dz);
        atomicAdd(&grad_inv_covariances[cov_idx + 4], grad_inv_cov_common * dx * dx);
        atomicAdd(&grad_inv_covariances[cov_idx + 5], grad_inv_cov_common * dx * dy);
        atomicAdd(&grad_inv_covariances[cov_idx + 6], grad_inv_cov_common * dy * dz);
        atomicAdd(&grad_inv_covariances[cov_idx + 7], grad_inv_cov_common * dy * dx);
        atomicAdd(&grad_inv_covariances[cov_idx + 8], grad_inv_cov_common * dy * dy);
    }
}

// C++ wrapper for sparse forward
torch::Tensor compute_intensity_sparse_forward(
    torch::Tensor gaussian_centers,
    torch::Tensor sparse_grid_points, // [M, 3]
    torch::Tensor intensities,
    torch::Tensor inv_covariances,
    torch::Tensor scalings,
    const int M
){
    const int num_gaussians = gaussian_centers.size(0);
    cudaMemcpyToSymbol(const_num_gaussians_sparse, &num_gaussians, sizeof(int));

    // Create output Tensor
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(gaussian_centers.device());
    auto sparse_intensity_out = torch::zeros({M, 1}, options);

    if (M == 0) return sparse_intensity_out;

    const int threads_per_block = 256;
    const int num_blocks = (M + threads_per_block - 1) / threads_per_block;

    compute_intensity_sparse_forward_kernel<<<num_blocks, threads_per_block>>>(
        gaussian_centers.data_ptr<float>(),
        sparse_grid_points.data_ptr<float>(),
        intensities.data_ptr<float>(),
        inv_covariances.data_ptr<float>(),
        scalings.data_ptr<float>(),
        sparse_intensity_out.data_ptr<float>(),
        M
    );

    return sparse_intensity_out;
}

// C++ wrapper for sparse backward
std::vector<torch::Tensor> compute_intensity_sparse_backward(
    torch::Tensor grad_output, // [M, 1]
    torch::Tensor gaussian_centers,
    torch::Tensor sparse_grid_points, // [M, 3]
    torch::Tensor intensities,
    torch::Tensor inv_covariances,
    torch::Tensor scalings,
    torch::Tensor computed_intensities, // [M, 1] from forward pass
    const int M
){
    const int num_gaussians = gaussian_centers.size(0);
    cudaMemcpyToSymbol(const_num_gaussians_sparse, &num_gaussians, sizeof(int));

    // Create gradient output Tensors (initialized to 0)
    auto grad_gaussian_centers = torch::zeros_like(gaussian_centers);
    auto grad_intensities = torch::zeros_like(intensities);
    auto grad_inv_covariances = torch::zeros_like(inv_covariances);

    if (M == 0) {
        return {grad_gaussian_centers, grad_intensities, grad_inv_covariances};
    }

    const int threads_per_block = 256;
    const int num_blocks = (M + threads_per_block - 1) / threads_per_block;

    compute_intensity_sparse_backward_kernel<<<num_blocks, threads_per_block>>>(
        grad_output.data_ptr<float>(),
        gaussian_centers.data_ptr<float>(),
        sparse_grid_points.data_ptr<float>(),
        intensities.data_ptr<float>(),
        inv_covariances.data_ptr<float>(),
        scalings.data_ptr<float>(),
        computed_intensities.data_ptr<float>(),
        grad_gaussian_centers.data_ptr<float>(),
        grad_intensities.data_ptr<float>(),
        grad_inv_covariances.data_ptr<float>(),
        M
    );

    return {grad_gaussian_centers, grad_intensities, grad_inv_covariances};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("compute_intensity", &compute_intensity, "Compute Intensity (CUDA)");
    m.def("compute_intensity_backward", &compute_intensity_backward, "Compute intensity backward (CUDA)");

    m.def("compute_intensity_sparse_forward", &compute_intensity_sparse_forward, "Compute Sparse Intensity Forward (CUDA)");
    m.def("compute_intensity_sparse_backward", &compute_intensity_sparse_backward, "Compute Sparse Intensity Backward (CUDA)");
}
