#include <math_constants.h>

#include "BC.h"

/**
 * Calculates the next finite difference step given a
 * grid point and step lengths.
 *
 * @param curr Pointer to the grid point that should be updated.
 * @param width Number of grid points in the x dimension.
 * @param xcfl Courant number for x dimension.
 * @param ycfl Courant number for y dimension.
 * @returns Grid value of next timestep.
 */
template<int order>
__device__
float Stencil(const float* curr, int width, float xcfl, float ycfl) {
    switch(order) {
        case 2:
            return curr[0] + xcfl * (curr[-1] + curr[1] - 2.f * curr[0]) +
                   ycfl * (curr[width] + curr[-width] - 2.f * curr[0]);

        case 4:
            return curr[0] + xcfl * (-curr[2] + 16.f * curr[1] - 30.f * curr[0]
                                     + 16.f * curr[-1] - curr[-2])
                           + ycfl * (- curr[2 * width] + 16.f * curr[width]
                                     - 30.f * curr[0] + 16.f * curr[-width]
                                     - curr[-2 * width]);

        case 8:
            return curr[0] + xcfl * (-9.f * curr[4] + 128.f * curr[3]
                                     - 1008.f * curr[2] + 8064.f * curr[1]
                                     - 14350.f * curr[0] + 8064.f * curr[-1]
                                     - 1008.f * curr[-2] + 128.f * curr[-3]
                                     - 9.f * curr[-4])
                           + ycfl * (-9.f * curr[4 * width]
                                     + 128.f * curr[3 * width]
                                     - 1008.f * curr[2 * width]
                                     + 8064.f * curr[width]
                                     - 14350.f * curr[0]
                                     + 8064.f * curr[-width]
                                     - 1008.f * curr[-2 * width]
                                     + 128.f * curr[-3 * width]
                                     - 9.f * curr[-4 * width]);

        default:
            printf("ERROR: Order %d not supported", order);
            return CUDART_NAN_F;
    }
}

/**
 * Kernel to propagate finite difference grid from the current
 * time point to the next.
 *
 * This kernel should be very simple and only use global memory
 * and 1d threads and blocks.
 *
 * @param next[out] Next grid state.
 * @param curr Current grid state.
 * @param gx Number of grid points in the x dimension.
 * @param nx Number of grid points in the x dimension to which the full
 *           stencil can be applied (ie the number of points that are at least
 *           order/2 grid points away from the boundary).
 * @param ny Number of grid points in the y dimension to which th full
 *           stencil can be applied.
 * @param xcfl Courant number for x dimension.
 * @param ycfl Courant number for y dimension.
 */
template<int order>
__global__
void gpuStencilGlobal(float* next, const float* __restrict__ curr, int gx, int nx, int ny,
                float xcfl, float ycfl) {
    // TODO
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    int i = id / nx;
    if (id < nx*ny){
        id += gx * order/2 + order/2 + order * i;
        next[id] = Stencil<order>(&curr[id],gx,xcfl,ycfl);
    }
}

/**
 * Propagates the finite difference 2D heat diffusion solver
 * using the gpuStencilGlobal kernel.
 *
 * Use this function to do necessary setup and propagate params.iters()
 * number of times.
 *
 * @param curr_grid The current state of the grid.
 * @param params Parameters for the finite difference computation.
 * @returns Time required for computation.
 */
double gpuComputationGlobal(Grid& curr_grid, const simParams& params) {

    boundary_conditions BC(params);

    Grid next_grid(curr_grid);

    // TODO: Declare variables/Compute parameters.
    int gx = params.gx();
    int gy = params.gy();
    int nx = params.nx();
    int ny = params.ny();
    float xcfl = params.xcfl();
    float ycfl = params.ycfl();
    int threadPerBlock = 1024;
    int blockPerGrid = nx * ny / threadPerBlock + 1;

    event_pair timer;
    start_timer(&timer);


    for(int i = 0; i < params.iters(); ++i) {
        // update the values on the boundary only
        BC.updateBC(next_grid.dGrid_, curr_grid.dGrid_);

        // TODO: Apply stencil.
        if (params.order() == 2){
             gpuStencilGlobal<2><<<blockPerGrid,threadPerBlock>>>(next_grid.dGrid_,curr_grid.dGrid_,gx,nx,ny,xcfl,ycfl);
        }
        else if (params.order() == 4){
             gpuStencilGlobal<4><<<blockPerGrid,threadPerBlock>>>(next_grid.dGrid_,curr_grid.dGrid_,gx,nx,ny,xcfl,ycfl);
        }
        else if (params.order() == 8){
             gpuStencilGlobal<8><<<blockPerGrid,threadPerBlock>>>(next_grid.dGrid_,curr_grid.dGrid_,gx,nx,ny,xcfl,ycfl);
        }

        Grid::swap(curr_grid, next_grid);
    }

    check_launch("gpuStencilGlobal");
    return stop_timer(&timer);
}


/**
 * Kernel to propagate finite difference grid from the current
 * time point to the next.
 *
 * This kernel should be optimized to compute finite difference updates
 * in blocks of size (blockDim.y * numYPerStep) * blockDim.x. Each thread
 * should calculate at most numYPerStep updates. It should still only use
 * global memory.
 *
 * @param next[out] Next grid state.
 * @param curr Current grid state.
 * @param gx Number of grid points in the x dimension.
 * @param nx Number of grid points in the x dimension to which the full
 *           stencil can be applied (ie the number of points that are at least
 *           order/2 grid points away from the boundary).
 * @param ny Number of grid points in the y dimension to which th full
 *           stencil can be applied.
 * @param xcfl Courant number for x dimension.
 * @param ycfl Courant number for y dimension.
 */
template<int order, int numYPerStep>
__global__
void gpuStencilBlock(float* next, const float* __restrict__ curr, int gx, int nx, int ny,
                    float xcfl, float ycfl) {
    // TODO
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    
    if ((idx < nx) && (idy * numYPerStep + numYPerStep -1  < ny)){
        for (int i = 0; i < numYPerStep; i++){
            int x = idx; // x index on inner grid
            int y = idy * numYPerStep + i; // y index on inner grid
            int arr_idx =  x + gx * (y + order/2) + order/2; //index in 1d array in terms of x,y
            next[arr_idx] = Stencil<order>(&curr[arr_idx],gx,xcfl,ycfl);
        }
    }
}

/**
 * Propagates the finite difference 2D heat diffusion solver
 * using the gpuStencilBlock kernel.
 *
 * Use this function to do necessary setup and propagate params.iters()
 * number of times.
 *
 * @param curr_grid The current state of the grid.
 * @param params Parameters for the finite difference computation.
 * @returns Time required for computation.
 */
double gpuComputationBlock(Grid& curr_grid, const simParams& params) {

    boundary_conditions BC(params);

    Grid next_grid(curr_grid);

    // TODO: Declare variables/Compute parameters.
    int gx = params.gx();
    int gy = params.gy();
    int nx = params.nx();
    int ny = params.ny();
    float xcfl = params.xcfl();
    float ycfl = params.ycfl();
    int numYPerStep = 4;
    int threadX = 64;
    int threadY = 8;
    int blockX = nx / threadX  + 1;
    int blockY = ny / (threadY*numYPerStep) + 1;
    dim3 threads(threadX, threadY);
    dim3 blocks(blockX, blockY);

    event_pair timer;
    start_timer(&timer);

    for(int i = 0; i < params.iters(); ++i) {
        // update the values on the boundary only
        BC.updateBC(next_grid.dGrid_, curr_grid.dGrid_);

        // TODO: Apply stencil.
        if (params.order() == 2){
             gpuStencilBlock<2,4><<<blocks,threads>>>(next_grid.dGrid_,curr_grid.dGrid_,gx,nx,ny,xcfl,ycfl);
        }
        else if (params.order() == 4){
             gpuStencilBlock<4,4><<<blocks,threads>>>(next_grid.dGrid_,curr_grid.dGrid_,gx,nx,ny,xcfl,ycfl);
        }
        else if (params.order() == 8){
             gpuStencilBlock<8,4><<<blocks,threads>>>(next_grid.dGrid_,curr_grid.dGrid_,gx,nx,ny,xcfl,ycfl);
        }


        Grid::swap(curr_grid, next_grid);
    }

    check_launch("gpuStencilBlock");
    return stop_timer(&timer);
}


/**
 * Kernel to propagate finite difference grid from the current
 * time point to the next.
 *
 * This kernel should be optimized to compute finite difference updates
 * in blocks of size side * side using shared memory.
 *
 * @param next[out] Next grid state.
 * @param curr Current grid state.
 * @param gx Number of grid points in the x dimension.
 * @param gy Number of grid points in the y dimension.
 * @param xcfl Courant number for x dimension.
 * @param ycfl Courant number for y dimension.
 */
template<int side, int order>
__global__
void gpuStencilShared(float* next, const float* __restrict__ curr, int gx, int gy,
               float xcfl, float ycfl) {
    // TODO
    // int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // int idy = blockDim.y * blockIdx.y + threadIdx.y;
    
    // __shared__ float block[side][side];

    // if (idx < nx && idy < ny){
    //     int g_id =  idx + gx * (idy + order/2) + order/2;
    //     block[threadIdx.x][threadIdx.y] = curr[g_id];
    // }
    // __syncthreads();

}

/**
 * Propagates the finite difference 2D heat diffusion solver
 * using the gpuStencilShared kernel.
 *
 * Use this function to do necessary setup and propagate params.iters()
 * number of times.
 *
 * @param curr_grid The current state of the grid.
 * @param params Parameters for the finite difference computation.
 * @returns Time required for computation.
 */
template<int order>
double gpuComputationShared(Grid& curr_grid, const simParams& params) {

    boundary_conditions BC(params);

    Grid next_grid(curr_grid);

    // TODO: Declare variables/Compute parameters.
    int gx = params.gx();
    int gy = params.gy();
    int nx = params.nx();
    int ny = params.ny();
    float xcfl = params.xcfl();
    float ycfl = params.ycfl();

    int threadX = 32;
    int threadY = 32;
    int blockX = nx / threadX  + 1;
    int blockY = ny / threadY + 1;
    dim3 threads(threadX, threadY);
    dim3 blocks(blockX, blockY);

    event_pair timer;
    start_timer(&timer);

    for(int i = 0; i < params.iters(); ++i) {
        // update the values on the boundary only
        BC.updateBC(next_grid.dGrid_, curr_grid.dGrid_);

        // TODO: Apply stencil.

        Grid::swap(curr_grid, next_grid);
    }

    check_launch("gpuStencilShared");
    return stop_timer(&timer);
}
