#include <vector>
#include <omp.h>

std::vector<uint> serialSum(const std::vector<uint> &v) {
    std::vector<uint> sums(2);
    // TODO
    for (auto i = v.begin(); i != v.end(); i++){
        if ((*i) % 2 == 0)
            sums[0] += (*i);
        else
            sums[1] += (*i);
    }

    return sums;
}

std::vector<uint> parallelSum(const std::vector<uint> &v) {
    std::vector<uint> sums(2);
    
    uint odd_sum = 0;
    uint even_sum = 0;
    #pragma omp parallel for reduction(+:odd_sum) reduction(+:even_sum)
    for (size_t i = 0; i < v.size(); i++){
        if (v[i] % 2 == 0)
            even_sum += v[i];
        else
            odd_sum += v[i];
    }
    sums[0] = even_sum;
    sums[1] = odd_sum;
    return sums;
}