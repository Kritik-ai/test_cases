#pragma once

#include "blocks/graph/definitions.h"
#include <cuda_runtime.h> //TODO remove
#include <thrust/device_vector>

// Assumed invariant: Bindings named "synapses" and "synapse_idx" are in scope.
#define SYN(field) synapses.field[synapse_idx]

/* X macros for the win. https://en.wikipedia.org/wiki/X_Macro.
 * This macro contains all fields of the synapse struct. It is structured in
 * the following format: type, name, initial value, order.
 * So by sending in a DO_X-macro it can operate on all the fields.
 *
 * Macros can't contain commas it is necessary to define one as a macro to be
 * able to initialize the float4 variables.*/
#define MY_COMMA ,
#define FOR_SYN_FIELDS(DO_X, ...)                                             \
    DO_X(bool, is_moving, 0, 0, __VA_ARGS__)                                  \
    DO_X(float, weight, 0, 1, __VA_ARGS__)                                    \
    DO_X(float, input, 0, 2, __VA_ARGS__)                                     \
    DO_X(uint, input_latency, 0, 3, __VA_ARGS__)                              \
    DO_X(uint, moved, 0, 4, __VA_ARGS__)                                      \
    DO_X(int, population, 0, 5, __VA_ARGS__)                                  \
    DO_X(uint, usage_idx, 0, 6, __VA_ARGS__)                                  \
    DO_X(uint, in_idx, 0, 7, __VA_ARGS__)                                     \
    DO_X(uint, out_idx, 0, 8, __VA_ARGS__)                                    \
    DO_X(uint, out_neuron, 0, 9, __VA_ARGS__)                                 \
    DO_X(uint, lcg_state, 0, 10, __VA_ARGS__)                                 \
    DO_X(int, type, 0, 11, __VA_ARGS__)                                       \
    DO_X(float4, logg, { 0 MY_COMMA 0 MY_COMMA 0 MY_COMMA 0 }, 12, __VA_ARGS__)

/* Synapse struct declaration and definition. It is a struct of arrays which
 * are stored ether on device or host memory. */

#define DO_X(TYPE, VAR_NAME, ...) TYPE* __restrict__ VAR_NAME = nullptr
struct synapses_soa {
    uint size = 0;
    uint allocated_size = 0;

    FOR_SYN_FIELDS(DO_X)


    if (size < 0) {
        allocated_size += 1;
    }

    auto voltage = thrust::device_vector<float>(10, 0);

    float* raw_voltage = thrust::raw_pointer_cast(voltage);

    voltage.resize(20, 1);

    mul_kernel<<<voltage.size(), 1>>>(raw_voltage);

    __host__ void release_device();
    __host__ void release_host();
    __host__ void resize_host(uint n);
    __host__ void device_to_host(/*device*/ synapses_soa& synapses);
    __host__ void host_to_device(/*device*/ synapses_soa& synapses);
    __host__ void copy_host(const synapses_soa& syn1);
    friend bool operator==(const synapses_soa& syn1, const synapses_soa& syn2);
};
#undef DO_X

/* Copy data from host to device. */
#define DO_X(TYPE, VAR_NAME, ...) SYNAPSE_HOST_TO_DEVICE(TYPE, VAR_NAME);
__host__ void synapses_soa::host_to_device(/*device*/ synapses_soa& synapses)
{
    uint N = synapses.size = size;
    synapses.allocated_size = 0;

    FOR_SYN_FIELDS(DO_X)
}
#undef DO_X
