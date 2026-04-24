#pragma once

#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <limits>


class ThompsonSamplingScheduler {

    struct Stat {
        double count = 0;
        double sum_val = 0.0;
        double sum_val2 = 0.0;
        size_t num_no_update = 0;

        double mean() {  // mean estimate
            return sum_val / count;
        }
        double var() {  // unbiased variance estimate
            return (sum_val2 / count - mean()*mean())
                * (double)count / (count-1.0);
        }
        double stdev() {  // unbiased standard deviation estimate
            return sqrt(var());
        }
        double var_mean() {  // variance of mean estimate
            return (sum_val2 / count - mean()*mean()) / (count-1.0);
        }
        double stdev_mean(double noise_std=0.0) {  // stdev of mean estimate, with added noise
            return sqrt(var_mean()+noise_std*noise_std);
        }
    };
    // TODO: log normal is probably more suitable for modeling timing distribution than normal

    size_t num_options;
    double stdev_offset, adapt_beta, warmup_beta;
    size_t max_no_update;
    std::vector<Stat> stats;

    std::default_random_engine rng;

public:
    ThompsonSamplingScheduler()
        : num_options(0) {};

    // initial_noise_std: initial noise std (in ms), getting enough samples for each method
    // max_no_update: require each method to be sampled at least once every this number of steps
    // adapt_tau: time constant in steps, for adapting to timing change as training progress
    // warmup_tau: warmup time constant in steps, for decaying initial noise
    ThompsonSamplingScheduler(
        size_t num_options, double initial_noise_std=1e3,
        size_t max_no_update=100, double adapt_tau=1000, double warmup_tau=100,
        uint_fast32_t seed=42);

    void update(size_t idx, double val);

    size_t sample();

    bool empty() const { return num_options == 0; }
};
