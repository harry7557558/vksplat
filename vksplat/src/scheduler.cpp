#include "scheduler.h"



ThompsonSamplingScheduler::ThompsonSamplingScheduler(
    size_t num_options, double initial_noise_std,
    size_t max_no_update, double adapt_tau, double warmup_tau,
    uint_fast32_t seed
) : num_options(num_options), stdev_offset(initial_noise_std), stats(num_options) {
    this->adapt_beta = exp(-1.0/adapt_tau);
    this->warmup_beta = exp(-1.0/warmup_tau);
    this->max_no_update = std::max(max_no_update-num_options, 2*num_options);
    rng.seed(seed);
}


void ThompsonSamplingScheduler::update(size_t idx, double val) {

    // decay running stats
    static constexpr double kMinCount = 2.0;
    for (Stat& stat : stats) {
        if (stat.count <= kMinCount)
            continue;
        double factor = fmax(adapt_beta, kMinCount/stat.count);
        stat.count *= factor;
        stat.sum_val *= factor;
        stat.sum_val2 *= factor;
        stat.num_no_update += 1;
    }
    stdev_offset *= warmup_beta;

    // update at idx
    Stat& stat = stats[idx];
    stat.count += 1.0;
    stat.sum_val += val;
    stat.sum_val2 += val*val;
    stat.num_no_update = 0;

    // printf("update %d with %lf - mean=%lf std=%lf stdm=%lf\n",
    //     (int)idx, val, stat.mean(), stat.stdev(), stat.stdev_mean(stdev_offset));
}


size_t ThompsonSamplingScheduler::sample() {

    // make sure all option stats are initialized first
    std::vector<size_t> ustats;
    for (size_t i = 0; i < stats.size(); i++) {
        if (stats[i].count < 2.0) ustats.push_back(i);
        if (stats[i].count < 1.0) ustats.push_back(i);
    }
    if (!ustats.empty()) {
        size_t idx = std::uniform_int_distribution<size_t>(0, ustats.size()-1)(rng);
        return ustats[idx];
    }

    // if sampling is needed, do that one first
    for (size_t i = 0; i < stats.size(); i++) {
        if (stats[i].num_no_update >= max_no_update)
            return i;
    }

    // pick random (Thompson sampling with normal distribution)
    size_t best_idx = (size_t)-1;
    double best_val = std::numeric_limits<double>::max();
    for (size_t i = 0; i < stats.size(); i++) {
        double mean = stats[i].mean();
        double stdev = stats[i].stdev_mean(stdev_offset);
        double val = std::normal_distribution<double>(mean, stdev)(rng);
        if (val < best_val)
            best_idx = i, best_val = val;
    }
    // printf("Pick %d\n", (int)best_idx);
    return best_idx;

}
