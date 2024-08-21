#pragma once

#include <cmath>
#include <numeric>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <algorithm>
#include <functional>
#include <optional>

#include <unordered_map>
#include <unordered_set>

#undef VERBOSE
#undef PROGRESS_BAR

#ifdef VERBOSE
#include <iostream>
#include <iomanip>
#ifdef PROGRESS_BAR
#include "thread_progress.hpp"
#endif
// inline void print_fiber(const phast::Fiber &trial)
// {
//     std::cout << "fiber_id: " << trial.fiber_id
//               << " trial_id: " << trial.stats.trial_id
//               << " n_pulses: " << trial.stats.n_pulses
//               << " n_spikes: " << trial.stats.n_spikes << std::endl;
// }
#endif

namespace phast
{

    namespace constants
    {
        inline double us = 1e-6;
        inline double time_step = 1. * us;
        inline double time_to_ap = 0. * us;
        inline double min_amp = 1e-12;
        inline double min_rate = 1e-12;
        inline double min_refr = 1e-12;
    }

    inline int SEED = 69;

    struct RandomGenerator
    {
        std::mt19937_64 _generator;
        int _seed;
        std::normal_distribution<double> _dist;
        bool use_random;

        RandomGenerator(const int seed, const bool use_random = true)
            : _generator(seed), _seed(seed), _dist(0., 1.), use_random(use_random)
        {
        }

        double operator()()
        {
            if (use_random)
                return _dist(_generator);
            return 0.0;
        }

        void set_seed(const int seed)
        {
            _seed = seed;
            _generator.seed(_seed);
            _dist = std::normal_distribution<double>(0., 1.);
        }
    };

    inline RandomGenerator GENERATOR(SEED);

    void set_seed(const int seed)
    {
        SEED = seed;
        GENERATOR.set_seed(SEED);
    }
}
