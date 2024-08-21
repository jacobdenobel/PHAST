#pragma once

#include "fiber.hpp"
#include <thread>

namespace phast
{
    std::vector<FiberStats> phast(
        std::vector<Fiber> fibers,
        const PulseTrain &pulse_train,
        const bool evaluate_in_parallel,
        const size_t n_trials = 1,
        const bool use_random = true)
    {
        GENERATOR.use_random = use_random;
        const size_t n_exper = fibers.size() * n_trials;

        std::vector<Fiber> trials(n_exper);
        std::vector<std::thread> threads;

        for (size_t fi = 0; fi < fibers.size(); fi++)
        {
            auto &fiber = fibers[fi];

            fiber.decay->setup(pulse_train);

            for (size_t t = 0; t < n_trials; t++)
            {
                const size_t ti = t * (fi + 1);

                trials[ti] = fiber.randomize();
                trials[ti].stats.trial_id = ti;
                if (SEED   && evaluate_in_parallel)
                    trials[ti]._generator = RandomGenerator(SEED + ti);

                if (!evaluate_in_parallel)
                {
                    trials[ti].process_pulse_train(pulse_train);
                    continue;
                }
                threads.push_back(std::thread(&Fiber::process_pulse_train, &trials[ti], std::ref(pulse_train)));
            }
        }

        for (auto &th : threads)
            th.join();

        std::vector<FiberStats> result;
        for (const auto &trial : trials)
            result.push_back(trial.stats);
        return result;
    }

    std::vector<FiberStats> phast(
        const std::vector<double> &i_det,
        const std::vector<double> &i_min,
        const std::vector<std::vector<double>> &pulse_train_array,
        std::shared_ptr<Decay> decay,
        const double relative_spread = 0.06,
        const size_t n_trials = 1,
        const RefractoryPeriod &refractory_period = RefractoryPeriod(),
        const bool use_random = true,
        const int fiber_id = 0,
        const double sigma_rs = 0.0,
        const bool evaluate_in_parallel = false,
        const double time_step = constants::time_step,
        const double time_to_ap = constants::time_to_ap,
        const bool store_stats = false)
    {
        const auto pulse_train = CompletePulseTrain(pulse_train_array, time_step, time_to_ap);

        auto default_fiber = Fiber(
            i_det, i_min, relative_spread,
            fiber_id,
            sigma_rs,
            refractory_period,
            decay,
            store_stats);

        return phast({default_fiber}, pulse_train, evaluate_in_parallel, n_trials, use_random);
    }
}
