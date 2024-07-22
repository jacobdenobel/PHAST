#pragma once

#include "fiber.hpp"

namespace phast
{
    std::vector<FiberStats> phast(
        std::vector<Fiber> fibers,
        const PulseTrain &pulse_train,
        const bool evaluate_in_parallel,
        const int generate_trials = 1,
        const bool use_random = true)
    {
        GENERATOR.use_random = use_random;
        
        const size_t n_trials = std::max(1, generate_trials);
        const size_t n_exper  = fibers.size() * n_trials;

        std::vector<Fiber> trials;
        trials.reserve(n_exper);
                    
        int trial_id = 0; 
        for (auto &fiber : fibers)
        {
            fiber.decay->setup(pulse_train);
             
            for (size_t t = 0; t < n_trials; t++)
            {
                auto trial = fiber.randomize();
                trial.stats.trial_id = trial_id++;
                if (SEED != 0 && evaluate_in_parallel)
                    trial._generator = RandomGenerator(SEED + trial_id);
                trials.push_back(trial);
            }
        }
        
#ifdef PROGRESS_BAR
        ThreadProgress p(pulse_train.n_pulses, n_exper, evaluate_in_parallel);
#endif
        auto run_single_trail = [&pulse_train, &n_exper, &evaluate_in_parallel
#ifdef PROGRESS_BAR
        , &p
#endif
        ](Fiber &fiber)
        {
            for (size_t i = 0; i < pulse_train.n_pulses; i++)
            {
#ifdef PROGRESS_BAR
                p.update(fiber.stats.trial_id); //TODO don't always call this
#endif
                fiber.process_pulse(pulse_train.get_pulse(i), pulse_train);
            }
            fiber.stats.fit_to_size();
            return fiber.stats;
        };

        std::vector<FiberStats> result(trials.size());
#ifdef PROGRESS_BAR
        std::cout << "Progress:" << std::endl;
#endif
        if (evaluate_in_parallel)
        {
            std::transform(
#ifdef HASTBB
                std::execution::par_unseq,
#endif
                trials.begin(), trials.end(), result.begin(), run_single_trail);
            std::sort(result.begin(), result.end(),
                      [](const FiberStats &a, const FiberStats &b)
                      {
                          if (a.fiber_id < b.fiber_id)
                              return true;
                          if (b.fiber_id < a.fiber_id)
                              return false;
                          return a.trial_id < b.trial_id;
                      });
        }
        else
        {
            std::transform(trials.begin(), trials.end(), result.begin(), run_single_trail);
        }
#ifdef VERBOSE
        std::cout << std::endl;
#endif
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
        const bool store_stats = false        
    )
    {
        const auto pulse_train = CompletePulseTrain(pulse_train_array, time_step, time_to_ap);

        auto default_fiber = Fiber(
            i_det, i_min, relative_spread,
            fiber_id,
            pulse_train.n_pulses,
            sigma_rs,
            refractory_period,
            decay,
            store_stats);

        return phast({default_fiber}, pulse_train, evaluate_in_parallel, n_trials, use_random);
    }
}
