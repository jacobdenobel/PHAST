#pragma once

#include "decay.hpp"
#include "fiber_stats.hpp"
#include "pulse_train.hpp"
#include "refractory_period.hpp"

namespace phast
{
    struct Fiber
    {
        std::vector<double> i_det;
        std::vector<double> spatial_constant;
        std::vector<double> sigma;
        int fiber_id;

        double stochastic_threshold;
        double threshold;
        double refractoriness;
        double accommodation;
        double adaptation;
        double sigma_rs;

        FiberStats stats;
        RefractoryPeriod refractory_period;
        std::shared_ptr<Decay> decay;

        std::optional<RandomGenerator> _generator = std::nullopt;

        Fiber() = default;

        Fiber(const std::vector<double> &i_det,
              const std::vector<double> &spatial_constant,
              const std::vector<double> &sigma,
              const int fiber_id,
              const double sigma_rs,
              const RefractoryPeriod &refractory_period,
              const std::shared_ptr<Decay> decay,
              const bool store_stats = false)

            : i_det(i_det), spatial_constant(spatial_constant), sigma(sigma), fiber_id(fiber_id),
              stochastic_threshold(0.0), threshold(0.0), refractoriness(0.0), accommodation(0.0), adaptation(0.0),
              sigma_rs(sigma_rs),
              stats(fiber_id, store_stats),
              refractory_period(refractory_period),
              decay(decay)
        {
        }

        template <typename... Args>
        Fiber(
            const std::vector<double> &i_det,
            const std::vector<double> &i_min,
            const double relative_spread,
            Args &&...args) : Fiber(i_det, std::vector<double>(), std::vector<double>(), std::forward<Args>(args)...)
        {
            for (size_t i = 0; i < i_det.size(); i++)
            {
                spatial_constant.push_back(i_min[i] / i_det[i]);
                sigma.push_back(i_det[i] * relative_spread);
            }
        }

        void process_pulse(const Pulse& pulse, const PulseTrain &pulse_train)
        {
            adaptation = decay->compute_spike_adaptation(pulse.time, stats, i_det);
            accommodation = decay->compute_pulse_accommodation(pulse.time, stats);
            refractoriness = refractory_period.compute(pulse.time, pulse_train.time_step, stats, generator());

            stochastic_threshold = i_det[pulse.electrode] + (sigma[pulse.electrode] * generator()());
            threshold = stochastic_threshold * refractoriness + adaptation + accommodation;

            auto ap_time = pulse.time + pulse_train.steps_to_ap;

            if (pulse_train.sigma_ap != 0.0)
            {
                const auto z = generator()();
                ap_time = pulse.time + static_cast<size_t>(std::floor(
                                  std::max(0., (pulse_train.time_to_ap + (pulse_train.sigma_ap * z))) / pulse_train.time_step));
            }

            stats.update(pulse.time, pulse.electrode, pulse.amplitude,
                         threshold, stochastic_threshold,
                         refractoriness, adaptation, accommodation,
                         pulse.amplitude * spatial_constant[pulse.electrode],
                         i_det[pulse.electrode],
                         ap_time,
                         decay->is_historical()
                         );
        }

        void process_pulse_train(const PulseTrain& pulse_train)
        {
            for (size_t i = 0; i < pulse_train.n_pulses; i++)
                process_pulse(pulse_train.get_pulse(i), pulse_train);
            stats.shrink_to_fit();
        }

        Fiber randomize(const int trial_id)
        {
            auto new_sigma = std::vector<double>(sigma.size());
            const double new_rs = std::max(0., (sigma[0] / i_det[0]) + (sigma_rs * generator()()));

            for (size_t i = 0; i < new_sigma.size(); i++)
                new_sigma[i] = i_det[i] * new_rs;

            // This is super ugly but whatev
            auto new_decay = decay->randomize(generator());
            new_decay->time_step = decay->time_step;

            auto fiber =  Fiber(
                i_det, spatial_constant, new_sigma, fiber_id,
                sigma_rs,
                refractory_period.randomize(generator()),
                new_decay,
                stats.store_stats
            );
            fiber.stats.time_step = decay->time_step;
            return fiber;
        }

        RandomGenerator &generator()
        {
            if (_generator)
                return _generator.value();
            return GENERATOR;
        }

        std::string repr() const
        {
            std::string result = "<Fiber threshold: " + std::to_string(threshold) + ">";
            return result;
        }
    };
}
