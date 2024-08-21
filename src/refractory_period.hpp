#pragma once

#include "fiber_stats.hpp"

namespace phast
{
    struct Period
    {
        double mu;
        Period(const double mu) : mu(mu) {}
        double tau(const double r) const
        {
            return mu + (0.05 * mu * r);
        }
    };

    struct RefractoryPeriod
    {
        double sigma_absolute;
        double sigma_relative;

        Period absolute;
        Period relative;

        RefractoryPeriod(const double absolute_refractory_period = 4e-4,
                         const double relative_refractory_period = 8e-4,
                         const double sigma_absolute_refractory_period = 0.0,
                         const double sigma_relative_refractory_period = 0.0)
            : sigma_absolute(sigma_absolute_refractory_period),
              sigma_relative(sigma_relative_refractory_period),
              absolute(absolute_refractory_period),
              relative(relative_refractory_period) {}

        double compute(const size_t t, const double time_step, const FiberStats &stats, RandomGenerator &rng) const
        {
            if (stats.n_spikes <= 0)
                return 1.0;

            const double r1 = rng();
            const auto time_since_last_spike = static_cast<double>(t - stats.spikes[stats.n_spikes - 1]) * time_step;

            if (time_since_last_spike < absolute.tau(r1))
                return std::numeric_limits<double>::infinity();

            const double r2 = rng();
            return 1. / (1. - exp(-(time_since_last_spike - absolute.tau(r1)) / relative.tau(r2)));
        }

        RefractoryPeriod randomize(RandomGenerator &rng) const
        {
            double r1 = rng(), r2 = rng();
            if (r1 < 0)
                r1 *= .5;

            if (r2 < 0)
                r2 *= .5;

            return RefractoryPeriod(
                std::max(constants::min_refr, absolute.mu + (sigma_absolute * r1)),  
                std::max(constants::min_refr, relative.mu + (sigma_relative * r2)),
                sigma_absolute,
                sigma_relative
            );
        }
    };
}