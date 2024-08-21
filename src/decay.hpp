#pragma once

#include "fiber_stats.hpp"
#include "pulse_train.hpp"

namespace phast
{
    struct Decay
    {
        double time_step;

        virtual void setup(const PulseTrain &pt)
        {
            time_step = pt.time_step;
        }

        virtual double decay(const size_t t)
        {
            return std::numeric_limits<double>::signaling_NaN();
        };

        virtual double compute_spike_adaptation(const size_t t, const FiberStats &stats, const std::vector<double> &i_det) = 0;

        virtual double compute_pulse_accommodation(const size_t t, const FiberStats &stats) = 0;

        virtual std::shared_ptr<Decay> randomize(RandomGenerator &rng) = 0;

        virtual bool is_historical() const {return false;}
    };

    namespace original
    {
        struct HistoricalDecay : Decay
        {
            double adaptation_amplitude;
            double accommodation_amplitude;
            double sigma_adaptation_amplitude;
            double sigma_accommodation_amplitude;
            size_t memory_size;

            HistoricalDecay(const double adaptation_amplitude,
                            const double accommodation_amplitude,
                            const double sigma_adaptation_amplitude,
                            const double sigma_accommodation_amplitude,
                            const size_t memory_size,
                            bool allow_precomputed_accommodation = false,
                            bool cached_decay = false,
                            const std::vector<double> &cache = {})
                : adaptation_amplitude(adaptation_amplitude),
                  accommodation_amplitude(accommodation_amplitude),
                  sigma_adaptation_amplitude(sigma_adaptation_amplitude),
                  sigma_accommodation_amplitude(sigma_accommodation_amplitude),
                  memory_size(memory_size),
                  allow_precomputed_accommodation_(allow_precomputed_accommodation),
                  cached_decay_(cached_decay),
                  cache_(cache)
            {
            }

            bool is_historical() const override {return true;}

            void setup(const PulseTrain &pt) override
            {
                Decay::setup(pt);
                cache_decay(pt);
                if (can_use_precompomputed_accomodation(pt))
                    set_precomputed_accommodation(true);
            }

            bool can_use_precompomputed_accomodation(const PulseTrain &pt)
            {
                return pt.n_used_electrodes == 1 && pt.n_unique_pulses == 1 && pt.n_delta_t == 1;
            }

            void cache_decay(const PulseTrain &pt)
            {
#ifdef VERBOSE
                std::cout << "caching decay\n";
#endif
                cache_.resize(pt.t_max);
                cached_decay_ = pt.sigma_ap == 0.0;

                // for (const auto t : pt.pulse_times)
                for (size_t i = 0; i < pt.n_pulses; i++)
                {
                    auto t = pt.get_pulse(i).time;
                    cache_[t] = this->decay(t);
                    if (t > pt.steps_to_ap)
                    {
                        const auto t_ap = t - pt.steps_to_ap;
                        cache_[t_ap] = this->decay(t_ap);
                    }
                }
            }

            double get_decay(const size_t t)
            {
                if (cached_decay_)
                    return cache_[t];
                return this->decay(t);
            }

            void set_precomputed_accommodation(const bool allowed)
            {
#ifdef VERBOSE
                if (allowed)
                    std::cout << "using precomputed accomodation\n";
#endif
                allow_precomputed_accommodation_ = allowed;
            }

            double compute_spike_adaptation(const size_t t, const FiberStats &stats, const std::vector<double> &i_det) override
            {
                double adaptation = 0.0;
                for (size_t i = 0; i < stats.n_spikes; i++)
                {
                    const auto time_since_spike = t - stats.spikes[i];
                    adaptation += (adaptation_amplitude * i_det[stats.electrodes[i]]) * get_decay(time_since_spike);
                }
                return adaptation;
            }

            double compute_pulse_accommodation(const size_t t, const FiberStats &stats) override
            {
                if (allow_precomputed_accommodation_)
                    return compute_accommodation_precomputed(t, stats);
                return compute_accommodation_historical(t, stats);
            }

            double compute_accommodation_historical(const size_t t, const FiberStats &stats)
            {
                const size_t m = (memory_size != 0) *
                                 std::max(static_cast<int>(stats.n_pulses) - static_cast<int>(memory_size), 0);

                double accommodation = 0.0;

                for (size_t i = m; i < stats.n_pulses; i++)
                {
                    const auto time_since_pulse = t - stats.pulse_times[i];
                    accommodation += (accommodation_amplitude * stats.scaled_i_given[i]) * get_decay(time_since_pulse);
                }
                return accommodation;
            }

            double compute_accommodation_precomputed(const size_t t, const FiberStats &stats)
            {
                precomputed_accommodation_ += (accommodation_amplitude * stats.scaled_i_given[0]) * get_decay(t);
                return precomputed_accommodation_;
            }

        protected:
            /**
             * @brief precompute accomodation, this is ONLY possible when:
             *  - the pulse train has a constant amplitude
             *  - the pulse train has a constant rate
             *  - only a single electrode is used
             *
             * @note use with caution.
             */
            bool allow_precomputed_accommodation_;
            bool cached_decay_;
            double precomputed_accommodation_ = 0.;
            std::vector<double> cache_;
        };

        struct Exponential : HistoricalDecay
        {
            using Exponents = std::vector<std::pair<double, double>>;
            Exponents exponents;

            Exponential(const double adaptation_amplitude = 0.01,
                        const double accommodation_amplitude = 0.0003,
                        const double sigma_adaptation_amplitude = 0.0,
                        const double sigma_accommodation_amplitude = 0.0,
                        const Exponents &exponents = Exponents({{0.6875, 0.088}, {0.1981, 0.7}, {0.0571, 5.564}}),
                        const size_t memory_size = 0,
                        bool allow_precomputed_accommodation = false,
                        bool cached_decay = false,
                        const std::vector<double> &cache = {}
                        )
                : HistoricalDecay(adaptation_amplitude, accommodation_amplitude, sigma_adaptation_amplitude,
                                  sigma_accommodation_amplitude, memory_size, allow_precomputed_accommodation, cached_decay, cache),
                  exponents(exponents)
            {
            }

            double decay(const size_t t) override
            {
                double res = 0.0;
                for (const auto &exponent : exponents)
                    res += exponent.first * std::exp(-static_cast<double>(t) * time_step / exponent.second);
                return res;
            }

            std::shared_ptr<Decay> randomize(RandomGenerator &rng) override
            {
                return std::make_shared<Exponential>(
                    std::max(constants::min_amp, adaptation_amplitude + (sigma_adaptation_amplitude * rng())),
                    std::max(constants::min_amp, accommodation_amplitude + (sigma_accommodation_amplitude * rng())),
                    sigma_adaptation_amplitude, sigma_accommodation_amplitude,
                    exponents,
                    memory_size,
                    allow_precomputed_accommodation_,
                    cached_decay_, cache_
                    );
            };
        };

        inline double powerlaw(const double x, const double c, const double b)
        {
            return pow(x + c, b);
        }

        struct Powerlaw : HistoricalDecay
        {
            double offset;
            double exp;
            Powerlaw(const double adaptation_amplitude = 2e-4,
                     const double accommodation_amplitude = 8e-6,
                     const double sigma_adaptation_amplitude = 0.0,
                     const double sigma_accommodation_amplitude = 0.0,
                     const double offset = 0.06,
                     const double exp = -1.5,
                     const size_t memory_size = 0,
                     bool allow_precomputed_accommodation = false,
                     bool cached_decay = false,
                     const std::vector<double> &cache = {}                     
                     )
                : HistoricalDecay(adaptation_amplitude, accommodation_amplitude, sigma_adaptation_amplitude,
                                  sigma_accommodation_amplitude, memory_size, allow_precomputed_accommodation, cached_decay, cache),
                  offset(offset), exp(exp) {}

            double decay(const size_t t) override
            {
                return powerlaw(static_cast<double>(t) * time_step, offset, exp);
            }

            std::shared_ptr<Decay> randomize(RandomGenerator &rng) override
            {
                return std::make_shared<Powerlaw>(
                    std::max(constants::min_amp, adaptation_amplitude + (sigma_adaptation_amplitude * rng())),
                    std::max(constants::min_amp, accommodation_amplitude + (sigma_accommodation_amplitude * rng())),
                    sigma_adaptation_amplitude, sigma_accommodation_amplitude, memory_size,
                    offset, exp,
                    allow_precomputed_accommodation_,
                    cached_decay_, cache_
                    );
            };
        };

    }

    namespace approximated
    {
        /**
         * @brief Helper method to generate a sequence of n evenly spaced
         * points between start and stop.
         *
         * @param start the starting point
         * @param stop the end points
         * @param n the number of points to generate
         * @return std::vector<double>
         */
        inline std::vector<double> linspace(const double start, const double stop, const size_t n)
        {
            std::vector<double> res(n, start);
            const double step = std::abs((start - stop) / (n - 1));
            for (size_t i = 1; i < n; i++)
                res[i] = res[i - 1] + step;
            return res;
        }

        struct LeakyIntegrator
        {
            double scale;
            double rate;
            double value;
            double last_t;
            LeakyIntegrator(const double scale, const double rate)
                : scale(scale), rate(rate), value(0.), last_t(0.)
            {
            }

            double operator()(const double c, const double t)
            {
                const double dt = t - last_t;
                last_t = t;
                const double decay = -rate * value;
                const double dx = decay + c;
                value = value + dx * dt;
                return scale * value;
            }
        };

        struct LeakyIntegratorDecay : Decay
        {
            LeakyIntegrator adaptation;
            LeakyIntegrator accommodation;
            double sigma_rate;
            double sigma_amp;

            LeakyIntegratorDecay(
                const double adaptation_amplitude = 1.0,
                const double accommodation_amplitude = 1.0,
                const double adaptation_rate = 2.0,
                const double accommodation_rate = 2.0,
                const double sigma_amp = 0.0,
                const double sigma_rate = 0.0)
                : adaptation(adaptation_amplitude, adaptation_rate),
                  accommodation(accommodation_amplitude, accommodation_rate),
                  sigma_rate(sigma_rate), sigma_amp(sigma_amp)

            {
            }

            std::shared_ptr<Decay> randomize(RandomGenerator &rng) override
            {
                return std::make_shared<LeakyIntegratorDecay>(
                    std::max(constants::min_amp, adaptation.scale + (sigma_rate * rng())),
                    std::max(constants::min_amp, accommodation.scale + (sigma_rate * rng())),
                    std::max(constants::min_rate, adaptation.rate + (sigma_rate * rng())), 
                    std::max(constants::min_rate, accommodation.rate +  (sigma_rate * rng())), 
                    sigma_rate, sigma_amp);
            };

            double compute_spike_adaptation(const size_t t, const FiberStats &stats, const std::vector<double> &i_det) override
            {
                return adaptation(stats.last_idet, static_cast<double>(t) * time_step);
            }

            double compute_pulse_accommodation(const size_t t, const FiberStats &stats) override
            {
                return accommodation(stats.last_igiven, static_cast<double>(t) * time_step);
            }
        };
    }
}
