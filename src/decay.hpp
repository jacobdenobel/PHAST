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
                    std::max(0., adaptation_amplitude + (sigma_adaptation_amplitude * rng())),
                    std::max(0., accommodation_amplitude + (sigma_accommodation_amplitude * rng())),
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
                    std::max(0., adaptation_amplitude + (sigma_adaptation_amplitude * rng())),
                    std::max(0., accommodation_amplitude + (sigma_accommodation_amplitude * rng())),
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

        /**
         * @brief Container for xy points
         *
         */
        struct Point
        {
            double x, y;
        };

        /**
         * @brief Get the x coordinate of a powerlaw function given a y value
         *
         * @param y the y value for which to get the x value
         * @param c the offset of the powerlaw
         * @param b the exponent of the powerlaw
         * @return double the x coordinate
         */
        inline double pla_x(const double y, const double c, const double b)
        {

            const double x = std::exp(std::log(y) / b) - c;
            return x;
        }

        /**
         * @brief Get the x,y value for the powerlaw for a given percentage of
         * decay compared to powerlaw(0)
         *
         * @param perc
         * @param c the offset of the power law
         * @param b the exponent of the power law
         * @return Point the x y coordinates
         */
        inline Point pla_at_perc(const double perc, const double c, const double b)
        {
            const double pla0 = original::powerlaw(0., c, b);
            const double y = pla0 * perc;
            const double x = pla_x(y, c, b);
            return {x, y};
        }
        /**
         * @brief Get the exponential smoothing parameter alpha, such that the
         * exponential decay curve passes through the points x and y, calculated
         * for a given time delta dt.
         *
         * @param x the x coordinate
         * @param y the y coordinate
         * @param scale the scale of the exponential decay curve, for usage with powerlaw,
         * this should be powerlaw(0)
         * @param dt the time delta
         * @return double the value for alpha
         */
        inline double alpha_xy(const double x, const double y, const double scale, const double dt)
        {
            const double alpha = 1 / (-x / (std::log(y / scale) * dt));
            return alpha;
        }

        /**
         * @brief Get the alpha smoothing parameter from a given tau value
         *
         * @param tau the time constant for exponential smoothing
         * @param dt the time delta
         * @return double
         */
        inline double get_alpha(const double tau, const double dt)
        {
            const double alpha = 1 - std::exp(-dt / tau);
            return alpha;
        }
        /**
         * @brief Get the tau value, which is the time constant for a given value
         * of alpha
         * @param alpha the smoothing factor
         * @param dt the time delta
         * @return double the time constant tau
         */
        inline double get_tau(const double alpha, const double dt)
        {
            const double tau = -dt / std::log(1 - alpha);
            return tau;
        }

        struct WeightedExponentialSmoothing
        {
            std::vector<double> value;
            std::vector<double> weight;
            std::vector<double> tau;

            double prev_t;
            size_t n;

            double offset;
            double expon;
            double scale;
            double pla0;

            WeightedExponentialSmoothing(const double scale = 1.0, const double offset = 0.06, const double expon = -1.5, const size_t n = 10)
                : value(std::vector<double>(n, 0.0)), weight(std::vector<double>(n, 1.0 / static_cast<double>(n))), tau(std::vector<double>(n)),
                  prev_t(0), n(n), offset(offset), expon(expon), scale(scale), pla0(original::powerlaw(0.0, offset, expon))
            {
                const auto percentiles = linspace(0.01, 0.99, n);
                for (size_t i = 0; i < n; i++)
                {
                    const auto xy = pla_at_perc(1 - percentiles[i], offset, expon);
                    const auto alpha = alpha_xy(xy.x, xy.y, pla0, 1e-6);
                    tau[i] = get_tau(alpha, 1e-6);
                }
            }

            double operator()(const double s, const double t)
            {
                const auto dt = t - prev_t;
                prev_t = t;

                const double x = s * scale;
                double res = 0.0;
                for (size_t i = 0; i < n; i++)
                {
                    const auto alpha = get_alpha(tau[i], dt);
                    value[i] = value[i] + (alpha * (x - value[i]));
                    res += value[i] * weight[i];
                }
                return res;
            }
        };

        struct WeightedExponentialSmoothingDecay : Decay
        {
            WeightedExponentialSmoothing adaptation;
            WeightedExponentialSmoothing accommodation;

            double sigma;

            WeightedExponentialSmoothingDecay(
                const double adaptation_amplitude = 2e-4,
                const double accommodation_amplitude = 8e-6,
                const double sigma = 0.0,
                const double offset = 0.06,
                const double exp = -1.5,
                const size_t n = 5) : adaptation(adaptation_amplitude, offset, exp, n),
                                      accommodation(accommodation_amplitude, offset, exp, n),
                                      sigma(sigma)
            {
            }

            std::shared_ptr<Decay> randomize(RandomGenerator &rng) override
            {
                return std::make_shared<WeightedExponentialSmoothingDecay>(
                    std::max(0., adaptation.scale + (sigma * rng())),
                    std::max(0., accommodation.scale + (sigma * rng())),
                    sigma, adaptation.offset, adaptation.expon, adaptation.n);
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
                    std::max(0., adaptation.scale + (sigma_rate * rng())),
                    std::max(0., accommodation.scale + (sigma_rate * rng())),
                    std::max(0., adaptation.rate + (sigma_rate * rng())), 
                    std::max(0., accommodation.rate +  (sigma_rate * rng())), 
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
