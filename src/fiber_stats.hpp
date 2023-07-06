#pragma once

#include "common.hpp"

namespace phast
{

    class FiberStats
    {
        std::vector<double> _stochastic_threshold;
        std::vector<double> _refractoriness;
        std::vector<double> _accommodation;
        std::vector<double> _adaptation;

    public:
        std::vector<size_t> spikes;
        std::vector<size_t> electrodes;
        std::vector<size_t> pulse_times;
        std::vector<double> scaled_i_given;

        size_t n_spikes;
        size_t n_pulses;
        int trial_id;
        int fiber_id;

        double last_idet;
        double last_igiven;

        bool store_stats;

        FiberStats() = default;

        FiberStats(const size_t n_max, const int fiber_id, const bool store_stats = false)
            : _stochastic_threshold(n_max * store_stats),
              _refractoriness(n_max * store_stats),
              _accommodation(n_max * store_stats),
              _adaptation(n_max * store_stats),
              spikes(n_max),
              electrodes(n_max),
              pulse_times(n_max),
              scaled_i_given(n_max, 0.),
              n_spikes(0),
              n_pulses(0),
              fiber_id(fiber_id),
              last_idet(0.),
              last_igiven(0.),
              store_stats(store_stats)
        {
        }

        FiberStats(
            const std::vector<double>& t, 
            const std::vector<double>& r, 
            const std::vector<double>& ac,
            const std::vector<double>& ad
        ): _stochastic_threshold(t), _refractoriness(r), _accommodation(ac), _adaptation(ad) {} 

        bool operator==(const FiberStats &other)
        {
            // This is an oversimplification
            return n_spikes == other.n_spikes && n_pulses == other.n_pulses;
        }

        void update(const size_t t,
                    const size_t e,
                    const double i_given,
                    const double threshold,
                    const double stochastic_threshold,
                    const double refractoriness,
                    const double adaptation,
                    const double accommodation,
                    const double i_given_sp,
                    const double idet,
                    const size_t ap_time
                )
        {
            bool spiked = i_given > threshold;
            if (spiked)
            {
                spikes[n_spikes] = ap_time;
                electrodes[n_spikes] = e;
                n_spikes++;
            }

            if(store_stats) {
                _stochastic_threshold[n_pulses] = stochastic_threshold;
                _refractoriness[n_pulses] = refractoriness;
                _adaptation[n_pulses] = adaptation;
                _accommodation[n_pulses] = accommodation;
            }           
            pulse_times[n_pulses] = t;
            scaled_i_given[n_pulses] = i_given_sp;
            last_idet = spiked * idet;
            last_igiven = i_given_sp;
            n_pulses++;
        }

        void fit_to_size()
        {
            _stochastic_threshold.resize(n_pulses);
            _refractoriness.resize(n_pulses);
            _adaptation.resize(n_pulses);
            _accommodation.resize(n_pulses);

            pulse_times.resize(n_pulses);
            scaled_i_given.resize(n_pulses);

            spikes.resize(n_spikes);
            electrodes.resize(n_spikes);
        }

        std::vector<size_t> get_spikes() const
        {
            return std::vector<size_t>(spikes.begin(), spikes.begin() + n_spikes);
        }

        std::vector<size_t> get_pulse_times() const
        {
            return std::vector<size_t>(pulse_times.begin(), pulse_times.begin() + n_pulses);
        }

        std::vector<double> get_stochastic_threshold() const
        {
            return std::vector<double>(_stochastic_threshold.begin(), _stochastic_threshold.begin() + n_pulses);
        }

        std::vector<double> get_refractoriness() const
        {
            return std::vector<double>(_refractoriness.begin(), _refractoriness.begin() + n_pulses);
        }
        std::vector<double> get_accommodation() const
        {
            return std::vector<double>(_accommodation.begin(), _accommodation.begin() + n_pulses);
        }
        std::vector<double> get_adaptation() const
        {
            return std::vector<double>(_adaptation.begin(), _adaptation.begin() + n_pulses);
        }
        std::vector<double> get_scaled_i_given() const
        {
            return std::vector<double>(scaled_i_given.begin(), scaled_i_given.begin() + n_pulses);
        }
        std::string repr() const
        {
            std::string result = "<FiberStats n_pulses: " + std::to_string(n_pulses) + " n_spikes: " + std::to_string(n_spikes) + ">";
            return result;
        }
    };

}
