#pragma once

#include "common.hpp"

namespace phast
{

    class FiberStats
    {
    public:
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

        FiberStats(const int fiber_id, const bool store_stats = false)
            : _stochastic_threshold(),
              _refractoriness(),
              _accommodation(),
              _adaptation(),
              spikes(),
              electrodes(),
              pulse_times(),
              scaled_i_given(1, 0.0),
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
                    const size_t ap_time, 
                    const bool historical_decay
                )
        {
            bool spiked = i_given > threshold;
            if (spiked)
            {
                n_spikes++;
                spikes.push_back(ap_time);
                if (historical_decay || store_stats)
                    electrodes.push_back(e);
            }

            if (store_stats) {
                _stochastic_threshold.push_back(stochastic_threshold);
                _refractoriness.push_back(refractoriness);
                _adaptation.push_back(adaptation);
                _accommodation.push_back(accommodation);
            }   

            if (historical_decay || store_stats)
            {
                if (n_pulses == 0)
                    scaled_i_given[0] = i_given_sp;
                else
                    scaled_i_given.push_back(i_given_sp);
                pulse_times.push_back(t);
            }

            last_idet = spiked * idet;
            last_igiven = i_given_sp;
            n_pulses++;
        }

        std::string repr() const
        {
            std::string result = "<FiberStats n_pulses: " + std::to_string(n_pulses) + " n_spikes: " + std::to_string(n_spikes) + ">";
            return result;
        }
    };

}
