#pragma once

#include "common.hpp"
#include "fiber_stats.hpp"

namespace phast
{
    struct Neurogram
    {

        double binsize_;
        double duration_;
        std::vector<int> fiber_ids_;
        std::vector<std::vector<int>> data_;

        Neurogram(
            const std::vector<FiberStats> &fs,
            const double binsize) : binsize_(binsize),
                                    duration_(max_duration(fs)),
                                    fiber_ids_(get_fiber_ids(fs)),
                                    data_(fiber_ids_.size(), std::vector<int>(std::ceil(duration_ / binsize_), 0))
        {
            for (int data_id = 0; data_id < fiber_ids_.size(); data_id++)
                compute_spike_rate(fs, data_id, fs[0].time_step);
        }

        void compute_spike_rate(
            const std::vector<FiberStats> &fs,
            const int data_id,
            const double ts)
        {
            for (const auto &fiber_stats : fs)
            {
                if (fiber_stats.fiber_id != fiber_ids_[data_id])
                    continue;

                for (const size_t sp : fiber_stats.spikes)
                {
                    const double spike_time = sp * ts;
                    const int bin_idx = std::floor(spike_time / binsize_);
                    data_[data_id][bin_idx] += 1;
                }
            }
        }

        static double max_duration(const std::vector<FiberStats> &fs)
        {
            double max_d = 0;
            for (const auto &f : fs)
                max_d = std::max(max_d, f.duration());
            return max_d;
        }

        static std::vector<int> get_fiber_ids(const std::vector<FiberStats> &fs)
        {
            std::vector<int> result;
            for (auto &f : fs)
                result.push_back(f.fiber_id);
            std::sort(result.begin(), result.end());
            auto ptr = std::unique(result.begin(), result.end());
            result.erase(ptr, result.end());
            return result;
        }
    };
}