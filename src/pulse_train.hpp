#pragma once

#include "common.hpp"

namespace phast
{
    struct Pulse
    {
        double amplitude;
        size_t electrode;
        size_t time;
    };

    struct PulseTrain
    {
        double time_step;
        double time_to_ap;
        double sigma_ap;

        size_t steps_to_ap;
        size_t n_electrodes;
        size_t t_max;
        size_t n_pulses;

        size_t n_used_electrodes;
        size_t n_unique_pulses;
        size_t n_delta_t;

        double duration;

        PulseTrain(const size_t t_max, const size_t n_pulses, const size_t n_electrodes,
                   const double time_step = constants::time_step,
                   const double time_to_ap = constants::time_to_ap,
                   const double sigma_ap = 0.0,
                   const size_t n_used_electrodes = 1,
                   const size_t n_unique_pulses = 1,
                   const size_t n_delta_t = 1)
            : time_step(time_step),
              time_to_ap(time_to_ap),
              sigma_ap(sigma_ap),
              steps_to_ap(static_cast<size_t>(std::floor(time_to_ap / time_step))),
              n_electrodes(n_electrodes),
              t_max(t_max),
              n_pulses(n_pulses),
              n_used_electrodes(n_used_electrodes),
              n_unique_pulses(n_unique_pulses),
              n_delta_t(n_delta_t),
              duration(t_max * time_step)
        {
        }

        virtual Pulse get_pulse(const size_t) const = 0;

        std::string repr() const
        {
            std::string result = "<PulseTrain n_pulses: " + std::to_string(n_pulses) + 
                " duration : " + std::to_string(duration) + ">";
            return result;
        }
    };

    struct CompletePulseTrain : PulseTrain
    {
        std::vector<double> pulses;
        std::vector<size_t> pulse_times;
        std::vector<size_t> electrodes;

        CompletePulseTrain(const std::vector<std::vector<double>> &pulse_train_array,
                           const double time_step = constants::time_step,
                           const double time_to_ap = constants::time_to_ap,
                           const double sigma_ap = 0.0)
            : PulseTrain(pulse_train_array[0].size(), 0., pulse_train_array.size(), time_step, time_to_ap, sigma_ap)
        {
            std::vector<int> used_electrodes(n_electrodes, 0);
            std::vector<size_t> delta_t;

            size_t previous_t = 0;

            for (size_t t = 0; t < t_max; t++)
            {
                for (size_t e = 0; e < n_electrodes; e++)
                {
                    const auto pulse = pulse_train_array[e][t];
                    if (pulse != 0)
                    {
                        pulses.push_back(std::abs(pulse));
                        pulse_times.push_back(t);
                        electrodes.push_back(e);

                        n_pulses++;
                        used_electrodes[e] = 1;

                        if (t > 1)
                            delta_t.push_back(t - previous_t);
                        previous_t = t;
                    }
                }
            }
            n_used_electrodes = std::accumulate(used_electrodes.begin(), used_electrodes.end(), 0);
            n_unique_pulses = std::unordered_set<double>(pulses.begin(), pulses.end()).size();
            n_delta_t = std::unordered_set<double>(delta_t.begin(), delta_t.end()).size();
        };

        Pulse get_pulse(const size_t i) const override
        {
            return {pulses[i], electrodes[i], pulse_times[i]};
        }
    };

    struct ConstantPulseTrain : PulseTrain
    {
        size_t pulse_interval;
        double amplitude;

        ConstantPulseTrain(
            const double duration,
            const double rate,
            const double amplitude,
            const double time_step = constants::time_step,
            const double time_to_ap = constants::time_to_ap,
            const double sigma_ap = 0.0) : 
                PulseTrain(static_cast<size_t>(std::floor(duration / time_step)),
                                               duration * rate, 1, time_step, time_to_ap, sigma_ap),
                pulse_interval(static_cast<size_t>(std::floor(1 / rate / time_step))),
                amplitude(amplitude)
        {
        }


        Pulse get_pulse(const size_t i) const override {
            return {amplitude, 0, i*pulse_interval};
        }
    };
}
