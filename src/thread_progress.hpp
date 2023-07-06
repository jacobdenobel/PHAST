#include <iomanip>
#include <iostream>
#include <mutex>
#include <vector>
#include <cmath>

struct ThreadProgress
{
    size_t total;
    size_t n;
    std::vector<size_t> all_progress;
    std::vector<size_t> next_log;
    bool threaded;
    std::mutex io_mutex;
    size_t increment_step;
    float increment = .1;

    ThreadProgress(const size_t tot, const size_t n = 7, const bool threaded = true) 
        : total(tot), n(n), all_progress(n, 0), next_log(n), threaded(threaded), io_mutex{}
    {
        increment_step = static_cast<size_t>(std::ceil((tot / 100) *increment));
        for(size_t i = 0; i < n; i++)
            next_log[i] = increment_step;
    }

    void update(const size_t id, const int v = 1)
    {
        if (total < 10'000)
            return; 

        static constexpr size_t width = 6;
        all_progress[id] += v;
        
        if (all_progress[id] < next_log[id])
            return;

        while(all_progress[id] > next_log[id])
            next_log[id] = std::min(next_log[id] + increment_step, total);

        const float progress = (static_cast<float>(all_progress[id]) / total) * 100.;
        const int pos = id * (2 * width) * threaded;
        const int pzero = 1 * (pos == 0);
        {

            std::lock_guard<std::mutex> lk(io_mutex);
            std::cout << "\033[" << pos << "C";
            std::cout << id << ":";
            std::cout << std::fixed << std::setprecision(2) << std::setw(width) << progress << "%";
            std::cout << "\033[" << (pzero + pos + width + 3) << "D" << std::flush;
        }
    }
};
