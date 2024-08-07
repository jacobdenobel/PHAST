#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "phast.hpp"

namespace py = pybind11;

using namespace phast;

void define_common(py::module &m)
{
#ifdef HASTBB
    m.attr("CAN_THREAD") = true;
#else
    m.attr("CAN_THREAD") = false;
#endif
    py::class_<Pulse>(m, "Pulse")
        .def(py::init<double, size_t, size_t>())
        .def_readonly("amplitude", &Pulse::amplitude)
        .def_readonly("time", &Pulse::time)
        .def_readonly("electrode", &Pulse::electrode)
        .def("__repr__", [](const Pulse& p) {
            return "<Pulse " 
            + std::to_string(p.amplitude) + "A (t: " 
            + std::to_string(p.time) + " e: "      
            + std::to_string(p.electrode) + ")>";
        });
    
    py::class_<PulseTrain, std::shared_ptr<PulseTrain>>(m, "AbstractPulseTrain")
        .def_readonly("t_max", &PulseTrain::t_max)
        .def_readonly("n_electrodes", &PulseTrain::n_electrodes)
        .def_readonly("n_pulses", &PulseTrain::n_pulses)
        .def_readonly("time_step", &PulseTrain::time_step)
        .def_readonly("time_to_ap", &PulseTrain::time_to_ap)
        .def_readonly("steps_to_ap", &PulseTrain::steps_to_ap)
        .def_readonly("sigma_ap", &PulseTrain::sigma_ap)
        .def_readwrite("n_used_electrodes", &PulseTrain::n_used_electrodes)
        .def_readonly("n_unique_pulses", &PulseTrain::n_unique_pulses)
        .def_readonly("n_delta_t", &PulseTrain::n_delta_t)
        .def("__repr__", &PulseTrain::repr);

    py::class_<CompletePulseTrain, PulseTrain, std::shared_ptr<CompletePulseTrain>>(m, "PulseTrain")
        .def(
            py::init<
                std::vector<std::vector<double>>,
                double,
                double,
                double>(),
            py::arg("pulse_train"),
            py::arg("time_step") = constants::time_step,
            py::arg("time_to_ap") = constants::time_to_ap,
            py::arg("sigma_ap") = 0.0)
        .def_property_readonly("pulses", [](const CompletePulseTrain &p)
                               { return py::array(p.pulses.size(), p.pulses.data()); })
        .def_property_readonly("pulse_times", [](const CompletePulseTrain &p)
                               { return py::array(p.pulse_times.size(), p.pulse_times.data()); })
        .def_property_readonly("electrodes", [](const CompletePulseTrain &p)
                               { return py::array(p.electrodes.size(), p.electrodes.data()); })
        .def("get_pulse", &CompletePulseTrain::get_pulse);

    py::class_<ConstantPulseTrain, PulseTrain, std::shared_ptr<ConstantPulseTrain>>(m, "ConstantPulseTrain")
        .def(
            py::init<double, double, double, double, double, double>(),
            py::arg("duration"),
            py::arg("rate"),
            py::arg("amplitude"),            
            py::arg("time_step") = constants::time_step,
            py::arg("time_to_ap") = constants::time_to_ap,
            py::arg("sigma_ap") = 0.0
        )
        .def_readonly("pulse_interval", &ConstantPulseTrain::pulse_interval)
        .def_readonly("amplitude", &ConstantPulseTrain::amplitude)
        .def("get_pulse", &ConstantPulseTrain::get_pulse);

    py::class_<Period>(m, "Period")
        .def(py::init<double>())
        .def_readwrite("mu", &Period::mu)
        .def("tau", &Period::tau)
        .def("__repr__", [](const Period &p)
             { return "<Period " + std::to_string(p.mu) + ">"; });

    py::class_<RefractoryPeriod>(m, "RefractoryPeriod")
        .def(
            py::init<double, double, double, double>(),
            py::arg("absolute_refractory_period") = 4e-4,
            py::arg("relative_refractory_period") = 8e-4,
            py::arg("sigma_absolute_refractory_period") = 0.0,
            py::arg("sigma_relative_refractory_period") = 0.0)
        .def("compute", &RefractoryPeriod::compute)
        .def("randomize", &RefractoryPeriod::randomize)
        .def_readonly("absolute", &RefractoryPeriod::absolute)
        .def_readonly("relative", &RefractoryPeriod::relative)
        .def_readonly("sigma_absolute", &RefractoryPeriod::sigma_absolute)
        .def_readonly("sigma_relative", &RefractoryPeriod::sigma_relative)
        .def("__repr__", [](const RefractoryPeriod &r)
             { return "<RefractoryPeriod abs: " + std::to_string(r.absolute.mu) + ", rel:" + std::to_string(r.relative.mu) + ">"; });

    py::class_<FiberStats>(m, "FiberStats")
        .def(py::init<size_t, int>(), py::arg("n_max") = 100, py::arg("fiber_id") = 1)
        .def_property_readonly("spikes", [](const FiberStats &p)
                               {
                                   const auto v = p.get_spikes();
                                   return py::array(v.size(), v.data()); })
        .def_property_readonly("pulse_times", [](const FiberStats &p)
                               {
                                   const auto v = p.get_pulse_times();
                                   return py::array(v.size(), v.data()); })
        .def_property_readonly("stochastic_threshold", [](const FiberStats &p)
                               {
                                   const auto v = p.get_stochastic_threshold();
                                   return py::array(v.size(), v.data()); })
        .def_property_readonly("refractoriness", [](const FiberStats &p)
                               {
                                   const auto v = p.get_refractoriness();
                                   return py::array(v.size(), v.data()); })
        .def_property_readonly("accommodation", [](const FiberStats &p)
                               {
                                   const auto v = p.get_accommodation();
                                   return py::array(v.size(), v.data()); })
        .def_property_readonly("adaptation", [](const FiberStats &p)
                               {
                                   const auto v = p.get_adaptation();
                                   return py::array(v.size(), v.data()); })
        .def_property_readonly("scaled_i_given", [](const FiberStats &p)
                               {
                                   const auto v = p.get_scaled_i_given();
                                   return py::array(v.size(), v.data()); })
        .def_readonly("n_spikes", &FiberStats::n_spikes)
        .def_readonly("n_pulses", &FiberStats::n_pulses)
        .def_readonly("trial_id", &FiberStats::trial_id)
        .def_readonly("fiber_id", &FiberStats::fiber_id)
        .def("__eq__", &FiberStats::operator==)
        .def("__repr__", &FiberStats::repr)
        .def(py::pickle(
            [](FiberStats &fs)
            {
                return py::make_tuple(
                    fs.get_stochastic_threshold(),
                    fs.get_refractoriness(),
                    fs.get_accommodation(),
                    fs.get_adaptation(),
                    fs.spikes,
                    fs.electrodes,
                    fs.pulse_times,
                    fs.scaled_i_given,
                    fs.n_spikes,
                    fs.n_pulses,
                    fs.trial_id,
                    fs.fiber_id,
                    fs.last_idet,
                    fs.last_igiven,
                    fs.store_stats);
            },
            [](py::tuple t)
            {
                auto fs = FiberStats(
                    t[0].cast<std::vector<double>>(),
                    t[1].cast<std::vector<double>>(),
                    t[2].cast<std::vector<double>>(),
                    t[3].cast<std::vector<double>>());
                fs.spikes = t[4].cast<std::vector<size_t>>();
                fs.electrodes = t[5].cast<std::vector<size_t>>();
                fs.pulse_times = t[6].cast<std::vector<size_t>>();
                fs.scaled_i_given = t[7].cast<std::vector<double>>();
                fs.n_spikes = t[8].cast<size_t>();
                fs.n_pulses = t[9].cast<size_t>();
                fs.trial_id = t[10].cast<int>();
                fs.fiber_id = t[11].cast<int>();
                fs.last_idet = t[12].cast<double>();
                fs.last_igiven = t[13].cast<double>();
                fs.store_stats = t[14].cast<bool>();
                return fs;
            }));

    m.def("set_seed", &set_seed, py::arg("seed"));

    py::class_<RandomGenerator>(m, "RandomGenerator")
        .def(py::init<int, bool>(), py::arg("seed"), py::arg("use_random") = true)
        .def("__call__", &RandomGenerator::operator())
        .def("__repr__", [](const RandomGenerator &r)
             { return "<RandomGenerator " + std::to_string((size_t)&r) + ">"; });
    m.attr("GENERATOR") = &GENERATOR;
}

struct PyDecay : Decay
{
    using Decay::Decay;

    double decay(const size_t t) override
    {
        PYBIND11_OVERRIDE(double, Decay, decay, t);
    }
    double compute_spike_adaptation(const size_t t, const FiberStats &stats, const std::vector<double> &i_det) override
    {
        PYBIND11_OVERRIDE_PURE(double, Decay, compute_spike_adaptation, t, stats, i_det);
    }
    double compute_pulse_accommodation(const size_t t, const FiberStats &stats) override
    {
        PYBIND11_OVERRIDE_PURE(double, Decay, compute_pulse_accommodation, t, stats);
    }
    std::shared_ptr<Decay> randomize(RandomGenerator &rng) override
    {
        PYBIND11_OVERRIDE_PURE(std::shared_ptr<Decay>, Decay, randomize, rng);
    }
};

void define_decay(py::module &m)
{
    py::class_<Decay, PyDecay, std::shared_ptr<Decay>>(m, "Decay")
        .def("decay", &Decay::decay)
        .def("compute_adaptation", &Decay::compute_spike_adaptation)
        .def("compute_accommodation", &Decay::compute_pulse_accommodation)
        .def("randomize", &Decay::randomize)
        .def_readonly("time_step", &Decay::time_step)
        .def("__repr__", [](const Decay &d)
             { return "<Decay " + std::to_string(d.time_step) + ">"; });

    py::class_<original::HistoricalDecay, Decay, std::shared_ptr<original::HistoricalDecay>>(m, "HistoricalDecay")
        .def_readonly("adaptation_amplitude", &original::HistoricalDecay::adaptation_amplitude)
        .def_readonly("accommodation_amplitude", &original::HistoricalDecay::accommodation_amplitude)
        .def_readonly("sigma_adaptation_amplitude", &original::HistoricalDecay::sigma_adaptation_amplitude)
        .def_readonly("sigma_accommodation_amplitude", &original::HistoricalDecay::sigma_accommodation_amplitude)
        .def_readonly("memory_size", &original::HistoricalDecay::memory_size)
        .def("__repr__", [](const original::HistoricalDecay &h)
             { return "<HistoricalDecay acco: " + std::to_string(h.accommodation_amplitude) + ", adap: " + std::to_string(h.adaptation_amplitude) + ">"; });

    py::class_<original::Exponential, original::HistoricalDecay, std::shared_ptr<original::Exponential>>(m, "Exponential")
        .def(
            py::init<double, double, double, double, original::Exponential::Exponents, size_t, bool, bool, std::vector<double>>(), 
            py::arg("adaptation_amplitude") = 0.01,
            py::arg("accommodation_amplitude") = 0.0003,
            py::arg("sigma_adaptation_amplitude") = 0.0,
            py::arg("sigma_accommodation_amplitude") = 0.0,
            py::arg("exponents") = original::Exponential::Exponents({{0.6875, 0.088}, {0.1981, 0.7}, {0.0571, 5.564}}),
            py::arg("memory_size") = 0,
            py::arg("allow_precomputed_accommodation") = false,
            py::arg("cached_decay") = false,
            py::arg("cache") = std::vector<double>()
        )
        .def_readonly("exponents", &original::Exponential::exponents)
        .def("__repr__", [](const original::Exponential &e)
             { return "<Exponential #" + std::to_string(e.exponents.size()) + ">"; });

    py::class_<original::Powerlaw, original::HistoricalDecay, std::shared_ptr<original::Powerlaw>>(m, "Powerlaw")
        .def(py::init<double, double, double, double,  double, double, size_t, bool, bool, std::vector<double>>(),
             py::arg("adaptation_amplitude") = 2e-4,
             py::arg("accommodation_amplitude") = 8e-6,
             py::arg("sigma_adaptation_amplitude") = 0.0,
             py::arg("sigma_accommodation_amplitude") = 0.0,
             py::arg("offset") = 0.06,
             py::arg("exp") = -1.5,
             py::arg("memory_size") = 0,
             py::arg("allow_precomputed_accommodation") = false,
             py::arg("cached_decay") = false,
             py::arg("cache") = std::vector<double>()
             )
        .def_readonly("offset", &original::Powerlaw::offset)
        .def_readonly("exp", &original::Powerlaw::exp)
        .def("__repr__", [](const original::Powerlaw &e)
             { return "<Powerlaw (" + std::to_string(e.offset) + "+ x)^" + std::to_string(e.exp) + ">"; });
}

void define_fiber(py::module &m)
{
    py::class_<Fiber>(m, "Fiber")
        .def(
            py::init<
                std::vector<double>,    // i_det
                std::vector<double>,    // spatial_constant
                std::vector<double>,    // sigma
                int,                    // fiber_id
                size_t,                 // n_max
                double,                 // sigma_rs
                RefractoryPeriod,       // refractory_period
                std::shared_ptr<Decay>, // decay
                bool                    // store_stats
                >(),
            py::arg("i_det"),
            py::arg("spatial_constant"),
            py::arg("sigma"),
            py::arg("fiber_id"),
            py::arg("n_max"),
            py::arg("sigma_rs") = 0.0,
            py::arg("refractory_period") = RefractoryPeriod(),
            py::arg("decay") = std::make_shared<original::Powerlaw>(),
            py::arg("store_stats") = false)
        .def_readwrite("i_det", &Fiber::i_det)
        .def_readwrite("spatial_constant", &Fiber::spatial_constant)
        .def_readwrite("sigma", &Fiber::sigma)
        .def_readonly("threshold", &Fiber::threshold)
        .def_readonly("stats", &Fiber::stats)
        .def_readonly("decay", &Fiber::decay)
        .def_readonly("refractory_period", &Fiber::refractory_period)
        .def("process_pulse", &Fiber::process_pulse)
        .def("randomize", &Fiber::randomize)
        .def("__repr__", &Fiber::repr);
}

void define_phast(py::module &m)
{
    m.def("phast",
          py::overload_cast<
              const std::vector<double> &,              // i_det,
              const std::vector<double> &,              // i_min,
              const std::vector<std::vector<double>> &, // pulse_train_array,
              std::shared_ptr<Decay>,                   // decay,
              double,                                   // relative_spread = 0.06,
              size_t,                                   // n_trials = 1,
              const RefractoryPeriod &,                 // refractory_period = RefractoryPeriod(),
              bool,                                     // use_random = true,
              int,                                      // fiber_id = 0,
              double,                                   // sigma_rs = 0.0,
              bool,                                     // evaluate_in_parallel = false,
              double,                                   // time_step = constants::time_step
              double,
              bool>(&phast::phast),
          py::arg("i_det"),
          py::arg("i_min"),
          py::arg("pulse_train"),
          py::arg("decay"),
          py::arg("relative_spread") = 0.06,
          py::arg("n_trials") = 1,
          py::arg("refractory_period") = RefractoryPeriod(),
          py::arg("use_random") = true,
          py::arg("fiber_id") = 0,
          py::arg("sigma_rs") = 0.0,
          py::arg("evaluate_in_parallel") = false,
          py::arg("time_step") = constants::time_step,
          py::arg("time_to_ap") = constants::time_to_ap,
          py::arg("store_stats") = false);

    m.def("phast", py::overload_cast<std::vector<Fiber>, const PulseTrain &, const bool, const int, bool>(&phast::phast),
          py::arg("fibers"),
          py::arg("pulse_train"),
          py::arg("evaluate_in_parallel") = false,
          py::arg("generate_trials") = 0,
          py::arg("use_random") = true);
}

void define_approximated(py::module &m)
{
    using namespace approximated;

    py::class_<Point>(m, "Point")
        .def(py::init<double, double>())
        .def_readwrite("x", &Point::x)
        .def_readwrite("y", &Point::y);

    py::class_<WeightedExponentialSmoothing>(m, "WeightedExponentialSmoothing")
        .def(py::init<double, double, double, size_t>(),
             py::arg("scale") = 1.0,
             py::arg("offset") = 0.06,
             py::arg("expon") = -1.5,
             py::arg("n") = 5)

        .def("__call__", &WeightedExponentialSmoothing::operator(), py::arg("sample"), py::arg("time"))
        .def_readwrite("value", &WeightedExponentialSmoothing::value)
        .def_readwrite("weight", &WeightedExponentialSmoothing::weight)
        .def_readwrite("tau", &WeightedExponentialSmoothing::tau)
        .def_readwrite("prev_t", &WeightedExponentialSmoothing::prev_t)
        .def_readwrite("n", &WeightedExponentialSmoothing::n)
        .def_readwrite("offset", &WeightedExponentialSmoothing::offset)
        .def_readwrite("expon", &WeightedExponentialSmoothing::expon)
        .def_readwrite("scale", &WeightedExponentialSmoothing::scale)
        .def_readwrite("pla0", &WeightedExponentialSmoothing::pla0)
        .def("__repr__", [](const WeightedExponentialSmoothing &em)
             { return "<WeightedExponentialSmoothing>"; });

    py::class_<WeightedExponentialSmoothingDecay, Decay, std::shared_ptr<WeightedExponentialSmoothingDecay>>(m, "WeightedExponentialSmoothingDecay")
        .def(py::init<double, double, double, double, double, size_t>(),
             py::arg("adaptation_amplitude") = 2e-4,
             py::arg("accommodation_amplitude") = 8e-6,
             py::arg("sigma") = 0.0,
             py::arg("offset") = 0.06,
             py::arg("exp") = -1.5,
             py::arg("n") = 5)
        .def_readonly("adaptation", &WeightedExponentialSmoothingDecay::adaptation)
        .def_readonly("accommodation", &WeightedExponentialSmoothingDecay::accommodation)
        .def_readonly("sigma", &WeightedExponentialSmoothingDecay::sigma)
        .def("__repr__", [](const WeightedExponentialSmoothingDecay &em)
             { return "<WeightedExponentialSmoothingDecay>"; });

    m.def("linspace", &linspace);
    m.def("pla_x", &pla_x);
    m.def("pla_at_perc", &pla_at_perc);
    m.def("alpha_xy", &alpha_xy);
    m.def("get_alpha", &get_alpha);
    m.def("get_tau", &get_tau);

    py::class_<LeakyIntegrator>(m, "LeakyIntegrator")
        .def(py::init<double, double>(), py::arg("scale") = 1.0, py::arg("rate") = 2.0)
        .def("__call__", &LeakyIntegrator::operator(), py::arg("c"), py::arg("t"))
        .def_readwrite("value", &LeakyIntegrator::value)
        .def_readwrite("scale", &LeakyIntegrator::scale)
        .def_readwrite("rate", &LeakyIntegrator::rate)
        .def_readwrite("last_t", &LeakyIntegrator::last_t);

    py::class_<LeakyIntegratorDecay, Decay, std::shared_ptr<LeakyIntegratorDecay>>(m, "LeakyIntegratorDecay")
        .def(
            py::init<double, double, double, double, double, double>(),
            py::arg("adaptation_amplitude") = 1.0,
            py::arg("accommodation_amplitude") = 1.0,
            py::arg("adaptation_rate") = 2.0,
            py::arg("accommodation_rate") = 2.0,
            py::arg("sigma_amp") = 0.0,
            py::arg("sigma_rate") = 0.0)
        .def_readonly("adaptation", &LeakyIntegratorDecay::adaptation)
        .def_readonly("accommodation", &LeakyIntegratorDecay::accommodation)
        .def_readonly("sigma_amp", &LeakyIntegratorDecay::sigma_amp)
        .def_readonly("sigma_rate", &LeakyIntegratorDecay::sigma_rate);
}

PYBIND11_MODULE(phastcpp, m)
{
    define_common(m);
    define_decay(m);
    define_fiber(m);
    define_approximated(m);
    define_phast(m);
}
