#include "../testing_utils.h"
#include <complex>
#include <limits>
#include <tuple>
#include <xsf/bessel.h>

TEST_CASE("iv tiny inputs", "[bessel][xsf_tests]") {
    using test_case = std::tuple<long, double, double, double>;

    // Reference values computed with mpmath.
    auto [n, z, ref, rtol] = GENERATE(
        test_case{1, 1e-50, 5e-51, 1e-14}, test_case{100, 0.1, 8.45293498689205e-289, 1e-10},
        test_case{10, 5.2250558491838786e-27, 4.0817497902418315e-273, 1e-14},
        test_case{3, 1e-50, 2.0833333333333333e-152, 1e-14}, test_case{10, 1e-29, 2.6911444554673707e-300, 1e-8},
        test_case{20, 1e-14, 3.919904349624791e-305, 1e-8}, test_case{200, 5, 5.065429051896385e-296, 1e-10}
    );

    double result = xsf::cyl_bessel_i(n, z); // calls cephes::iv
    double rel_err = xsf::extended_relative_error(result, ref);

    CAPTURE(n, z, result, ref, rel_err, rtol);
    REQUIRE(rel_err <= rtol);
}

// AMOS reports IERR=2 ("CABS(Z) too small") for every |z| below 1e3*DBL_MIN
// irrespective of the order, even though the result only overflows once
// v*log(2/|z|) exceeds the exponential range, i.e. for v > 0.94 at the very
// smallest subnormal. The v < 0 reflection formulas then combine a finite
// value with that failure. Before the leading-term branch these cases came
// back as NaN, +-inf or zero.
//
// Reference values computed with mpmath:
//
//     from mpmath import mp, besselj, besseli, mpf, mpc
//     mp.dps = 1000
//     besselj(mpf(v), mpc(mpf(z.real), mpf(z.imag)))   # besseli for I_v
//
// besseli does not converge at negative integer order, so the I_{-1} value was
// obtained from the exact identity I_{-n} = I_n. Each reference value above was
// also computed independently with Arb via python-flint at 600 bits, and the
// two agree on every one of them.
//
// The 5e-13 cases are the ones whose result is itself subnormal, where one ulp
// already spans 1e-13; the rest are held to 1e-15. Over 1300 random points in
// the region the relative error stayed below 6e-16 whenever the result was a
// normal number. The -0.9281245255377488 case is one that the earlier
// single-double form of the leading term got wrong by 1.8e-13.
//
TEST_CASE("cyl_bessel_j subnormal complex argument", "[bessel][xsf_tests]") {
    using test_case = std::tuple<double, std::complex<double>, std::complex<double>, double>;

    auto [v, z, ref, rtol] = GENERATE(
        test_case{-0.192414, {1.0901387e-310, 0.0}, {4.326755678778788e+59, 0.0}, 1e-15},
        test_case{0.5, {1e-310, 0.0}, {7.978845608028641e-156, 0.0}, 1e-15},
        test_case{-0.5, {1e-310, 0.0}, {7.978845608028666e+154, 0.0}, 1e-15},
        test_case{-1.0, {1e-310, 0.0}, {-5e-311, 0.0}, 5e-13},
        test_case{0.221616, {-5.5e-320, -1.5137764e-316}, {9.022405087028133e-71, -3.2749826617337185e-71}, 1e-15},
        test_case{-0.75, {3e-320, 1e-320}, {1.899439580418427e+239, -4.6746859099319957e+238}, 1e-15},
        test_case{
            -0.9281245255377488,
            {1.057466638724e-312, -1.125593994016e-312},
            {2.588030051375974e+288, 2.4494834943192487e+288},
            1e-15
        }
    );

    std::complex<double> result = xsf::cyl_bessel_j(v, z);
    double rel_err = xsf::extended_relative_error(result, ref);

    CAPTURE(v, z, result, ref, rel_err, rtol);
    REQUIRE(rel_err <= rtol);
}

TEST_CASE("cyl_bessel_i subnormal complex argument", "[bessel][xsf_tests]") {
    using test_case = std::tuple<double, std::complex<double>, std::complex<double>, double>;

    auto [v, z, ref, rtol] = GENERATE(
        test_case{-0.005785439999999999, {-1.421786e-317, -5e-324}, {68.12520569751437, 1.2383456180970775}, 1e-15},
        test_case{0.5, {1e-310, 0.0}, {7.978845608028641e-156, 0.0}, 1e-15},
        test_case{-1.0, {1e-310, 0.0}, {5e-311, 0.0}, 5e-13}
    );

    std::complex<double> result = xsf::cyl_bessel_i(v, z);
    double rel_err = xsf::extended_relative_error(result, ref);

    CAPTURE(v, z, result, ref, rel_err, rtol);
    REQUIRE(rel_err <= rtol);
}

// The leading-term branch sits in front of AMOS, so an infinite order must not
// reach it: Gamma(v + 1) and the power both degenerate and the J_{-n} symmetry
// would call fmod(infinity, 2). AMOS reports these as NaN and that is kept.
TEST_CASE("cyl_bessel_j and cyl_bessel_i infinite order at subnormal argument", "[bessel][xsf_tests]") {
    const double v = GENERATE(std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity());
    const std::complex<double> z = GENERATE(std::complex<double>{1e-310, 0.0}, std::complex<double>{3e-320, 1e-320});

    const std::complex<double> j = xsf::cyl_bessel_j(v, z);
    const std::complex<double> i = xsf::cyl_bessel_i(v, z);

    CAPTURE(v, z, j, i);
    REQUIRE(std::isnan(std::real(j)));
    REQUIRE(std::isnan(std::imag(j)));
    REQUIRE(std::isnan(std::real(i)));
    REQUIRE(std::isnan(std::imag(i)));
}
