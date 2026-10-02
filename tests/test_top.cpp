#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "server/top_list.hpp"

using namespace mr;

namespace {
using FramePtr = std::shared_ptr<const LandscapeFrame>;
const double kNaN = std::numeric_limits<double>::quiet_NaN();

// Frame at t with (i, h) nodes; nodes are stored in ascending i like the builder emits them.
FramePtr mk(TimePoint t, std::vector<std::pair<std::uint32_t, double>> nh) {
  auto f = std::make_shared<LandscapeFrame>();
  f->t = t;
  std::sort(nh.begin(), nh.end());
  for (auto [i, h] : nh)
    f->nodes.push_back({i, static_cast<std::int32_t>(i), 0.f, 0.f, h, display_height(h, HeightMode::SignedLog), 0.1 * i, -h});
  return f;
}
}  // namespace

TEST_CASE("top_hot: exact h descending, ties by lower i, non-finite excluded, n caps the list") {
  auto f = mk(10, {{0, 1.0}, {1, 3.0}, {2, kNaN}, {3, 3.0}, {4, -2.0}, {5, std::numeric_limits<double>::infinity()},
                   {6, 0.5}});
  auto rows = top_hot({f}, 10, 1, TopBy::Hotness);
  REQUIRE(rows.size() == 5);  // NaN and +inf excluded
  const std::vector<std::uint32_t> want{1, 3, 0, 6, 4};
  for (std::size_t k = 0; k < rows.size(); ++k) {
    CHECK(rows[k].i == want[k]);
    CHECK(rows[k].rank == k + 1);
    CHECK_FALSE(rows[k].prev_rank.has_value());  // no previous bar
  }
  CHECK(rows[0].h == 3.0);
  CHECK(rows[0].hdisp == doctest::Approx(std::log1p(3.0)));
  CHECK(rows[0].pi == doctest::Approx(0.1));
  CHECK(rows[0].score == -3.0);
  CHECK(top_hot({f}, 2, 1, TopBy::Hotness).size() == 2);
  CHECK(top_hot({}, 5, 1).empty());
}

TEST_CASE("top_hot: prev_rank is the rank at the previous cached bar, null when absent or non-finite") {
  auto prev = mk(9, {{0, 5.0}, {1, 1.0}, {3, kNaN}, {6, 2.0}});
  auto cur = mk(10, {{0, 1.0}, {1, 3.0}, {3, 2.0}, {6, 0.5}, {7, 9.0}});
  auto rows = top_hot({prev, cur}, 10, 1, TopBy::Hotness);
  REQUIRE(rows.size() == 5);
  // current order: 7 (9.0), 1 (3.0), 3 (2.0), 0 (1.0), 6 (0.5); previous order: 0, 6, 1
  CHECK(rows[0].i == 7);
  CHECK_FALSE(rows[0].prev_rank.has_value());  // inactive before
  CHECK(rows[1].prev_rank == std::optional<std::size_t>(3));
  CHECK_FALSE(rows[2].prev_rank.has_value());  // NaN before
  CHECK(rows[3].prev_rank == std::optional<std::size_t>(1));
  CHECK(rows[4].prev_rank == std::optional<std::size_t>(2));
}

TEST_CASE("top_hot: series covers the last `bars` bars ending at t, oldest first, null where missing") {
  auto f1 = mk(1, {{0, 1.0}, {1, 2.0}});
  auto f2 = mk(2, {{1, 4.0}});           // node 0 inactive
  auto f3 = mk(3, {{0, 3.0}, {1, kNaN}});  // node 1 non-finite
  auto f4 = mk(4, {{0, 5.0}, {1, 6.0}});
  auto rows = top_hot({f1, f2, f3, f4}, 2, 3, TopBy::Hotness);
  REQUIRE(rows.size() == 2);
  REQUIRE(rows[0].i == 1);
  REQUIRE(rows[0].series.size() == 3);
  CHECK(rows[0].series[0] == 4.0);
  CHECK(std::isnan(rows[0].series[1]));
  CHECK(rows[0].series[2] == 6.0);
  CHECK(std::isnan(rows[1].series[0]));
  CHECK(rows[1].series[1] == 3.0);
  CHECK(rows[1].series[2] == 5.0);
  // More bars than cached: padded with nulls at the front, still ending at t.
  auto wide = top_hot({f3, f4}, 1, 5, TopBy::Hotness);
  REQUIRE(wide[0].series.size() == 5);
  for (int k = 0; k < 3; ++k) CHECK(std::isnan(wide[0].series[static_cast<std::size_t>(k)]));
  CHECK(wide[0].series[4] == 6.0);
  CHECK(std::isnan(wide[0].series[3]));
}

TEST_CASE("top_hot by π sorts on π directly, ties by lower i") {
  auto f = std::make_shared<LandscapeFrame>();
  f->t = 1;
  for (std::uint32_t i = 0; i < 4; ++i) {
    const double pi = i == 3 ? 0.4 : 0.2;  // 0, 1, 2 tie
    f->nodes.push_back({i, static_cast<std::int32_t>(i), 0.f, 0.f, 0.0, 0.0, pi, 0.0});
  }
  const auto rows = top_hot({f}, 4, 1);
  REQUIRE(rows.size() == 4);
  CHECK(rows[0].i == 3);
  CHECK(rows[1].i == 0);
  CHECK(rows[2].i == 1);
  CHECK(rows[3].i == 2);
}
