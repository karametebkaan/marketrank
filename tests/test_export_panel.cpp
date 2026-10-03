// M4 --export-panel: arrays, meta.json, exact agreement with the walk-forward's functions, causality.
#include <doctest/doctest.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cli/export_panel.hpp"
#include "market/panel.hpp"
#include "market/synthetic_market.hpp"
#include "test_util.hpp"
#include "walkforward/universe.hpp"
#include "walkforward/walkforward.hpp"

using namespace mr;
namespace fs = std::filesystem;

namespace {

const Panel& xp_panel() {
  static const Panel p = [] {
    SyntheticConfig cfg;
    cfg.bars = 400;
    cfg.size_sigma = 0.5;
    BarStore store(test::temp_dir("export_panel"));
    auto secs = generate_synthetic(cfg, store);
    std::vector<std::string> tickers;
    for (auto& s : secs) tickers.push_back(s.ticker);
    return build_panel(store, tickers, cfg.tf);
  }();
  return p;
}

// Bitwise equality of float arrays (NaN payloads included).
bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

// Values at bars [0, upto] equal (NaN matches NaN).
bool same_prefix(const std::vector<float>& a, const std::vector<float>& b, std::size_t N, std::size_t upto) {
  for (std::size_t k = 0; k < (upto + 1) * N; ++k)
    if (!(a[k] == b[k] || (std::isnan(a[k]) && std::isnan(b[k])))) return false;
  return true;
}

}  // namespace

TEST_CASE("export-panel: arrays and meta.json round-trip; rebalance dates are the walk-forward's") {
  const Panel& panel = xp_panel();
  const std::size_t T = panel.T(), N = panel.N();
  PanelExportParams ep;
  ep.wf.min_dollar_volume = 0;  // the synthetic market is small; a non-trivial universe
  ep.wf.warmup_bars = 100;
  std::vector<std::string> sectors;
  for (std::size_t i = 0; i < N; ++i) sectors.push_back("S" + std::to_string(i % 3));
  const auto dir = test::temp_dir("export_panel_out");
  export_panel(panel, sectors, ep, dir);

  const std::vector<std::vector<float>> want{export_ret1(panel),  export_ldv(panel),
                                             export_dvshock(panel), export_vol20(panel),
                                             export_pressure(panel, ep.wf.core), export_elig(panel, ep.wf),
                                             export_label(panel, 5)};
  REQUIRE(export_array_names().size() == want.size());
  for (std::size_t k = 0; k < want.size(); ++k) {
    CAPTURE(export_array_names()[k]);
    const fs::path f = dir / (export_array_names()[k] + ".f32");
    REQUIRE(fs::exists(f));
    CHECK(fs::file_size(f) == T * N * sizeof(float));
    const auto v = read_f32(f);
    CHECK(same_bits(v, want[k]));
    std::size_t finite = 0;
    for (float x : v) finite += std::isfinite(x) ? 1 : 0;
    CHECK(finite > T * N / 2);  // every array carries data on this market
  }

  std::ifstream in(dir / "meta.json");
  const auto j = nlohmann::json::parse(in);
  CHECK(j.at("N").get<std::size_t>() == N);
  CHECK(j.at("T").get<std::size_t>() == T);
  CHECK(j.at("tickers").get<std::vector<std::string>>() == panel.tickers);
  CHECK(j.at("sectors").get<std::vector<std::string>>() == sectors);
  CHECK(j.at("times").get<std::vector<TimePoint>>() == panel.times);
  CHECK(j.at("label_horizons").at("label_w").get<int>() == 5);
  CHECK(j.at("eligibility").at("window").get<int>() == 20);
  CHECK(j.at("eligibility").at("top_n").get<int>() == 0);
  CHECK(j.contains("causality"));
  CHECK(j.at("feature_names").size() == 5);
  const auto bars = j.at("rebalance").at("bars").get<std::vector<std::size_t>>();
  CHECK(bars == walkforward_dates(panel, ep.wf));
  WalkForwardParams wf = ep.wf;
  wf.bt.base = {{panel.tickers[0], 1.0}};
  CHECK(bars == run_walkforward(panel, wf).dates);
  CHECK(bars.size() > 30);

  // Defaults are the walk-forward's (weekly, warm-up 252, 20 / 50e6 / 0, market_rank core).
  const PanelExportParams d;
  CHECK(d.wf.warmup_bars == 252);
  CHECK(d.wf.elig_window == 20);
  CHECK(d.wf.min_dollar_volume == 50e6);
  CHECK(d.wf.top_n == 0);
  CHECK(d.wf.rebalance == Rebalance::Weekly);
  CHECK(d.wf.core == CoreParams::market_rank());
  CHECK(d.label_h == 5);
  CHECK_THROWS_AS(export_panel(panel, {"x"}, ep, dir), std::invalid_argument);
}

TEST_CASE("export-panel: elig and label_w are eligible_at and forward_oo_return exactly") {
  const Panel& panel = xp_panel();
  const std::size_t T = panel.T(), N = panel.N();
  WalkForwardParams wf;
  wf.min_dollar_volume = 0;
  wf.top_n = N / 3;  // a binding, non-trivial mask
  const auto elig = export_elig(panel, wf);
  const auto label = export_label(panel, 5);
  std::size_t ones = 0;
  for (std::size_t t = 0; t < T; ++t) {
    const auto e = eligible_at(panel, t, 20, 0, N / 3);
    const auto r = forward_oo_return(panel, t, 5);
    for (std::size_t i = 0; i < N; ++i) {
      CHECK(elig[panel.idx(t, i)] == (e[i] ? 1.0f : 0.0f));
      const float want = static_cast<float>(r[i]);
      CHECK((label[panel.idx(t, i)] == want || (std::isnan(want) && std::isnan(label[panel.idx(t, i)]))));
      ones += e[i] ? 1 : 0;
    }
  }
  CHECK(ones > 0);
  CHECK(std::isnan(label[panel.idx(T - 6, 0)]));  // t + 1 + 5 >= T
}

TEST_CASE("export-panel: features at bar t do not depend on later bars") {
  const Panel& panel = xp_panel();
  const std::size_t N = panel.N();
  WalkForwardParams wf;
  wf.min_dollar_volume = 0;
  wf.top_n = N / 3;
  wf.elig_window = 3;  // one perturbed bar would move the median
  const auto base_ret = export_ret1(panel), base_ldv = export_ldv(panel), base_dvs = export_dvshock(panel),
             base_vol = export_vol20(panel), base_phi = export_pressure(panel, wf.core), base_el = export_elig(panel, wf);
  for (std::size_t cut : {std::size_t{150}, std::size_t{300}}) {
    Panel q = panel;
    for (std::size_t t = cut + 1; t < q.T(); ++t)  // every bar after `cut`
      for (std::size_t i = 0; i < N; ++i) {
        const double f = 1.0 + 0.05 * std::sin(static_cast<double>(t * 13 + i * 5));
        for (auto* v : {&q.open, &q.close, &q.vwap, &q.high, &q.low})
          if (std::isfinite((*v)[q.idx(t, i)])) (*v)[q.idx(t, i)] *= f;
        if (std::isfinite(q.volume[q.idx(t, i)])) q.volume[q.idx(t, i)] *= (i + t) % 2 ? 40.0 : 0.03;
      }
    CAPTURE(cut);
    const auto ret = export_ret1(q), ldv = export_ldv(q), dvs = export_dvshock(q), vol = export_vol20(q),
               phi = export_pressure(q, wf.core), el = export_elig(q, wf);
    CHECK(same_prefix(ret, base_ret, N, cut));
    CHECK(same_prefix(ldv, base_ldv, N, cut));
    CHECK(same_prefix(dvs, base_dvs, N, cut));
    CHECK(same_prefix(vol, base_vol, N, cut));
    CHECK(same_prefix(phi, base_phi, N, cut));
    CHECK(same_prefix(el, base_el, N, cut));
    // The perturbation is visible right after the cut (the check above is not vacuous).
    CHECK_FALSE(same_prefix(dvs, base_dvs, N, cut + 1));
    CHECK_FALSE(same_prefix(vol, base_vol, N, cut + 1));
    CHECK_FALSE(same_prefix(phi, base_phi, N, cut + 1));
  }
}

TEST_CASE("export-panel: feature definitions on a hand-built panel") {
  Panel p;
  const std::size_t T = 30;
  p.tickers = {"A", "B"};
  for (std::size_t t = 0; t < T; ++t) p.times.push_back(1700000000 + static_cast<TimePoint>(t) * 86400);
  const double nan = std::nan("");
  for (std::size_t t = 0; t < T; ++t) {
    const double ca = 100.0 * std::exp(0.01 * static_cast<double>(t) + (t % 2 ? 0.02 : 0.0));
    const double va = 1000.0 + 10.0 * static_cast<double>(t);
    for (auto* v : {&p.open, &p.high, &p.low, &p.vwap, &p.close}) v->push_back(ca), v->push_back(t == 3 ? nan : 50.0);
    p.volume.push_back(va);
    p.volume.push_back(10.0);
  }
  const auto ret = export_ret1(p), ldv = export_ldv(p), dvs = export_dvshock(p), vol = export_vol20(p);
  auto C = [&](std::size_t t, std::size_t i) { return p.close[p.idx(t, i)]; };
  auto DV = [&](std::size_t t, std::size_t i) { return p.close[p.idx(t, i)] * p.volume[p.idx(t, i)]; };
  CHECK(std::isnan(ret[p.idx(0, 0)]));
  CHECK(ret[p.idx(5, 0)] == static_cast<float>(std::log(C(5, 0) / C(4, 0))));
  CHECK(std::isnan(ret[p.idx(3, 1)]));  // missing close at 3
  CHECK(std::isnan(ret[p.idx(4, 1)]));
  CHECK(ldv[p.idx(7, 0)] == static_cast<float>(std::log(DV(7, 0))));
  // dvshock at t = 25: median of dv over bars 5..24 (20 values, increasing in t for A: mean of the 10th and 11th
  // smallest). dv of A is not monotone (alternating close), so compute the median by sorting.
  {
    std::vector<double> w;
    for (std::size_t u = 5; u < 25; ++u) w.push_back(DV(u, 0));
    std::sort(w.begin(), w.end());
    CHECK(dvs[p.idx(25, 0)] == static_cast<float>(DV(25, 0) / (0.5 * (w[9] + w[10]))));
  }
  CHECK(std::isnan(dvs[p.idx(9, 0)]));        // 9 previous values < 10
  CHECK(!std::isnan(dvs[p.idx(10, 0)]));      // 10 previous values
  CHECK(dvs[p.idx(12, 1)] == 1.0f);           // B: flat dollar volume (bar 3 missing, skipped)
  // vol20 at t = 25: sample sd of ret1 over 6..25.
  {
    double m = 0, ss = 0;
    for (std::size_t u = 6; u <= 25; ++u) m += std::log(C(u, 0) / C(u - 1, 0));
    m /= 20;
    for (std::size_t u = 6; u <= 25; ++u) ss += std::pow(std::log(C(u, 0) / C(u - 1, 0)) - m, 2);
    CHECK(vol[p.idx(25, 0)] == static_cast<float>(std::sqrt(ss / 19)));
  }
  CHECK(std::isnan(vol[p.idx(9, 0)]));   // ret1 at 1..9: 9 values
  CHECK(!std::isnan(vol[p.idx(10, 0)]));
}
