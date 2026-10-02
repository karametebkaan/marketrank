#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>

#include "core/time.hpp"
#include "market/alpaca_client.hpp"
#include "market/market_sync.hpp"
#include "market/panel.hpp"
#include "market/synthetic_market.hpp"
#include "market/universe.hpp"
#include "pipeline/core_pipeline.hpp"

namespace {

struct Args {
  std::string mode = "synthetic";
  fx::Timeframe tf = fx::Timeframe::Day;
  int lookback_days = -1;
  std::size_t top = 15;
  std::filesystem::path data = "data";
};

Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) throw std::invalid_argument("missing value for " + flag);
      return argv[++i];
    };
    if (flag == "--mode") a.mode = value();
    else if (flag == "--timeframe") a.tf = fx::parse_timeframe(value());
    else if (flag == "--lookback-days") a.lookback_days = std::stoi(value());
    else if (flag == "--top") a.top = static_cast<std::size_t>(std::stoul(value()));
    else if (flag == "--data") a.data = value();
    else if (flag == "--help" || flag == "-h") {
      std::cout << "usage: fluxscape [--mode synthetic|replay|alpaca] [--timeframe 1h|1d|1w]\n"
                   "                 [--lookback-days N] [--top N] [--data DIR]\n";
      std::exit(0);
    } else throw std::invalid_argument("unknown flag " + flag);
  }
  if (a.mode != "synthetic" && a.mode != "replay" && a.mode != "alpaca")
    throw std::invalid_argument("--mode must be synthetic, replay or alpaca");
  if (a.lookback_days < 0)
    a.lookback_days = a.tf == fx::Timeframe::Hour ? 60 : a.tf == fx::Timeframe::Day ? 365 : 5 * 365;
  return a;
}

void print_row(std::size_t rank, const fx::Security& s, double h, double pi, double score) {
  std::printf("%4zu  %-7s %-24.24s %+9.4f  %.6f  %+9.4f\n", rank, s.ticker.c_str(),
              s.sector.c_str(), h, pi, score);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    fx::BarStore store(args.data / "cache");
    fx::Universe universe;
    std::optional<fx::PortfolioSpec> portfolio;

    if (args.mode == "synthetic") {
      fx::SyntheticConfig cfg;
      cfg.tf = args.tf;
      universe = fx::Universe::from_securities(fx::generate_synthetic(cfg, store));
    } else {
      universe = fx::Universe::load(args.data / "universe" / "sp500.csv",
                                    args.data / "universe" / "funds.csv");
      portfolio = fx::load_portfolio(args.data / "portfolio.json");
      universe.add_extras(*portfolio);
      store.load_all(universe.price_tickers(), args.tf);
      if (args.mode == "alpaca") {
        fx::load_dotenv(".env");
        auto cfg = fx::alpaca_config_from_env();
        if (!cfg) throw std::runtime_error("APCA_API_KEY_ID / APCA_API_SECRET_KEY not set (.env)");
        fx::AlpacaClient client(*cfg);
        const auto now = std::chrono::duration_cast<std::chrono::seconds>(
                             std::chrono::system_clock::now().time_since_epoch())
                             .count();
        const fx::TimePoint end = now - 16 * 60;
        const fx::TimePoint start = end - static_cast<fx::TimePoint>(args.lookback_days) * 86400;
        std::cerr << "syncing " << universe.price_tickers().size() << " tickers ("
                  << fx::to_string(args.tf) << ") from " << fx::format_rfc3339(start) << "...\n";
        const auto stale = fx::sync_bars(client, store, universe.price_tickers(), args.tf, start, end);
        if (!stale.empty()) std::cerr << stale.size() << " stale tickers\n";
      }
    }

    const fx::Panel panel = fx::build_panel(store, universe.node_tickers(), args.tf);
    if (panel.T() < 2) {
      throw std::runtime_error("not enough cached bars; run with --mode alpaca first");
    }
    const fx::CoreParams params;
    const auto t0 = std::chrono::steady_clock::now();
    const fx::Frame f = fx::run_panel_last(panel, params);
    const double total_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    std::printf("mode=%s timeframe=%s nodes=%zu bars=%zu last=%s\n", args.mode.c_str(),
                std::string(fx::to_string(args.tf)).c_str(), panel.N(), panel.T(),
                fx::format_rfc3339(f.t).c_str());
    std::printf("solver: %s in %d iterations (residual %.2e); last frame %.1f ms, run %.0f ms\n\n",
                f.solve.converged ? "converged" : "NOT converged", f.solve.iterations,
                f.solve.residual, f.compute_ms, total_ms);

    const auto& nodes = universe.nodes();
    std::vector<std::size_t> hills;
    for (std::size_t i = 0; i < panel.N(); ++i)
      if (f.active[i]) hills.push_back(i);
    const std::size_t inactive = panel.N() - hills.size();
    auto by_ticker = [&](std::size_t a, std::size_t b) { return nodes[a].ticker < nodes[b].ticker; };
    std::sort(hills.begin(), hills.end(), [&](auto a, auto b) {
      return f.h[a] > f.h[b] || (f.h[a] == f.h[b] && by_ticker(a, b));
    });
    std::vector<std::size_t> valleys(hills);
    std::sort(valleys.begin(), valleys.end(), [&](auto a, auto b) {
      return f.h[a] < f.h[b] || (f.h[a] == f.h[b] && by_ticker(a, b));
    });
    const auto& score = f.forecasts.front().score;
    const std::size_t top = std::min(args.top, hills.size());
    if (inactive > 0) std::printf("%zu inactive (no data)\n\n", inactive);
    std::printf("HILLS (money accumulating)          hotness        pi     score+%d\n",
                f.forecasts.front().k);
    for (std::size_t r = 0; r < top; ++r)
      print_row(r + 1, nodes[hills[r]], f.h[hills[r]], f.pi[hills[r]], score[hills[r]]);
    std::printf("\nVALLEYS (money draining)\n");
    for (std::size_t r = 0; r < top; ++r) {
      const std::size_t i = valleys[r];
      print_row(valleys.size() - r, nodes[i], f.h[i], f.pi[i], score[i]);
    }
    if (portfolio) {
      std::printf("\nPORTFOLIO HOLDINGS\n");
      for (const auto& hld : portfolio->holdings) {
        if (auto i = universe.index_of(hld.ticker); i && !f.active[*i]) {
          std::printf("  %-6s %5.1f%%  (no data)\n", hld.ticker.c_str(), hld.weight * 100);
        } else if (i) {
          std::printf("  %-6s %5.1f%%  hotness %+8.4f  score+%d %+8.4f\n", hld.ticker.c_str(),
                      hld.weight * 100, f.h[*i], f.forecasts.front().k, score[*i]);
        } else {
          std::printf("  %-6s %5.1f%%  (fund: look-through hotness arrives in milestone 3)\n",
                      hld.ticker.c_str(), hld.weight * 100);
        }
      }
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "fluxscape: " << e.what() << "\n";
    return 1;
  }
}
