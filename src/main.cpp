#include <omp.h>

#include <atomic>
#include <csignal>
#include <thread>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli/args.hpp"
#include "core/time.hpp"
#include "market/alpaca_client.hpp"
#include "market/asset_universe.hpp"
#include "market/market_sync.hpp"
#include "market/panel.hpp"
#include "market/sec_sectors.hpp"
#include "market/synthetic_market.hpp"
#include "market/universe.hpp"
#include "pipeline/core_pipeline.hpp"
#include "pipeline/evaluation.hpp"
#include "pipeline/shock.hpp"
#include "server/http_server.hpp"
#include "storage/csv_migration.hpp"
#include "storage/lake.hpp"

namespace {
namespace fs = std::filesystem;

std::atomic<bool> g_signalled{false};  // lock-free atomic: safe to set from a signal handler
extern "C" void on_signal(int) { g_signalled.store(true); }

fx::TimePoint now_utc() {
  return std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

std::string today_string() { return fx::format_rfc3339(now_utc()).substr(0, 10); }

std::int64_t days_since(const std::string& ymd) {
  const int y = std::stoi(ymd.substr(0, 4));
  const auto m = static_cast<unsigned>(std::stoi(ymd.substr(5, 2)));
  const auto d = static_cast<unsigned>(std::stoi(ymd.substr(8, 2)));
  return fx::floor_div(now_utc(), 86400) - fx::days_from_civil(y, m, d);
}

fx::Universe sp500_universe(const fs::path& data) {
  return fx::Universe::load(data / "universe" / "sp500.csv", data / "universe" / "funds.csv");
}

// Compaction and retention, shared by --maintain and the post-sync pass. Retention never
// reaches into the lookback window: a shorter keep_days is warned about and skipped.
void maintain_lake(fx::BarStore& store, const fs::path& data, int lookback_days,
                   fx::Timeframe lookback_tf) {
  auto policy = fx::RetentionPolicy::load(data / "lake" / "retention.json");
  for (auto& [tf, keep] : policy.keep_days) {
    const int needed = tf == lookback_tf ? lookback_days
                       : tf == fx::Timeframe::Hour ? 60
                       : tf == fx::Timeframe::Day  ? 365
                                                   : 5 * 365;
    if (keep && *keep < needed) {
      std::cerr << "warning: retention keeps " << *keep << " days of " << fx::to_string(tf)
                << " bars but the lookback is " << needed << "; skipping retention for it\n";
      keep = std::nullopt;
    }
  }
  std::size_t compacted = 0;
  for (auto tf : {fx::Timeframe::Hour, fx::Timeframe::Day, fx::Timeframe::Week})
    compacted += store.lake().compact(tf, 8);
  const auto removed = store.lake().apply_retention(policy, now_utc());
  std::cerr << "compacted " << compacted << " partitions, removed " << removed
            << " expired partitions\n";
}

fs::path sec_cache_path(const fs::path& data) { return data / "sectors" / "sec_sic.csv"; }

// Spec: S&P GICS sector, then SEC SIC sector, then ETF/Fund, then Unclassified; filled at load time.
void fill_sectors(fx::Universe& universe, const fs::path& data) {
  fx::apply_sector_fill(universe, fx::load_sec_cache(sec_cache_path(data)));
}

std::map<std::string, std::size_t> sector_counts(const fx::Universe& u) {
  std::map<std::string, std::size_t> out;
  for (const auto& s : u.nodes()) ++out[s.sector];
  return out;
}

double unclassified_pct(const fx::Universe& u) {
  if (u.nodes().empty()) return 0;
  std::size_t n = 0;
  for (const auto& s : u.nodes())
    if (s.sector == fx::kSectorUnclassified || s.sector.empty()) ++n;
  return 100.0 * static_cast<double>(n) / static_cast<double>(u.nodes().size());
}

// Resolves the universe as rank does (replay path), fetches SIC for every ticker, prints coverage.
int run_sync_sectors(const fx::CliArgs& args) {
  const fs::path dir = args.data / "universe";
  std::optional<fs::path> snapshot;
  if (args.universe != fx::UniverseSource::Sp500) snapshot = fx::latest_snapshot(dir);
  if (!snapshot && args.universe == fx::UniverseSource::Snapshot)
    throw std::runtime_error("no universe snapshot; run --mode alpaca first");
  fx::Universe universe =
      snapshot ? fx::load_snapshot(*snapshot, dir / "funds.csv") : sp500_universe(args.data);
  const double before = unclassified_pct(universe);
  fx::load_dotenv(".env");
  fx::SecSyncOptions opt;
  opt.now = now_utc();
  const auto tickers = universe.node_tickers();
  std::cerr << "syncing SEC SIC codes for " << tickers.size() << " tickers"
            << (snapshot ? " from " + snapshot->string() : std::string(" (sp500)")) << "...\n";
  const auto stats = fx::sync_sec_sectors(tickers, sec_cache_path(args.data), fx::make_sec_client, opt);
  fill_sectors(universe, args.data);
  std::printf("sec sync: %zu fetched, %zu fresh in cache, %zu without CIK, %zu failed (retried next run)\n",
              stats.fetched, stats.fresh, stats.no_cik, stats.failed);
  std::printf("%-26s %7s\n", "sector", "count");
  for (const auto& [sector, n] : sector_counts(universe)) std::printf("%-26s %7zu\n", sector.c_str(), n);
  std::printf("unclassified: %.1f%% before, %.1f%% after (%zu tickers)\n", before,
              unclassified_pct(universe), universe.nodes().size());
  return 0;
}

fs::path ensure_snapshot(const fx::CliArgs& args, const fx::AlpacaConfig& cfg,
                         fx::AlpacaClient& data_client, fx::BarStore& store,
                         const fx::PortfolioSpec& portfolio) {
  const fs::path dir = args.data / "universe";
  // Reuse a fresh (< 7 days) snapshot built for exactly the requested size.
  if (!args.refresh_universe)
    if (auto found = fx::find_snapshot(dir, args.universe_size, now_utc(), 7)) return *found;
  fx::AlpacaConfig trading_cfg = cfg;
  trading_cfg.host = cfg.trading_host;
  fx::AlpacaClient trading(trading_cfg);
  std::cerr << "fetching asset list from " << cfg.trading_host << "...\n";
  const auto assets =
      fx::parse_assets(trading.get("/v2/assets?status=active&asset_class=us_equity"));
  const fx::Universe sp = sp500_universe(args.data);
  fx::UniverseRules rules;
  for (const auto& t : sp.node_tickers()) rules.always_include.insert(t);
  for (const auto& h : portfolio.holdings)
    if (!sp.is_fund(h.ticker)) rules.always_include.insert(h.ticker);
  for (const auto& t : fx::read_ticker_list(dir / "include.csv")) rules.always_include.insert(t);
  rules.exclude = fx::read_ticker_list(dir / "exclude.csv");
  std::vector<fx::AssetInfo> candidates;
  for (const auto& a : assets)
    if (fx::passes_universe_rules(a, rules)) candidates.push_back(a);
  std::vector<std::string> symbols;
  for (const auto& a : candidates) symbols.push_back(a.symbol);
  const fx::TimePoint end = now_utc() - 16 * 60;
  store.load_range(symbols, fx::Timeframe::Day, end - 40 * 86400, end);
  std::cerr << assets.size() << " assets, " << candidates.size()
            << " candidates; fetching 40 days of daily bars to rank liquidity...\n";
  const auto stale = fx::sync_bars(data_client, store, symbols, fx::Timeframe::Day,
                                   end - 40 * 86400, end);
  if (!stale.empty()) std::cerr << stale.size() << " stale tickers during ranking\n";
  const auto ranked = fx::rank_by_liquidity(candidates, store, 20, args.universe_size);
  const fs::path path =
      dir / ("universe_" + today_string() + "_n" + std::to_string(args.universe_size) + ".csv");
  fx::write_universe_snapshot(path, ranked, sp);
  std::cerr << "wrote " << ranked.size() << "-ticker universe snapshot " << path.string() << "\n";
  return path;
}

void print_row(std::size_t rank, const fx::Security& s, double h, double pi, double score) {
  std::printf("%5zu  %-7s %-24.24s %+10.4f  %.7f  %+9.4f\n", rank, s.ticker.c_str(),
              s.sector.c_str(), h, pi, score);
}

int run_eval(const fx::CliArgs& args, const fx::Panel& panel, const fx::Universe& universe) {
  std::printf("evaluation: last %zu of %zu bars, nodes=%zu, timeframe=%s, threads=%d\n\n",
              args.eval_bars, panel.T(), panel.N(), std::string(fx::to_string(args.tf)).c_str(),
              omp_get_max_threads());
  std::printf("%-20s %6s %6s %6s %6s %9s %6s %9s %6s %9s %6s %9s %6s %9s\n", "config", "floor",
              "gini", "coher", "struct", "IC(score)", "t", "IC(h)", "t", "IC_oo", "t", "IC_h_oo", "t",
              "ms/frame");
  for (const auto& c : fx::evaluation_grid()) {
    const fx::EvalMetrics m = fx::evaluate(panel, universe.nodes(), c.params, args.eval_bars);
    std::printf("%-20s %6.3f %6.3f %6.3f %6.3f %+9.4f %+6.2f %+9.4f %+6.2f %+9.4f %+6.2f %+9.4f %+6.2f %9.1f\n",
                c.name.c_str(), m.floor_share, m.gini, m.sector_coherence, m.structure_gain, m.ic_mean,
                m.ic_t, m.ic_h_mean, m.ic_h_t, m.ic_oo_mean, m.ic_oo_t, m.ic_h_oo_mean, m.ic_h_oo_t,
                m.mean_frame_ms);
    std::fflush(stdout);
  }
  return 0;
}

int run_shock(const fx::CliArgs& args, const fx::Panel& panel, const fx::Universe& universe,
              const std::optional<fx::PortfolioSpec>& portfolio) {
  std::vector<fx::Shock> shocks;
  for (const auto& [ticker, size] : args.shocks) {
    const auto i = universe.index_of(ticker);
    if (!i) throw std::invalid_argument("--shock: unknown ticker " + ticker);
    shocks.push_back({*i, size});
  }
  const auto [base, shocked] = fx::run_with_shock(panel, args.params, shocks);  // throws if inactive
  const fx::ShockDelta d = fx::shock_response(base, shocked);
  const auto& nodes = universe.nodes();
  std::printf("mode=%s timeframe=%s nodes=%zu bars=%zu last=%s threads=%d\n", args.mode.c_str(),
              std::string(fx::to_string(args.tf)).c_str(), panel.N(), panel.T(),
              fx::format_rfc3339(base.t).c_str(), omp_get_max_threads());
  std::printf("params: %s\n", fx::describe(args.params).c_str());
  std::printf("SHOCK at the last bar (extra SIZE%% return at normal volume; <0 sell-off, >0 buying surge)\n");
  std::printf("%-7s %9s %10s %12s\n", "ticker", "size", "dh", "dpi");
  for (const auto& s : shocks)
    std::printf("%-7s %+9.2f %+10.4f %+12.3e\n", nodes[s.node].ticker.c_str(), s.size, d.dh[s.node],
                d.dpi[s.node]);
  std::printf("total |dpi| (L1) = %.4e\n\n", d.l1_dpi);

  std::vector<std::size_t> idx;
  for (std::size_t i = 0; i < panel.N(); ++i)
    if (base.active[i] && shocked.active[i]) idx.push_back(i);
  std::sort(idx.begin(), idx.end(), [&](auto a, auto b) {
    return d.dh[a] > d.dh[b] || (d.dh[a] == d.dh[b] && nodes[a].ticker < nodes[b].ticker);
  });
  const std::size_t top = std::min(args.top, idx.size());
  auto row = [&](std::size_t i) {
    std::printf("  %-7s %-24.24s %+10.4f %+12.3e %+10.4f\n", nodes[i].ticker.c_str(),
                nodes[i].sector.c_str(), d.dh[i], d.dpi[i], d.dscore[i]);
  };
  std::printf("RECEIVERS (largest dh)\n  %-7s %-24s %10s %12s %10s\n", "ticker", "sector", "dh", "dpi",
              "dscore+1");
  for (std::size_t r = 0; r < top; ++r) row(idx[r]);
  std::printf("\nLOSERS (largest drop in dh)\n");
  for (std::size_t r = 0; r < top; ++r) row(idx[idx.size() - 1 - r]);
  if (portfolio) {
    std::printf("\nPORTFOLIO HOLDINGS (dh)\n");
    for (const auto& hld : portfolio->holdings) {
      const auto i = universe.index_of(hld.ticker);
      if (i && base.active[*i] && shocked.active[*i])
        std::printf("  %-6s %5.1f%%  dh %+10.4f\n", hld.ticker.c_str(), hld.weight * 100, d.dh[*i]);
      else
        std::printf("  %-6s %5.1f%%  (no data or fund)\n", hld.ticker.c_str(), hld.weight * 100);
    }
  }
  return 0;
}

int run_rank(const fx::CliArgs& args, const fx::Panel& panel, const fx::Universe& universe,
             const std::optional<fx::PortfolioSpec>& portfolio) {
  const auto t0 = std::chrono::steady_clock::now();
  const fx::Frame f = fx::run_panel_last(panel, args.params);
  const double total_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  std::printf("mode=%s timeframe=%s nodes=%zu bars=%zu last=%s threads=%d\n", args.mode.c_str(),
              std::string(fx::to_string(args.tf)).c_str(), panel.N(), panel.T(),
              fx::format_rfc3339(f.t).c_str(), omp_get_max_threads());
  std::printf("params: %s\n", fx::describe(args.params).c_str());
  std::printf("solver: %s in %d iterations (residual %.2e); last frame %.1f ms, run %.0f ms\n",
              f.solve.converged ? "converged" : "NOT converged", f.solve.iterations,
              f.solve.residual, f.compute_ms, total_ms);
  std::printf("edges: %zu slow, %zu fast; %.1f%% of active nodes at the teleport floor\n\n",
              f.P.col.size(), f.P_fast.col.size(), 100.0 * fx::floor_share(f, args.params.alpha));

  const auto& nodes = universe.nodes();
  std::vector<std::size_t> hills;
  for (std::size_t i = 0; i < panel.N(); ++i)
    if (f.active[i]) hills.push_back(i);
  const std::size_t inactive = panel.N() - hills.size();
  auto by_ticker = [&](std::size_t a, std::size_t b) { return nodes[a].ticker < nodes[b].ticker; };
  std::sort(hills.begin(), hills.end(), [&](auto a, auto b) {
    return f.h[a] > f.h[b] || (f.h[a] == f.h[b] && by_ticker(a, b));
  });
  std::vector<std::size_t> valleys(hills.rbegin(), hills.rend());
  std::stable_sort(valleys.begin(), valleys.end(), [&](auto a, auto b) { return f.h[a] < f.h[b]; });
  const auto& score = f.forecasts.front().score;
  const std::size_t top = std::min(args.top, hills.size());
  if (inactive > 0) std::printf("%zu inactive (stale or below liquidity floor)\n\n", inactive);
  std::printf("HILLS (money accumulating)            hotness         pi     score+%d\n",
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
        std::printf("  %-6s %5.1f%%  hotness %+9.4f  score+%d %+9.4f\n", hld.ticker.c_str(),
                    hld.weight * 100, f.h[*i], f.forecasts.front().k, score[*i]);
      } else {
        std::printf("  %-6s %5.1f%%  (fund: look-through hotness arrives in milestone 3)\n",
                    hld.ticker.c_str(), hld.weight * 100);
      }
    }
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const fx::CliArgs args = fx::parse_cli(std::vector<std::string>(argv + 1, argv + argc));
    if (args.help) {
      std::cout << fx::cli_usage();
      return 0;
    }
    if (args.threads > 0) omp_set_num_threads(args.threads);
    if (args.migrate_cache) {
      fx::BarStore lake_store(args.data / "lake");
      const auto n = fx::migrate_csv_cache(args.migrate_from, lake_store);
      std::cout << "migrated " << n << " series from " << args.migrate_from.string() << " into "
                << (args.data / "lake").string() << "\n";
      return 0;
    }
    if (args.sync_sectors) return run_sync_sectors(args);
    if (args.maintain) {
      fx::BarStore lake_store(args.data / "lake");
      maintain_lake(lake_store, args.data, args.lookback_days, args.tf);
      return 0;
    }
    fx::BarStore store(args.data / "lake");
    const auto [window_start, end] = fx::data_window(args, now_utc());
    fx::Universe universe;
    std::optional<fx::PortfolioSpec> portfolio;

    if (args.mode == "synthetic") {
      fx::SyntheticConfig cfg;
      cfg.tf = args.tf;
      universe = fx::Universe::from_securities(fx::generate_synthetic(cfg, store));
    } else {
      portfolio = fx::load_portfolio(args.data / "portfolio.json");
      const fs::path dir = args.data / "universe";
      std::optional<fx::AlpacaConfig> cfg;
      std::optional<fx::AlpacaClient> client;
      if (args.mode == "alpaca") {
        fx::load_dotenv(".env");
        cfg = fx::alpaca_config_from_env();
        if (!cfg) throw std::runtime_error("APCA_API_KEY_ID / APCA_API_SECRET_KEY not set (.env)");
        client.emplace(*cfg);
      }
      std::optional<fs::path> snapshot;
      if (args.universe != fx::UniverseSource::Sp500) {
        snapshot = args.mode == "alpaca" ? ensure_snapshot(args, *cfg, *client, store, *portfolio)
                                         : fx::latest_snapshot(dir);
        if (!snapshot && args.universe == fx::UniverseSource::Snapshot)
          throw std::runtime_error("no universe snapshot; run --mode alpaca first");
      }
      universe = snapshot ? fx::load_snapshot(*snapshot, dir / "funds.csv")
                          : sp500_universe(args.data);
      fill_sectors(universe, args.data);
      universe.add_extras(*portfolio);
      store.load_range(universe.price_tickers(), args.tf, window_start, end);
      if (client) {
        const fx::TimePoint start = window_start;
        std::cerr << (args.refetch_full ? "refetching full history of " : "syncing ")
                  << universe.price_tickers().size() << " tickers (" << fx::to_string(args.tf) << ") from "
                  << fx::format_rfc3339(start) << " (or earlier stored history)...\n";
        const auto stale = fx::sync_bars(*client, store, universe.price_tickers(), args.tf, start, end,
                                         args.refetch_full);
        if (!stale.empty()) std::cerr << stale.size() << " stale tickers\n";
        store.flush();
        maintain_lake(store, args.data, args.lookback_days, args.tf);
      }
    }

    const fx::Panel panel = fx::build_panel(store, universe.node_tickers(), args.tf, window_start, end);
    if (panel.T() < 2) throw std::runtime_error("not enough cached bars; run with --mode alpaca first");
    if (args.serve) {
      if (!fx::is_loopback_host(args.host))
        std::cerr << "warning: --host " << args.host
                  << " is not a loopback address: the server (which has no authentication) is reachable from the "
                     "network\n";
      fx::FrameStore frames(panel, universe.nodes(), args.params, fx::LandscapeParams{});
      frames.start();
      fx::FluxServer server(frames, portfolio, args.mode + " " + std::string(fx::to_string(args.tf)));
      if (!fs::is_directory(args.web))
        std::cerr << "warning: web root '" << args.web.string() << "' not found; static files will 404\n";
      const int port = server.bind({args.host, args.port, args.web});
      std::cerr << "serving http://" << args.host << ":" << port << "  (Ctrl-C to stop)\n";
      std::signal(SIGINT, on_signal);
      std::signal(SIGTERM, on_signal);
      std::atomic<bool> finished{false};
      std::thread watcher([&] {  // does the non-async-signal-safe work outside the handler
        while (!finished && !g_signalled) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (g_signalled) server.stop();
      });
      const bool ok = server.listen();
      finished = true;
      watcher.join();
      if (!ok && !g_signalled) {
        std::cerr << "fluxscape: server failed to listen\n";
        return 1;
      }
      return 0;
    }
    if (args.eval) return run_eval(args, panel, universe);
    if (!args.shocks.empty()) return run_shock(args, panel, universe, portfolio);
    return run_rank(args, panel, universe, portfolio);
  } catch (const std::exception& e) {
    std::cerr << "fluxscape: " << e.what() << "\n";
    return 1;
  }
}
