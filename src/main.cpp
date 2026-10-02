#include <omp.h>

#include <atomic>
#include <csignal>
#include <thread>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "analysis/cluster_persistence.hpp"
#include "cli/args.hpp"
#include "cli/rank_report.hpp"
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
#include "pipeline/graph_slice.hpp"
#include "pipeline/shock.hpp"
#include "server/http_server.hpp"
#include "storage/csv_migration.hpp"
#include "storage/lake.hpp"
#include "walkforward/report.hpp"
#include "walkforward/walkforward.hpp"

namespace {
namespace fs = std::filesystem;

std::atomic<bool> g_signalled{false};  // lock-free atomic: safe to set from a signal handler
extern "C" void on_signal(int) { g_signalled.store(true); }

mr::TimePoint now_utc() {
  return std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

std::string today_string() { return mr::format_rfc3339(now_utc()).substr(0, 10); }

std::int64_t days_since(const std::string& ymd) {
  const int y = std::stoi(ymd.substr(0, 4));
  const auto m = static_cast<unsigned>(std::stoi(ymd.substr(5, 2)));
  const auto d = static_cast<unsigned>(std::stoi(ymd.substr(8, 2)));
  return mr::floor_div(now_utc(), 86400) - mr::days_from_civil(y, m, d);
}

mr::Universe sp500_universe(const fs::path& data) {
  return mr::Universe::load(data / "universe" / "sp500.csv", data / "universe" / "funds.csv");
}

// Compaction and retention, shared by --maintain and the post-sync pass. Retention never
// reaches into the lookback window: a shorter keep_days is warned about and skipped.
void maintain_lake(mr::BarStore& store, const fs::path& data, int lookback_days,
                   mr::Timeframe lookback_tf) {
  auto policy = mr::RetentionPolicy::load(data / "lake" / "retention.json");
  for (auto& [tf, keep] : policy.keep_days) {
    const int needed = tf == lookback_tf ? lookback_days
                       : tf == mr::Timeframe::Hour ? 60
                       : tf == mr::Timeframe::Day  ? 365
                                                   : 5 * 365;
    if (keep && *keep < needed) {
      std::cerr << "warning: retention keeps " << *keep << " days of " << mr::to_string(tf)
                << " bars but the lookback is " << needed << "; skipping retention for it\n";
      keep = std::nullopt;
    }
  }
  std::size_t compacted = 0;
  for (auto tf : {mr::Timeframe::Hour, mr::Timeframe::Day, mr::Timeframe::Week})
    compacted += store.lake().compact(tf, 8);
  const auto removed = store.lake().apply_retention(policy, now_utc());
  std::cerr << "compacted " << compacted << " partitions, removed " << removed
            << " expired partitions\n";
}

fs::path sec_cache_path(const fs::path& data) { return data / "sectors" / "sec_sic.csv"; }

// Spec: S&P GICS sector, then SEC SIC sector, then ETF/Fund, then Unclassified; filled at load time.
void fill_sectors(mr::Universe& universe, const fs::path& data) {
  mr::apply_sector_fill(universe, mr::load_sec_cache(sec_cache_path(data)));
}

std::map<std::string, std::size_t> sector_counts(const mr::Universe& u) {
  std::map<std::string, std::size_t> out;
  for (const auto& s : u.nodes()) ++out[s.sector];
  return out;
}

double unclassified_pct(const mr::Universe& u) {
  if (u.nodes().empty()) return 0;
  std::size_t n = 0;
  for (const auto& s : u.nodes())
    if (s.sector == mr::kSectorUnclassified || s.sector.empty()) ++n;
  return 100.0 * static_cast<double>(n) / static_cast<double>(u.nodes().size());
}

// Resolves the universe as rank does (replay path), fetches SIC for every ticker, prints coverage.
int run_sync_sectors(const mr::CliArgs& args) {
  const fs::path dir = args.data / "universe";
  std::optional<fs::path> snapshot;
  if (args.universe != mr::UniverseSource::Sp500) snapshot = mr::latest_snapshot(dir);
  if (!snapshot && args.universe == mr::UniverseSource::Snapshot)
    throw std::runtime_error("no universe snapshot; run --mode alpaca first");
  mr::Universe universe =
      snapshot ? mr::load_snapshot(*snapshot, dir / "funds.csv") : sp500_universe(args.data);
  const double before = unclassified_pct(universe);
  mr::load_dotenv(".env");
  mr::SecSyncOptions opt;
  opt.now = now_utc();
  const auto tickers = universe.node_tickers();
  std::cerr << "syncing SEC SIC codes for " << tickers.size() << " tickers"
            << (snapshot ? " from " + snapshot->string() : std::string(" (sp500)")) << "...\n";
  const auto stats = mr::sync_sec_sectors(tickers, sec_cache_path(args.data), mr::make_sec_client, opt);
  fill_sectors(universe, args.data);
  std::printf("sec sync: %zu fetched, %zu fresh in cache, %zu without CIK, %zu failed (retried next run)\n",
              stats.fetched, stats.fresh, stats.no_cik, stats.failed);
  std::printf("%-26s %7s\n", "sector", "count");
  for (const auto& [sector, n] : sector_counts(universe)) std::printf("%-26s %7zu\n", sector.c_str(), n);
  std::printf("unclassified: %.1f%% before, %.1f%% after (%zu tickers)\n", before,
              unclassified_pct(universe), universe.nodes().size());
  return 0;
}

fs::path ensure_snapshot(const mr::CliArgs& args, const mr::AlpacaConfig& cfg,
                         mr::AlpacaClient& data_client, mr::BarStore& store,
                         const mr::PortfolioSpec& portfolio) {
  const fs::path dir = args.data / "universe";
  // Reuse a fresh (< 7 days) snapshot built for exactly the requested size.
  if (!args.refresh_universe)
    if (auto found = mr::find_snapshot(dir, args.universe_size, now_utc(), 7)) return *found;
  mr::AlpacaConfig trading_cfg = cfg;
  trading_cfg.host = cfg.trading_host;
  mr::AlpacaClient trading(trading_cfg);
  std::cerr << "fetching asset list from " << cfg.trading_host << "...\n";
  const auto assets =
      mr::parse_assets(trading.get("/v2/assets?status=active&asset_class=us_equity"));
  const mr::Universe sp = sp500_universe(args.data);
  mr::UniverseRules rules;
  for (const auto& t : sp.node_tickers()) rules.always_include.insert(t);
  for (const auto& h : portfolio.holdings)
    if (!sp.is_fund(h.ticker)) rules.always_include.insert(h.ticker);
  for (const auto& t : mr::read_ticker_list(dir / "include.csv")) rules.always_include.insert(t);
  rules.exclude = mr::read_ticker_list(dir / "exclude.csv");
  std::vector<mr::AssetInfo> candidates;
  for (const auto& a : assets)
    if (mr::passes_universe_rules(a, rules)) candidates.push_back(a);
  std::vector<std::string> symbols;
  for (const auto& a : candidates) symbols.push_back(a.symbol);
  const mr::TimePoint end = now_utc() - 16 * 60;
  store.load_range(symbols, mr::Timeframe::Day, end - 40 * 86400, end);
  std::cerr << assets.size() << " assets, " << candidates.size()
            << " candidates; fetching 40 days of daily bars to rank liquidity...\n";
  const auto stale = mr::sync_bars(data_client, store, symbols, mr::Timeframe::Day,
                                   end - 40 * 86400, end);
  if (!stale.empty()) std::cerr << stale.size() << " stale tickers during ranking\n";
  const auto ranked = mr::rank_by_liquidity(candidates, store, 20, args.universe_size);
  const fs::path path =
      dir / ("universe_" + today_string() + "_n" + std::to_string(args.universe_size) + ".csv");
  mr::write_universe_snapshot(path, ranked, sp);
  std::cerr << "wrote " << ranked.size() << "-ticker universe snapshot " << path.string() << "\n";
  return path;
}

// rank, ticker, sector, MarketRank (pi*N), heartbeat (delta log pi), hotness h, score+k.
void print_row(std::size_t rank, const mr::Security& s, double mr_score, double pulse, double h, double score) {
  char beat[16];
  if (std::isfinite(pulse)) std::snprintf(beat, sizeof beat, "%+10.5f", pulse);
  else std::snprintf(beat, sizeof beat, "%10s", "n/a");
  std::printf("%5zu  %-7s %-24.24s %10.4f %s %+10.4f %+9.4f\n", rank, s.ticker.c_str(), s.sector.c_str(), mr_score,
              beat, h, score);
}

int run_eval(const mr::CliArgs& args, const mr::Panel& panel, const mr::Universe& universe) {
  std::printf("evaluation: last %zu of %zu bars, nodes=%zu, timeframe=%s, threads=%d\n\n",
              args.eval_bars, panel.T(), panel.N(), std::string(mr::to_string(args.tf)).c_str(),
              omp_get_max_threads());
  std::printf("%-20s %6s %6s %6s %6s %9s %6s %9s %6s %9s %6s %9s %6s %9s\n", "config", "floor",
              "gini", "coher", "struct", "IC(score)", "t", "IC(h)", "t", "IC_oo", "t", "IC_h_oo", "t",
              "ms/frame");
  for (const auto& c : mr::evaluation_grid()) {
    const mr::EvalMetrics m = mr::evaluate(panel, universe.nodes(), c.params, args.eval_bars);
    std::printf("%-20s %6.3f %6.3f %6.3f %6.3f %+9.4f %+6.2f %+9.4f %+6.2f %+9.4f %+6.2f %+9.4f %+6.2f %9.1f\n",
                c.name.c_str(), m.floor_share, m.gini, m.sector_coherence, m.structure_gain, m.ic_mean,
                m.ic_t, m.ic_h_mean, m.ic_h_t, m.ic_oo_mean, m.ic_oo_t, m.ic_h_oo_mean, m.ic_h_oo_t,
                m.mean_frame_ms);
    std::fflush(stdout);
  }
  return 0;
}

int run_export_slice(const mr::CliArgs& args, const mr::Panel& panel, const mr::Universe& universe) {
  const mr::GraphSlice s = mr::export_slice(panel, args.params, args.export_slice);
  const auto j = mr::slice_json(s, universe.nodes(), std::string(mr::to_string(args.tf)), args.preset);
  if (args.slice_out.has_parent_path()) fs::create_directories(args.slice_out.parent_path());
  std::ofstream(args.slice_out) << j.dump(2) << "\n";
  std::printf("slice of %zu stocks at %s (%zu active) -> %s\n", s.nodes.size(), mr::format_rfc3339(s.t).c_str(),
              s.n_active, args.slice_out.string().c_str());
  std::printf("%-7s %-24s %12s %10s %10s\n", "ticker", "sector", "pi", "pi*N", "slice_pi");
  for (std::size_t k = 0; k < s.nodes.size(); ++k) {
    const auto& sec = universe.nodes()[s.nodes[k]];
    std::printf("%-7s %-24.24s %12.4e %10.4f %10.5f\n", sec.ticker.c_str(), sec.sector.c_str(), s.pi[k], s.mr[k],
                s.slice_pi[k]);
  }
  std::printf("%zu edges among the slice\n", s.edges.size());
  return 0;
}

int run_shock(const mr::CliArgs& args, const mr::Panel& panel, const mr::Universe& universe,
              const std::optional<mr::PortfolioSpec>& portfolio) {
  std::vector<mr::Shock> shocks;
  for (const auto& [ticker, size] : args.shocks) {
    const auto i = universe.index_of(ticker);
    if (!i) throw std::invalid_argument("--shock: unknown ticker " + ticker);
    shocks.push_back({*i, size});
  }
  const auto [base, shocked] = mr::run_with_shock(panel, args.params, shocks);  // throws if inactive
  const mr::ShockDelta d = mr::shock_response(base, shocked);
  const auto& nodes = universe.nodes();
  std::printf("mode=%s timeframe=%s nodes=%zu bars=%zu last=%s threads=%d\n", args.mode.c_str(),
              std::string(mr::to_string(args.tf)).c_str(), panel.N(), panel.T(),
              mr::format_rfc3339(base.t).c_str(), omp_get_max_threads());
  std::printf("params: %s\n", mr::describe(args.params).c_str());
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

int run_rank(const mr::CliArgs& args, const mr::Panel& panel, const mr::Universe& universe,
             const std::optional<mr::PortfolioSpec>& portfolio) {
  const auto t0 = std::chrono::steady_clock::now();
  const mr::Frame f = mr::run_panel_last(panel, args.params);
  const double total_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  std::printf("mode=%s timeframe=%s nodes=%zu bars=%zu last=%s threads=%d\n", args.mode.c_str(),
              std::string(mr::to_string(args.tf)).c_str(), panel.N(), panel.T(),
              mr::format_rfc3339(f.t).c_str(), omp_get_max_threads());
  std::printf("params: %s\n", mr::describe(args.params).c_str());
  std::printf("solver: %s in %d iterations (residual %.2e); last frame %.1f ms, run %.0f ms\n",
              f.solve.converged ? "converged" : "NOT converged", f.solve.iterations,
              f.solve.residual, f.compute_ms, total_ms);
  std::printf("edges: %zu slow, %zu fast; %.1f%% of active nodes at the teleport floor\n\n",
              f.P.col.size(), f.P_fast.col.size(), 100.0 * mr::floor_share(f, args.params.alpha));

  const auto& nodes = universe.nodes();
  const bool by_pi = args.rank_by == mr::RankBy::Pi;
  // Primary metric: MarketRank pi (default) or hotness h; descending, ties by lower index (as /api/top).
  const std::vector<std::size_t> order = mr::rank_order(f, args.rank_by);
  const std::size_t n_active = order.size();
  const std::size_t inactive = panel.N() - n_active;
  std::vector<std::size_t> rank_of(panel.N(), 0);
  for (std::size_t r = 0; r < order.size(); ++r) rank_of[order[r]] = r + 1;
  const auto& score = f.forecasts.front().score;
  const std::size_t top = std::min(args.top, order.size());
  if (inactive > 0) std::printf("%zu inactive (stale or below liquidity floor)\n\n", inactive);
  std::printf("MarketRank = pi*N (1 = average, N = %zu active); heartbeat = delta log(pi*N) vs the previous bar\n\n",
              n_active);
  std::printf("%s\n", by_pi ? "TOP MARKETRANK" : "HILLS (money accumulating)");
  std::printf("%5s  %-7s %-24s %10s %10s %10s %9s\n", "rank", "ticker", "sector", "MarketRank", "heartbeat", "hotness",
              ("score+" + std::to_string(f.forecasts.front().k)).c_str());
  auto row = [&](std::size_t i) {
    print_row(rank_of[i], nodes[i], mr::market_rank_score(f.pi[i], n_active), f.pulse[i], f.h[i], score[i]);
  };
  for (std::size_t r = 0; r < top; ++r) row(order[r]);
  std::printf("\n%s\n", by_pi ? "BOTTOM MARKETRANK" : "VALLEYS (money draining)");
  const mr::BottomSection bottom = mr::bottom_section(f, order, args.rank_by, top);
  if (bottom.floor_count > 0) std::printf("%s\n", mr::floor_line(bottom).c_str());
  for (std::size_t i : bottom.rows) row(i);
  if (portfolio) {
    std::printf("\nPORTFOLIO HOLDINGS\n");
    for (const auto& hld : portfolio->holdings) {
      if (auto i = universe.index_of(hld.ticker); i && !f.active[*i]) {
        std::printf("  %-6s %5.1f%%  (no data)\n", hld.ticker.c_str(), hld.weight * 100);
      } else if (i) {
        std::printf("  %-6s %5.1f%%  MarketRank %8.4f  hotness %+9.4f  score+%d %+9.4f\n", hld.ticker.c_str(),
                    hld.weight * 100, mr::market_rank_score(f.pi[*i], n_active), f.h[*i], f.forecasts.front().k,
                    score[*i]);
      } else {
        std::printf("  %-6s %5.1f%%  (fund: look-through hotness arrives in milestone 3)\n",
                    hld.ticker.c_str(), hld.weight * 100);
      }
    }
  }
  return 0;
}

// --walkforward: the base mix is data/portfolio.json (the spec's mix if it has no holdings); holdings missing
// from the panel are dropped with a warning (their weight stays in cash).
int run_walkforward_cli(const mr::CliArgs& args, const mr::Panel& panel,
                        const std::optional<mr::PortfolioSpec>& portfolio) {
  mr::WalkForwardParams p = mr::walkforward_params(args);
  std::vector<mr::BaseWeight> base;
  if (portfolio)
    for (const auto& h : portfolio->holdings) base.push_back({h.ticker, h.weight});
  if (base.empty()) base = mr::default_base_mix();
  for (const auto& b : base) {
    if (std::find(panel.tickers.begin(), panel.tickers.end(), b.ticker) != panel.tickers.end())
      p.bt.base.push_back(b);
    else
      std::cerr << "warning: base holding " << b.ticker << " is not in the panel; its " << b.weight * 100
                << "% stays in cash\n";
  }
  const std::string hash = mr::params_hash(p);
  std::string run_id = args.wf_run_id;
  if (run_id.empty()) {
    std::string ts = mr::format_rfc3339(now_utc());
    ts.erase(std::remove_if(ts.begin(), ts.end(), [](char c) { return c == '-' || c == ':'; }), ts.end());
    run_id = ts + "-" + hash;
  }
  const fs::path out = args.wf_out.empty() ? args.data / "walkforward" : args.wf_out;
  // Fail before the long pass (write_report repeats the registry check under its lock).
  if (!p.largecap_run.empty() && !fs::exists(out / p.largecap_run / "results.json"))
    throw std::runtime_error("--wf-largecap-run: no " + (out / p.largecap_run / "results.json").string());
  mr::check_registry_conflict(out, run_id, hash);
  std::printf("walk-forward %s: nodes=%zu bars=%zu threads=%d\nparams: %s\n", run_id.c_str(), panel.N(), panel.T(),
              omp_get_max_threads(), mr::describe(p).c_str());
  std::fflush(stdout);
  const auto t0 = std::chrono::steady_clock::now();
  const mr::WalkForwardResult r = mr::run_walkforward(panel, p);
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  const fs::path dir = mr::write_report(r, p, panel, out, run_id);
  std::printf("%zu rebalances, %zu curves in %.1f s -> %s\n", r.dates.size(), r.curves.size(), secs,
              (dir / "report.md").string().c_str());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const mr::CliArgs args = mr::parse_cli(std::vector<std::string>(argv + 1, argv + argc));
    for (const auto& w : args.warnings) std::cerr << "warning: " << w << "\n";
    if (args.help) {
      std::cout << mr::cli_usage();
      return 0;
    }
    if (args.threads > 0) omp_set_num_threads(args.threads);
    if (args.migrate_cache) {
      mr::BarStore lake_store(args.data / "lake");
      const auto n = mr::migrate_csv_cache(args.migrate_from, lake_store);
      std::cout << "migrated " << n << " series from " << args.migrate_from.string() << " into "
                << (args.data / "lake").string() << "\n";
      return 0;
    }
    if (args.sync_sectors) return run_sync_sectors(args);
    if (!args.wf_rereport.empty()) {
      const fs::path out = args.wf_out.empty() ? args.data / "walkforward" : args.wf_out;
      const fs::path dir = mr::rereport(out, args.wf_rereport);
      std::printf("re-rendered %s\n", (dir / "report.md").string().c_str());
      return 0;
    }
    if (args.maintain) {
      mr::BarStore lake_store(args.data / "lake");
      maintain_lake(lake_store, args.data, args.lookback_days, args.tf);
      return 0;
    }
    const auto [window_start, end] = mr::data_window(args, now_utc());
    mr::Universe universe;
    std::optional<mr::PortfolioSpec> portfolio;
    mr::Panel panel;
    {
      // The bar store (and the lake's DuckDB lock) lives only until the panel is built, so a long-running
      // --serve does not block other replay runs or keep a second copy of every bar in memory.
      mr::BarStore store(args.data / "lake");
      if (args.mode == "synthetic") {
        mr::SyntheticConfig cfg;
        cfg.tf = args.tf;
        universe = mr::Universe::from_securities(mr::generate_synthetic(cfg, store));
      } else {
        portfolio = mr::load_portfolio(args.data / "portfolio.json");
        const fs::path dir = args.data / "universe";
        std::optional<mr::AlpacaConfig> cfg;
        std::optional<mr::AlpacaClient> client;
        if (args.mode == "alpaca") {
          mr::load_dotenv(".env");
          cfg = mr::alpaca_config_from_env();
          if (!cfg) throw std::runtime_error("APCA_API_KEY_ID / APCA_API_SECRET_KEY not set (.env)");
          client.emplace(*cfg);
        }
        std::optional<fs::path> snapshot;
        if (args.universe != mr::UniverseSource::Sp500) {
          snapshot = args.mode == "alpaca" ? ensure_snapshot(args, *cfg, *client, store, *portfolio)
                                           : mr::latest_snapshot(dir);
          if (!snapshot && args.universe == mr::UniverseSource::Snapshot)
            throw std::runtime_error("no universe snapshot; run --mode alpaca first");
        }
        universe = snapshot ? mr::load_snapshot(*snapshot, dir / "funds.csv")
                            : sp500_universe(args.data);
        fill_sectors(universe, args.data);
        universe.add_extras(*portfolio);
        store.load_range(universe.price_tickers(), args.tf, window_start, end);
        if (client) {
          const mr::TimePoint start = window_start;
          std::cerr << (args.refetch_full ? "refetching full history of " : "syncing ")
                    << universe.price_tickers().size() << " tickers (" << mr::to_string(args.tf) << ") from "
                    << mr::format_rfc3339(start) << " (or earlier stored history)...\n";
          const auto stale = mr::sync_bars(*client, store, universe.price_tickers(), args.tf, start, end,
                                           args.refetch_full);
          if (!stale.empty()) std::cerr << stale.size() << " stale tickers\n";
          store.flush();
          maintain_lake(store, args.data, args.lookback_days, args.tf);
        }
      }
      panel = mr::build_panel(store, universe.node_tickers(), args.tf, window_start, end);
    }
    if (panel.T() < 2) throw std::runtime_error("not enough cached bars; run with --mode alpaca first");
    if (args.walkforward) return run_walkforward_cli(args, panel, portfolio);
    if (args.cluster_persistence) {
      mr::ClusterPersistenceOptions opt;
      opt.stride = args.cp_stride;
      opt.out_dir = args.cp_out.empty() ? args.data / "analysis" : args.cp_out;
      std::printf("params: %s\n", mr::describe(args.params).c_str());
      mr::run_cluster_persistence(panel, universe.nodes(), args.params, opt);
      return 0;
    }
    if (args.serve) {
      if (!mr::is_loopback_host(args.host))
        std::cerr << "warning: --host " << args.host
                  << " is not a loopback address: the server (which has no authentication) is reachable from the "
                     "network\n";
      mr::LandscapeParams land;
      // Fixed serve defaults (the UI has no knobs): flux territories, the CVT smoother, subdivision 1 (all
      // LandscapeParams{} defaults) and, under the marketrank preset, the height log(pi / size share).
      land.value = mr::default_landscape_value(args.preset);
      // Pin the portfolio holdings: the smoothed surface passes exactly through each one (holdings missing from
      // the universe are ignored).
      if (portfolio) {
        const auto& nodes = universe.nodes();
        std::map<std::string, std::uint32_t> idx;
        for (std::size_t i = 0; i < nodes.size(); ++i) idx.emplace(nodes[i].ticker, static_cast<std::uint32_t>(i));
        for (const auto& hld : portfolio->holdings)
          if (auto it = idx.find(hld.ticker); it != idx.end()) land.pinned.push_back(it->second);
      }
      mr::FrameStore frames(std::move(panel), universe.nodes(), args.params, land);
      frames.start();
      mr::FluxServer server(frames, portfolio, args.mode + " " + std::string(mr::to_string(args.tf)));
      if (!fs::is_directory(args.web))
        std::cerr << "warning: web root '" << args.web.string() << "' not found; static files will 404\n";
      const int port = server.bind({args.host, args.port, args.web});
      std::cerr << "MarketRank Fluxscape serving http://" << args.host << ":" << port << "  (Ctrl-C to stop)\n";
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
        std::cerr << "marketrank: server failed to listen\n";
        return 1;
      }
      return 0;
    }
    if (args.export_slice > 0) return run_export_slice(args, panel, universe);
    if (args.eval) return run_eval(args, panel, universe);
    if (!args.shocks.empty()) return run_shock(args, panel, universe, portfolio);
    return run_rank(args, panel, universe, portfolio);
  } catch (const std::exception& e) {
    std::cerr << "marketrank: " << e.what() << "\n";
    return 1;
  }
}
