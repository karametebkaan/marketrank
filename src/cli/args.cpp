#include "cli/args.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace mr {
namespace {

std::size_t to_size(const std::string& flag, const std::string& v) {
  std::size_t pos = 0;
  long long x = 0;
  try {
    x = std::stoll(v, &pos);
  } catch (const std::exception&) {
    throw std::invalid_argument(flag + " expects a non-negative integer, got '" + v + "'");
  }
  if (pos != v.size() || x < 0)
    throw std::invalid_argument(flag + " expects a non-negative integer, got '" + v + "'");
  return static_cast<std::size_t>(x);
}

double to_double(const std::string& flag, const std::string& v) {
  std::size_t pos = 0;
  double x = 0;
  try {
    x = std::stod(v, &pos);
  } catch (const std::exception&) {
    throw std::invalid_argument(flag + " expects a number, got '" + v + "'");
  }
  if (pos != v.size()) throw std::invalid_argument(flag + " expects a number, got '" + v + "'");
  return x;
}

std::pair<std::string, double> to_shock(const std::string& flag, const std::string& v) {
  const auto colon = v.rfind(':');
  if (colon == std::string::npos || colon == 0)
    throw std::invalid_argument(flag + " expects TICKER:SIZE, got '" + v + "'");
  const double size = to_double(flag, v.substr(colon + 1));
  if (!std::isfinite(size)) throw std::invalid_argument(flag + " size must be finite, got '" + v + "'");
  return {v.substr(0, colon), size};
}

}  // namespace

CliArgs parse_cli(const std::vector<std::string>& args) {
  CliArgs a;
  // Presets apply first, regardless of position, so later model flags override them. The default (rank and
  // --serve alike) is the MarketRank concept model.
  auto has = [&](const char* f) { return std::find(args.begin(), args.end(), f) != args.end(); };
  const bool legacy = has("--legacy"), money = has("--money-flow"), market = has("--marketrank");
  if (int(legacy) + int(money) + int(market) > 1)
    throw std::invalid_argument("--marketrank, --money-flow and --legacy are mutually exclusive");
  if (legacy) a.params = CoreParams::legacy(), a.preset = "legacy";
  if (money) a.params = CoreParams::money_flow(), a.preset = "money-flow";
  if (market) a.params = CoreParams::market_rank(), a.preset = "marketrank";
  for (std::size_t i = 0; i < args.size(); ++i) {
    const std::string& flag = args[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= args.size()) throw std::invalid_argument("missing value for " + flag);
      return args[++i];
    };
    if (flag == "--mode") a.mode = value();
    else if (flag == "--timeframe") a.tf = parse_timeframe(value());
    else if (flag == "--lookback-days") a.lookback_days = static_cast<int>(to_size(flag, value()));
    else if (flag == "--top") a.top = to_size(flag, value());
    else if (flag == "--data") a.data = value();
    else if (flag == "--universe") {
      const std::string v = value();
      if (v == "auto") a.universe = UniverseSource::Auto;
      else if (v == "sp500") a.universe = UniverseSource::Sp500;
      else if (v == "snapshot") a.universe = UniverseSource::Snapshot;
      else throw std::invalid_argument("--universe must be auto, sp500 or snapshot");
    } else if (flag == "--universe-size") a.universe_size = to_size(flag, value());
    else if (flag == "--refresh-universe") a.refresh_universe = true;
    else if (flag == "--eval") a.eval = true;
    else if (flag == "--eval-bars") a.eval_bars = to_size(flag, value());
    else if (flag == "--threads") a.threads = static_cast<int>(to_size(flag, value()));
    else if (flag == "--migrate-cache") {
      a.migrate_cache = true;
      if (i + 1 < args.size() && args[i + 1].rfind("--", 0) != 0) a.migrate_from = args[++i];
    } else if (flag == "--maintain") a.maintain = true;
    else if (flag == "--refetch-full") a.refetch_full = true;
    else if (flag == "--sync-sectors") a.sync_sectors = true;
    else if (flag == "--serve") a.serve = true;
    else if (flag == "--port") {
      const std::size_t p = to_size(flag, value());
      if (p == 0 || p > 65535) throw std::invalid_argument("--port must be between 1 and 65535");
      a.port = static_cast<int>(p);
    } else if (flag == "--host") a.host = value();
    else if (flag == "--web") a.web = value();
    else if (flag == "--legacy" || flag == "--money-flow" || flag == "--marketrank") continue;
    else if (flag == "--rank-by") {
      const std::string v = value();
      if (v == "pi") a.rank_by = RankBy::Pi;
      else if (v == "hotness") a.rank_by = RankBy::Hotness;
      else throw std::invalid_argument("--rank-by must be pi or hotness");
    }
    else if (flag == "--pressure") a.params.pressure = parse_pressure_mode(value());
    else if (flag == "--lift") a.params.transition.lift = parse_lift_mode(value());
    else if (flag == "--k-out") a.params.transition.k_out = to_size(flag, value());
    else if (flag == "--k-in") a.params.transition.k_in = to_size(flag, value());
    else if (flag == "--retention") a.params.transition.retention = to_double(flag, value());
    else if (flag == "--h-ref") a.params.h_ref = parse_hot_ref(value());
    else if (flag == "--min-dollar-volume") a.params.min_dollar_volume = to_double(flag, value());
    else if (flag == "--max-volume-ratio") a.params.max_volume_ratio = to_double(flag, value());
    else if (flag == "--vol-scale") a.params.vol_scale = true;
    else if (flag == "--vol-window") a.params.vol_window = to_size(flag, value());
    else if (flag == "--lambda") a.params.flux.lambda = to_double(flag, value());
    else if (flag == "--shock") a.shocks.push_back(to_shock(flag, value()));
    else if (flag == "--help" || flag == "-h") a.help = true;
    else throw std::invalid_argument("unknown flag " + flag);
  }
  if (a.mode != "synthetic" && a.mode != "replay" && a.mode != "alpaca")
    throw std::invalid_argument("--mode must be synthetic, replay or alpaca");
  if (a.refetch_full && a.mode != "alpaca") throw std::invalid_argument("--refetch-full needs --mode alpaca");
  if (a.lookback_days < 0)
    a.lookback_days = a.tf == Timeframe::Hour ? 60 : a.tf == Timeframe::Day ? 365 : 5 * 365;
  a.params.validate();
  return a;
}

std::pair<TimePoint, TimePoint> data_window(const CliArgs& args, TimePoint now) {
  if (args.mode == "synthetic")
    return {std::numeric_limits<TimePoint>::min(), std::numeric_limits<TimePoint>::max()};
  const TimePoint end = args.mode == "alpaca" ? now - 16 * 60 : now;
  return {end - static_cast<TimePoint>(args.lookback_days) * 86400, end};
}

std::string cli_usage() {
  return "MarketRank - stock ranking by Markov steady state of money flows\n"
         "usage: marketrank [--mode synthetic|replay|alpaca] [--timeframe 1h|1d|1w]\n"
         "                 [--lookback-days N] [--top N] [--data DIR] [--threads N]\n"
         "                 [--universe auto|sp500|snapshot] [--universe-size N] [--refresh-universe]\n"
         "                 [--eval] [--eval-bars N]\n"
         "                 [--marketrank | --money-flow | --legacy]   (model preset; default --marketrank, for --serve too)\n"
         "                 [--rank-by pi|hotness]   (primary table: MarketRank pi*N (default) or hotness h)\n"
         "                 [--pressure dollar|sqrt|relative] [--lift off|excess|ratio]\n"
         "                 [--k-out N] [--k-in N] [--retention X] [--h-ref uniform|size|longrun|netflow]\n"
         "                 [--lambda X] [--min-dollar-volume X] [--max-volume-ratio X]\n"
         "                 [--vol-scale] [--vol-window N]\n"
         "                 [--shock TICKER:SIZE ...]   (extra SIZE% return at normal volume on the last bar)\n"
         "                 [--serve [--port N (8765)] [--host H] [--web DIR]]   (REST + SSE server)\n"
         "                 [--migrate-cache [DIR]] [--maintain]\n"
         "                 [--sync-sectors]   (fetch SEC EDGAR SIC sectors for the universe snapshot into\n"
         "                                   data/sectors/sec_sic.csv; needs SEC_USER_AGENT in .env, no Alpaca keys;\n"
         "                                   uses the newest snapshot (--universe-size is ignored); run it on its own, then rank/eval)\n"
         "                 [--refetch-full]   (alpaca: one-time refetch of every ticker's full stored history,\n"
         "                                   replacing old-basis bars; failed tickers stay untouched)\n";
}

std::string describe(const CoreParams& p) {
  std::ostringstream s;
  s << "pressure=" << to_string(p.pressure) << " lift=" << to_string(p.transition.lift)
    << " k_out=" << p.transition.k_out << " k_in=" << p.transition.k_in
    << " retention=" << p.transition.retention << " h_ref=" << to_string(p.h_ref)
    << " lambda=" << p.flux.lambda << " alpha=" << p.alpha
    << " min_dv=" << p.min_dollar_volume << " max_vr=" << p.max_volume_ratio
    << " vol_scale=" << (p.vol_scale ? 1 : 0) << " hl_slow=" << p.halflife_slow
    << " hl_fast=" << p.halflife_fast;
  return s.str();
}

}  // namespace mr
