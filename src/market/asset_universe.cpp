#include "market/asset_universe.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <regex>
#include <stdexcept>

#include "core/csv.hpp"

namespace fx {

std::vector<AssetInfo> parse_assets(const std::string& json) {
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(json);
  } catch (const nlohmann::json::exception&) {
    throw std::runtime_error("assets response is not valid JSON");
  }
  if (!j.is_array()) throw std::runtime_error("assets response is not a JSON array");
  std::vector<AssetInfo> out;
  out.reserve(j.size());
  for (const auto& a : j) {
    if (!a.is_object() || !a.contains("symbol") || !a["symbol"].is_string()) continue;
    AssetInfo info;
    info.symbol = a["symbol"].get<std::string>();
    if (a.contains("name") && a["name"].is_string()) info.name = a["name"].get<std::string>();
    if (a.contains("exchange") && a["exchange"].is_string())
      info.exchange = a["exchange"].get<std::string>();
    info.tradable = a.contains("tradable") && a["tradable"].is_boolean() && a["tradable"].get<bool>();
    out.push_back(std::move(info));
  }
  return out;
}

bool passes_universe_rules(const AssetInfo& a, const UniverseRules& rules) {
  if (!a.tradable) return false;
  if (rules.exclude.count(a.symbol)) return false;
  if (rules.always_include.count(a.symbol)) return true;
  static const std::set<std::string> kExchanges = {"NYSE", "NASDAQ", "ARCA",
                                                   "NYSEARCA", "AMEX", "BATS"};
  if (!kExchanges.count(a.exchange)) return false;
  static const std::regex kExcluded(
      R"((warrant|\bunits?\b|\brights?\b|\betf\b|\betn\b|ishares|spdr|proshares|direxion|\bfund\b|\bindex\b))",
      std::regex::icase);
  return !std::regex_search(a.name, kExcluded);
}

std::vector<RankedAsset> rank_by_liquidity(const std::vector<AssetInfo>& assets,
                                           const BarStore& store, std::size_t window,
                                           std::size_t top_n) {
  std::vector<RankedAsset> ranked;
  std::vector<double> dv;
  for (const auto& a : assets) {
    const auto& bars = store.bars(a.symbol, Timeframe::Day);
    if (bars.empty()) continue;
    dv.clear();
    const std::size_t from = bars.size() > window ? bars.size() - window : 0;
    for (std::size_t k = from; k < bars.size(); ++k)
      dv.push_back(bars[k].v * (bars[k].vw > 0 ? bars[k].vw : bars[k].c));
    std::sort(dv.begin(), dv.end());
    const std::size_t m = dv.size();
    const double median = m % 2 ? dv[m / 2] : 0.5 * (dv[m / 2 - 1] + dv[m / 2]);
    ranked.push_back({a, median});
  }
  std::sort(ranked.begin(), ranked.end(), [](const RankedAsset& x, const RankedAsset& y) {
    return x.median_dollar_volume > y.median_dollar_volume ||
           (x.median_dollar_volume == y.median_dollar_volume && x.asset.symbol < y.asset.symbol);
  });
  if (ranked.size() > top_n) ranked.resize(top_n);
  return ranked;
}

namespace {

std::string csv_field(const std::string& s) {
  if (s.find_first_of(",\"\n") == std::string::npos) return s;
  std::string q = "\"";
  for (char ch : s) {
    if (ch == '"') q += '"';
    q += ch;
  }
  return q + "\"";
}

const std::regex& snapshot_name() {
  static const std::regex re(R"(universe_(\d{4}-\d{2}-\d{2})\.csv)");
  return re;
}

}  // namespace

void write_universe_snapshot(const std::filesystem::path& path,
                             const std::vector<RankedAsset>& ranked, const Universe& sp500) {
  if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
  const std::string tmp = path.string() + ".tmp";
  {
    std::ofstream out(tmp);
    if (!out) throw std::runtime_error("cannot write " + tmp);
    out << "ticker,name,sector,exchange,median_dollar_volume\n" << std::setprecision(15);
    for (const auto& r : ranked) {
      const auto idx = sp500.index_of(r.asset.symbol);
      const std::string sector = idx ? sp500.nodes()[*idx].sector : "Unclassified";
      out << csv_field(r.asset.symbol) << ',' << csv_field(r.asset.name) << ','
          << csv_field(sector) << ',' << csv_field(r.asset.exchange) << ','
          << r.median_dollar_volume << '\n';
    }
    if (!out) throw std::runtime_error("write failed for " + tmp);
  }
  std::filesystem::rename(tmp, path);
}

std::optional<std::string> snapshot_date(const std::filesystem::path& path) {
  std::smatch m;
  const std::string name = path.filename().string();
  if (std::regex_match(name, m, snapshot_name())) return m[1].str();
  return std::nullopt;
}

std::optional<std::filesystem::path> latest_snapshot(const std::filesystem::path& dir) {
  if (!std::filesystem::is_directory(dir)) return std::nullopt;
  std::optional<std::filesystem::path> best;
  for (const auto& entry : std::filesystem::directory_iterator(dir)) {
    if (!entry.is_regular_file() || !snapshot_date(entry.path())) continue;
    if (!best || entry.path().filename() > best->filename()) best = entry.path();
  }
  return best;
}

Universe load_snapshot(const std::filesystem::path& path, const std::filesystem::path& funds_csv) {
  const CsvRows rows = read_csv_file(path);
  std::vector<Security> secs;
  for (std::size_t r = 1; r < rows.size(); ++r)
    if (rows[r].size() >= 3) secs.push_back({rows[r][0], rows[r][1], rows[r][2]});
  return Universe::from_securities(std::move(secs), load_funds(funds_csv));
}

std::set<std::string> read_ticker_list(const std::filesystem::path& path) {
  std::set<std::string> out;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    const auto first = line.find_first_not_of(" \t\r");
    if (first == std::string::npos || line[first] == '#') continue;
    const auto last = line.find_last_not_of(" \t\r");
    out.insert(line.substr(first, last - first + 1));
  }
  return out;
}

}  // namespace fx
