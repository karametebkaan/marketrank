#include "market/universe.hpp"

#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

#include "core/csv.hpp"

namespace mr {

PortfolioSpec load_portfolio(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path.string());
  const auto j = nlohmann::json::parse(in);
  PortfolioSpec p;
  p.initial_cash = j.at("initial_cash").get<double>();
  p.inception = j.at("inception").get<std::string>();
  double sum = 0;
  for (const auto& h : j.at("holdings")) {
    p.holdings.push_back({h.at("ticker").get<std::string>(), h.at("weight").get<double>()});
    sum += p.holdings.back().weight;
  }
  if (std::abs(sum - 1.0) > 1e-6) {
    throw std::runtime_error("portfolio weights sum to " + std::to_string(sum) + ", expected 1");
  }
  return p;
}

std::vector<Fund> load_funds(const std::filesystem::path& funds_csv) {
  std::vector<Fund> funds;
  const CsvRows fu = read_csv_file(funds_csv);
  for (std::size_t r = 1; r < fu.size(); ++r)
    if (fu[r].size() >= 2) funds.push_back({fu[r][0], fu[r][1]});
  return funds;
}

Universe Universe::load(const std::filesystem::path& sp500_csv,
                        const std::filesystem::path& funds_csv) {
  Universe u;
  const CsvRows sp = read_csv_file(sp500_csv);
  for (std::size_t r = 1; r < sp.size(); ++r) {
    if (sp[r].size() < 3) continue;
    u.add_node({sp[r][0], sp[r][1], sp[r][2]});
  }
  u.funds_ = load_funds(funds_csv);
  return u;
}

Universe Universe::from_securities(std::vector<Security> securities, std::vector<Fund> funds) {
  Universe u;
  u.funds_ = std::move(funds);
  for (auto& s : securities) u.add_node(std::move(s));
  return u;
}

void Universe::add_node(Security s) {
  if (index_.count(s.ticker)) return;
  index_[s.ticker] = nodes_.size();
  nodes_.push_back(std::move(s));
}

void Universe::add_extras(const PortfolioSpec& portfolio) {
  for (const Holding& h : portfolio.holdings) {
    if (is_fund(h.ticker)) continue;
    add_node({h.ticker, h.ticker, "Extra"});
  }
}

std::optional<std::size_t> Universe::index_of(std::string_view ticker) const {
  auto it = index_.find(std::string(ticker));
  if (it == index_.end()) return std::nullopt;
  return it->second;
}

bool Universe::is_fund(std::string_view ticker) const {
  for (const Fund& f : funds_)
    if (f.ticker == ticker) return true;
  return false;
}

std::vector<std::string> Universe::node_tickers() const {
  std::vector<std::string> out;
  out.reserve(nodes_.size());
  for (const auto& s : nodes_) out.push_back(s.ticker);
  return out;
}

std::vector<std::string> Universe::price_tickers() const {
  auto out = node_tickers();
  for (const auto& f : funds_) {
    if (!index_.count(f.ticker)) out.push_back(f.ticker);
  }
  return out;
}

}  // namespace mr
