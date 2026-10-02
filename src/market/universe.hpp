#pragma once
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace mr {

struct Security {
  std::string ticker, name, sector;
};

struct Fund {
  std::string ticker, tracks;
};

struct Holding {
  std::string ticker;
  double weight;
};

struct PortfolioSpec {
  double initial_cash = 0;
  std::string inception;
  std::vector<Holding> holdings;
};

PortfolioSpec load_portfolio(const std::filesystem::path& path);
std::vector<Fund> load_funds(const std::filesystem::path& funds_csv);

class Universe {
 public:
  static Universe load(const std::filesystem::path& sp500_csv,
                       const std::filesystem::path& funds_csv);
  static Universe from_securities(std::vector<Security> securities, std::vector<Fund> funds = {});

  // Adds held single stocks that are not yet nodes (sector "Extra"). Funds are skipped.
  void add_extras(const PortfolioSpec& portfolio);

  const std::vector<Security>& nodes() const { return nodes_; }
  void set_sector(std::size_t i, std::string sector) { nodes_.at(i).sector = std::move(sector); }
  const std::vector<Fund>& funds() const { return funds_; }
  std::optional<std::size_t> index_of(std::string_view ticker) const;
  bool is_fund(std::string_view ticker) const;
  std::vector<std::string> node_tickers() const;
  std::vector<std::string> price_tickers() const;

 private:
  void add_node(Security s);
  std::vector<Security> nodes_;
  std::vector<Fund> funds_;
  std::unordered_map<std::string, std::size_t> index_;
};

}  // namespace mr
