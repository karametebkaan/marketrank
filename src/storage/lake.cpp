#include "storage/lake.hpp"

#include <fcntl.h>
#include <omp.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>

#include "core/time.hpp"
#include "duckdb.hpp"

namespace fx {
namespace fs = std::filesystem;

namespace {

std::string tf_dir_name(Timeframe tf) { return "tf=" + std::string(to_string(tf)); }

std::string sql_str(const std::string& s) {
  std::string q = "'";
  for (char ch : s) {
    if (ch == '\'') q += '\'';
    q += ch;
  }
  return q + "'";
}

std::string random_id() {
  static thread_local std::mt19937_64 gen(std::random_device{}());
  std::ostringstream s;
  s << std::hex << gen() << gen();
  return s.str();
}

std::vector<fs::path> parquet_files(const fs::path& dir) {
  std::vector<fs::path> out;
  if (!fs::is_directory(dir)) return out;
  for (const auto& e : fs::recursive_directory_iterator(dir))
    if (e.is_regular_file() && e.path().extension() == ".parquet") out.push_back(e.path());
  std::sort(out.begin(), out.end());
  return out;
}

bool has_parquet(const fs::path& dir) {
  if (!fs::is_directory(dir)) return false;
  for (const auto& e : fs::recursive_directory_iterator(dir))
    if (e.is_regular_file() && e.path().extension() == ".parquet") return true;
  return false;
}

int year_of(TimePoint t) { return civil_from_days(floor_div(t, 86400)).y; }

// "year=2026" -> 2026; returns -1 if the name doesn't match.
long parse_kv(const std::string& name, const std::string& key) {
  if (name.rfind(key + "=", 0) != 0) return -1;
  try {
    return std::stol(name.substr(key.size() + 1));
  } catch (const std::exception&) {
    return -1;
  }
}

}  // namespace

void fsync_path(const fs::path& p) {
  const int fd = ::open(p.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) throw std::runtime_error("lake: cannot open " + p.string() + " for fsync: " + std::strerror(errno));
  if (::fsync(fd) != 0) {
    const int err = errno;
    ::close(fd);
    throw std::runtime_error("lake: fsync failed for " + p.string() + ": " + std::strerror(err));
  }
  ::close(fd);
}

RetentionPolicy RetentionPolicy::defaults() {
  RetentionPolicy p;
  p.keep_days[Timeframe::Hour] = 730;
  p.keep_days[Timeframe::Day] = std::nullopt;
  p.keep_days[Timeframe::Week] = std::nullopt;
  return p;
}

RetentionPolicy RetentionPolicy::load(const fs::path& json) {
  RetentionPolicy p = defaults();
  std::ifstream in(json);
  if (!in) return p;
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(in);
  } catch (const nlohmann::json::exception& e) {
    throw std::runtime_error("retention policy " + json.string() + " is not valid JSON");
  }
  if (!j.is_object()) throw std::runtime_error("retention policy must be a JSON object");
  for (auto& [key, value] : p.keep_days) {
    const std::string k(to_string(key));
    if (!j.contains(k)) {
      value = std::nullopt;
      continue;
    }
    if (j[k].is_null()) value = std::nullopt;
    else if (j[k].is_number_integer() && j[k].get<int>() > 0) value = j[k].get<int>();
    else throw std::runtime_error("retention for " + k + " must be a positive integer or null");
  }
  return p;
}

struct Lake::Impl {
  fs::path root;
  duckdb::DuckDB db;
  duckdb::Connection con;

  explicit Impl(fs::path r) : root(std::move(r)), db(open_path(root)), con(db) {}

  static std::string open_path(const fs::path& root) {
    fs::create_directories(root / "bars");
    return (root / "catalog.duckdb").string();
  }

  duckdb::unique_ptr<duckdb::MaterializedQueryResult> q(const std::string& sql) {
    auto r = con.Query(sql);
    if (r->HasError()) throw std::runtime_error("lake SQL failed: " + r->GetError());
    return r;
  }

  void set_want(const std::vector<std::string>& tickers) {
    q("DELETE FROM want");
    duckdb::Appender app(con, "want");
    for (const auto& t : tickers) {
      app.BeginRow();
      app.Append<duckdb::string_t>(duckdb::string_t(t.data(), static_cast<uint32_t>(t.size())));
      app.EndRow();
    }
    app.Close();
  }

  // Moves every unreadable Parquet file of tf to _quarantine/ and clears that tf's coverage (the bad
  // file's tickers are unknown), so the next sync back-fills. Returns the number of files moved.
  std::size_t quarantine(Timeframe tf) {
    std::size_t moved = 0;
    for (const auto& f : parquet_files(root / "bars" / tf_dir_name(tf))) {
      auto r = con.Query("SELECT count(*) FROM read_parquet(" + sql_str(f.string()) + ")");
      if (!r->HasError()) continue;
      const fs::path dest = root / "_quarantine" / fs::relative(f, root);
      fs::create_directories(dest.parent_path());
      fs::rename(f, dest);
      std::cerr << "lake: quarantined " << f.string() << ": " << r->GetError() << "\n";
      ++moved;
    }
    if (moved > 0) q("DELETE FROM coverage WHERE tf = " + sql_str(std::string(to_string(tf))));
    return moved;
  }

  // Moves pending rows into Parquet partitions, then applies pending coverage.
  void publish() {
    const auto n = q("SELECT count(*) FROM pending")->GetValue(0, 0).GetValue<int64_t>();
    if (n > 0) {
      fs::create_directories(root / "_staging");
      const fs::path staging = root / "_staging" / random_id();
      q("COPY (SELECT ticker, t, o, h, l, c, v, vw, seq, tf, year, month FROM pending) TO " +
        sql_str(staging.string()) +
        " (FORMAT parquet, PARTITION_BY (tf, year, month), FILENAME_PATTERN 'part-{uuid}')");
      std::vector<fs::path> moved;
      for (const auto& f : parquet_files(staging)) {
        const fs::path dest = root / "bars" / fs::relative(f, staging);
        fs::create_directories(dest.parent_path());
        fs::rename(f, dest);
        moved.push_back(dest);
      }
      fs::remove_all(staging);
      // Make the bars durable before the coverage that depends on them is committed (contract 3).
      std::set<fs::path> dirs;
      for (const auto& d : moved) {
        fsync_path(d);
        dirs.insert(d.parent_path());
      }
      for (const auto& d : dirs) fsync_path(d);
    }
    q("BEGIN TRANSACTION");
    try {
      q("DELETE FROM pending");
      q("INSERT INTO coverage SELECT ticker, tf, min(covered_from) FROM pending_cov GROUP BY ticker, tf "
        "ON CONFLICT DO UPDATE SET covered_from = least(covered_from, excluded.covered_from)");
      q("DELETE FROM pending_cov");
      q("INSERT INTO complete SELECT ticker, tf, max(t) FROM pending_complete GROUP BY ticker, tf "
        "ON CONFLICT DO UPDATE SET t = greatest(t, excluded.t)");
      q("DELETE FROM pending_complete");
      q("COMMIT");
    } catch (...) {
      con.Query("ROLLBACK");
      throw;
    }
  }
};

Lake::Lake(fs::path root) {
  try {
    impl_ = std::make_unique<Impl>(std::move(root));
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("cannot open lake (in use by another process?): ") + e.what());
  }
  auto& I = *impl_;
  I.q("SET threads TO " + std::to_string(std::max(1, omp_get_max_threads())));
  I.q("CREATE TABLE IF NOT EXISTS meta(key VARCHAR PRIMARY KEY, value BIGINT)");
  I.q("INSERT INTO meta VALUES ('seq', 0) ON CONFLICT DO NOTHING");
  I.q("CREATE TABLE IF NOT EXISTS coverage(ticker VARCHAR, tf VARCHAR, covered_from BIGINT, "
      "PRIMARY KEY (ticker, tf))");
  I.q("CREATE TABLE IF NOT EXISTS pending(ticker VARCHAR, t BIGINT, o DOUBLE, h DOUBLE, l DOUBLE, "
      "c DOUBLE, v DOUBLE, vw DOUBLE, seq BIGINT, tf VARCHAR, year INTEGER, month INTEGER)");
  I.q("CREATE TABLE IF NOT EXISTS pending_cov(ticker VARCHAR, tf VARCHAR, covered_from BIGINT)");
  I.q("CREATE TABLE IF NOT EXISTS complete(ticker VARCHAR, tf VARCHAR, t BIGINT, PRIMARY KEY (ticker, tf))");
  I.q("CREATE TABLE IF NOT EXISTS pending_complete(ticker VARCHAR, tf VARCHAR, t BIGINT)");
  I.q("CREATE TEMP TABLE IF NOT EXISTS want(ticker VARCHAR)");
  fs::remove_all(I.root / "_staging");
  I.publish();  // republish anything a crash left in pending
}

Lake::~Lake() = default;

const fs::path& Lake::root() const { return impl_->root; }

void Lake::write(Timeframe tf, const std::vector<LakeRow>& rows,
                 const std::vector<std::pair<std::string, TimePoint>>& coverage,
                 const std::vector<std::pair<std::string, TimePoint>>& complete) {
  if (rows.empty() && coverage.empty() && complete.empty()) return;
  auto& I = *impl_;
  const int64_t seq =
      I.q("UPDATE meta SET value = value + 1 WHERE key = 'seq' RETURNING value")->GetValue(0, 0).GetValue<int64_t>();
  const std::string tfs(to_string(tf));
  {
    // Last occurrence of each (ticker, t) wins within a batch (all rows share one seq).
    std::map<std::pair<std::string, TimePoint>, std::size_t> last;
    for (std::size_t i = 0; i < rows.size(); ++i) last[{rows[i].ticker, rows[i].bar.t}] = i;
    duckdb::Appender app(I.con, "pending");
    for (std::size_t i = 0; i < rows.size(); ++i) {
      const auto& r = rows[i];
      if (last[{r.ticker, r.bar.t}] != i) continue;
      const Civil c = civil_from_days(floor_div(r.bar.t, 86400));
      app.BeginRow();
      app.Append<duckdb::string_t>(duckdb::string_t(r.ticker.data(), static_cast<uint32_t>(r.ticker.size())));
      app.Append<int64_t>(r.bar.t);
      app.Append<double>(r.bar.o);
      app.Append<double>(r.bar.h);
      app.Append<double>(r.bar.l);
      app.Append<double>(r.bar.c);
      app.Append<double>(r.bar.v);
      app.Append<double>(r.bar.vw);
      app.Append<int64_t>(seq);
      app.Append<duckdb::string_t>(duckdb::string_t(tfs.data(), static_cast<uint32_t>(tfs.size())));
      app.Append<int32_t>(c.y);
      app.Append<int32_t>(static_cast<int32_t>(c.m));
      app.EndRow();
    }
    app.Close();
  }
  {
    duckdb::Appender app(I.con, "pending_cov");
    for (const auto& [ticker, from] : coverage) {
      app.BeginRow();
      app.Append<duckdb::string_t>(duckdb::string_t(ticker.data(), static_cast<uint32_t>(ticker.size())));
      app.Append<duckdb::string_t>(duckdb::string_t(tfs.data(), static_cast<uint32_t>(tfs.size())));
      app.Append<int64_t>(from);
      app.EndRow();
    }
    app.Close();
  }
  {
    duckdb::Appender app(I.con, "pending_complete");
    for (const auto& [ticker, t] : complete) {
      app.BeginRow();
      app.Append<duckdb::string_t>(duckdb::string_t(ticker.data(), static_cast<uint32_t>(ticker.size())));
      app.Append<duckdb::string_t>(duckdb::string_t(tfs.data(), static_cast<uint32_t>(tfs.size())));
      app.Append<int64_t>(t);
      app.EndRow();
    }
    app.Close();
  }
  I.publish();
}

std::map<std::string, std::vector<Bar>> Lake::read(Timeframe tf, const std::vector<std::string>& tickers,
                                                   TimePoint start, TimePoint end) {
  std::map<std::string, std::vector<Bar>> out;
  auto& I = *impl_;
  const fs::path dir = I.root / "bars" / tf_dir_name(tf);
  if (tickers.empty() || start > end || !has_parquet(dir)) return out;
  I.set_want(tickers);
  constexpr TimePoint kMin = -62135596800LL;    // 0001-01-01
  constexpr TimePoint kMax = 253402300799LL;    // 9999-12-31
  const TimePoint s = std::max(start, kMin), e = std::min(end, kMax);
  const std::string sql =
      "SELECT ticker, t, o, h, l, c, v, vw FROM read_parquet(" + sql_str((dir / "*" / "*" / "*.parquet").string()) +
      ", hive_partitioning = true) WHERE year BETWEEN " + std::to_string(year_of(s)) + " AND " +
      std::to_string(year_of(e)) + " AND t BETWEEN " + std::to_string(s) + " AND " + std::to_string(e) +
      " AND ticker IN (SELECT ticker FROM want)"
      " QUALIFY row_number() OVER (PARTITION BY ticker, t ORDER BY seq DESC) = 1 ORDER BY ticker, t";
  auto run = [&] {
    out.clear();
    auto r = I.q(sql);
    std::vector<Bar>* cur = nullptr;
    std::string cur_name;
    while (auto chunk = r->Fetch()) {
      chunk->Flatten();
      const std::size_t n = chunk->size();
      auto* tk = duckdb::FlatVector::GetData<duckdb::string_t>(chunk->data[0]);
      auto* t = duckdb::FlatVector::GetData<int64_t>(chunk->data[1]);
      double* cols[6];
      for (int k = 0; k < 6; ++k) cols[k] = duckdb::FlatVector::GetData<double>(chunk->data[2 + k]);
      for (std::size_t i = 0; i < n; ++i) {
        if (!cur || tk[i].GetSize() != cur_name.size() ||
            std::string_view(tk[i].GetData(), tk[i].GetSize()) != cur_name) {
          cur_name = tk[i].GetString();
          cur = &out[cur_name];
        }
        cur->push_back({t[i], cols[0][i], cols[1][i], cols[2][i], cols[3][i], cols[4][i], cols[5][i]});
      }
    }
  };
  try {
    run();
  } catch (const std::exception&) {
    // One bad file must not blank the whole timeframe: quarantine every unreadable file, then retry once.
    if (I.quarantine(tf) == 0) throw;
    if (!has_parquet(dir)) {
      out.clear();
      return out;
    }
    run();
  }
  return out;
}

std::map<std::string, TimePoint> Lake::coverage(Timeframe tf, const std::vector<std::string>& tickers) {
  std::map<std::string, TimePoint> out;
  auto& I = *impl_;
  if (tickers.empty()) return out;
  I.set_want(tickers);
  auto r = I.q("SELECT ticker, covered_from FROM coverage WHERE tf = " + sql_str(std::string(to_string(tf))) +
               " AND ticker IN (SELECT ticker FROM want)");
  for (std::size_t i = 0; i < r->RowCount(); ++i)
    out[r->GetValue(0, i).ToString()] = r->GetValue(1, i).GetValue<int64_t>();
  return out;
}

std::map<std::string, TimePoint> Lake::complete(Timeframe tf, const std::vector<std::string>& tickers) {
  std::map<std::string, TimePoint> out;
  auto& I = *impl_;
  if (tickers.empty()) return out;
  I.set_want(tickers);
  auto r = I.q("SELECT ticker, t FROM complete WHERE tf = " + sql_str(std::string(to_string(tf))) +
               " AND ticker IN (SELECT ticker FROM want)");
  for (std::size_t i = 0; i < r->RowCount(); ++i)
    out[r->GetValue(0, i).ToString()] = r->GetValue(1, i).GetValue<int64_t>();
  return out;
}

std::size_t Lake::compact(Timeframe tf, std::size_t max_files) {
  auto& I = *impl_;
  const fs::path dir = I.root / "bars" / tf_dir_name(tf);
  if (!fs::is_directory(dir)) return 0;
  std::size_t compacted = 0;
  for (const auto& ydir : fs::directory_iterator(dir)) {
    if (!ydir.is_directory()) continue;
    for (const auto& mdir : fs::directory_iterator(ydir.path())) {
      if (!mdir.is_directory()) continue;
      std::vector<fs::path> files;
      for (const auto& f : fs::directory_iterator(mdir.path()))
        if (f.is_regular_file() && f.path().extension() == ".parquet") files.push_back(f.path());
      if (files.size() <= max_files) continue;
      std::sort(files.begin(), files.end());
      std::string list = "[";
      for (std::size_t k = 0; k < files.size(); ++k) list += (k ? "," : "") + sql_str(files[k].string());
      list += "]";
      fs::create_directories(I.root / "_staging");
      const fs::path tmp = I.root / "_staging" / ("compact-" + random_id() + ".parquet");
      I.q("COPY (SELECT ticker, t, o, h, l, c, v, vw, seq FROM read_parquet(" + list +
          ", hive_partitioning = false) QUALIFY row_number() OVER (PARTITION BY ticker, t ORDER BY seq DESC) = 1 "
          "ORDER BY ticker, t) TO " + sql_str(tmp.string()) + " (FORMAT parquet)");
      const fs::path merged = mdir.path() / ("part-compact-" + random_id() + ".parquet");
      fs::rename(tmp, merged);
      fsync_path(merged);
      fsync_path(mdir.path());
      for (const auto& f : files) fs::remove(f);
      ++compacted;
    }
  }
  return compacted;
}

std::size_t Lake::apply_retention(const RetentionPolicy& policy, TimePoint now) {
  auto& I = *impl_;
  std::size_t removed = 0;
  for (const auto& [tf, days] : policy.keep_days) {
    if (!days) continue;
    const TimePoint cutoff = now - static_cast<TimePoint>(*days) * 86400;
    const fs::path dir = I.root / "bars" / tf_dir_name(tf);
    if (!fs::is_directory(dir)) continue;
    std::vector<fs::path> doomed;
    for (const auto& ydir : fs::directory_iterator(dir)) {
      const long y = parse_kv(ydir.path().filename().string(), "year");
      if (!ydir.is_directory() || y < 0) continue;
      for (const auto& mdir : fs::directory_iterator(ydir.path())) {
        const long m = parse_kv(mdir.path().filename().string(), "month");
        if (!mdir.is_directory() || m < 1 || m > 12) continue;
        const int ny = m == 12 ? static_cast<int>(y) + 1 : static_cast<int>(y);
        const unsigned nm = m == 12 ? 1u : static_cast<unsigned>(m) + 1u;
        if (utc_seconds(ny, nm, 1) <= cutoff) doomed.push_back(mdir.path());
      }
    }
    for (const auto& d : doomed) {
      fs::remove_all(d);
      ++removed;
    }
  }
  return removed;
}

std::size_t Lake::file_count(Timeframe tf) const {
  return parquet_files(impl_->root / "bars" / tf_dir_name(tf)).size();
}

}  // namespace fx
