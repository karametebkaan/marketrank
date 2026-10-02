# Fluxscape Milestone 1.6 — DuckDB/Parquet Storage and First Live 10K Run

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the per-ticker CSV cache with an embedded DuckDB catalog plus a Hive-partitioned Parquet bar lake, with crash-safe batched writes, newest-row-wins upserts, windowed reads, per-batch sync commits, compaction, retention and a one-time CSV migration. Then run the first live 10,000-stock universe sync and evaluation.

**Architecture:**
- **`Lake`** (`src/storage/lake.{hpp,cpp}`, pimpl so only `lake.cpp` includes the 2 MB `duckdb.hpp`) does every DuckDB and Parquet operation as plain SQL over `duckdb::Connection::Query`, with `duckdb::Appender` for bulk rows.
- **`BarStore`** keeps its public contract (upsert `merge`, `bars`, first and last time, `covered_from`). It persists through a lazily opened `Lake`, so merge-only users such as the synthetic market and most tests never touch disk.
- **`Panel`** gains a time window and OHLC.
- **`fetch_bars`** reports each 100-symbol batch through a callback, and `sync_bars` commits per batch.

**Tech Stack:** C++20, GCC 13, CMake 3.28, DuckDB 1.5.6 (prebuilt `libduckdb-linux-amd64.zip`), OpenMP, nlohmann/json, doctest.

**Spec:** `docs/superpowers/specs/2026-10-01-fluxscape-design.md` §4.3, plus the storage-swap contract below, which comes from the milestone-1.5 final review.

## Global Constraints

- Commit directly on `master`. Every commit message ends with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. Stage explicit paths only, and never the stray `*.whl` files in the repo root.
- C++20, GCC 13.3. Namespace `fx`. Headers use `#pragma once`. Warning-free under `-Wall -Wextra -Wpedantic`; the DuckDB headers come in as SYSTEM includes through an imported target.
- DuckDB: v1.5.6 from `https://github.com/duckdb/duckdb/releases/download/v1.5.6/libduckdb-linux-amd64.zip`, SHA256 `b845005f5132a7d8180057c35e14a7626632258782f871a90861b19c1c03841b`. Link it as a shared library. Use no DuckDB extensions.
- Only `src/storage/lake.cpp` may include `duckdb.hpp`.
- Lake layout: `data/lake/bars/tf=<tf>/year=<Y>/month=<M>/part-<uuid>.parquet` with columns `ticker, t, o, h, l, c, v, vw, seq`, plus `data/lake/catalog.duckdb`. Default CLI data root: `data/lake`.
- **Storage-swap contract (must hold):**
  1. `merge` upserts by (ticker, tf, t) and the last write wins. Series are sorted and unique. `bars()` of an unknown ticker is empty.
  2. `first_time` and `last_time` are the minimum and maximum t present.
  3. `covered_from` is the earliest start ever requested. It only moves earlier, persists, is independent of whether bars exist, and is committed after (never before) the bars of the same fetch.
  4. A failed write never corrupts existing data. Missing or corrupt data never aborts a load.
  5. Values are lossless: an IEEE double read back equals what was written. `vw ≤ 0` stays a sentinel.
  6. `build_panel`: times are the sorted union, columns follow `tickers` order, missing cells are NaN.
  7. `sync_bars` semantics are unchanged: back-fill unless coverage already reaches start, an inclusive tail fetch, a stale set returned, and data of stale tickers left untouched.
- Retention defaults: `{"1h": 730, "1d": null, "1w": null}` days. Compaction threshold: more than 8 files per partition.
- Never read or print `.env`. Only Task 6 runs `--mode alpaca`.
- Build: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j`. Tests: `./build/fluxtests`.

## File Structure

```
CMakeLists.txt                      + DuckDB imported target
src/storage/lake.hpp/.cpp           NEW  Lake (catalog, write batch, read window, coverage, compact, retention)
src/storage/csv_migration.hpp/.cpp  NEW  migrate_csv_cache (old CSV reader lives here only)
src/market/bar_store.hpp/.cpp       REWRITE  in-memory upsert + queued writes onto Lake
src/market/panel.hpp/.cpp           MOD  window [start, end], open/high/low
src/market/alpaca_client.hpp/.cpp   MOD  fetch_bars on_batch callback
src/market/market_sync.cpp          MOD  per-batch commit via callback
src/main.cpp, src/cli/args.*        MOD  data/lake root, windowed loads, --migrate-cache, --maintain, post-sync maintenance
tests/test_lake.cpp                 NEW
tests/test_csv_migration.cpp        NEW
tests/test_bar_store.cpp            MOD
tests/test_alpaca.cpp               MOD  (assertions on CSV files become lake reloads; per-batch test)
```

---

### Task 1: DuckDB dependency and the Lake

**Files:**
- Modify: `CMakeLists.txt`
- Create: `src/storage/lake.hpp`, `src/storage/lake.cpp`
- Test: `tests/test_lake.cpp`

**Interfaces:**
- Produces:
  - `struct fx::LakeRow { std::string ticker; Bar bar; }`
  - `struct fx::RetentionPolicy { std::map<Timeframe, std::optional<int>> keep_days; static RetentionPolicy defaults(); static RetentionPolicy load(const std::filesystem::path&); }`. A missing file gives the defaults. A malformed file throws `std::runtime_error`.
  - `class fx::Lake`:
    - `explicit Lake(std::filesystem::path root)`. It creates the lake if needed, cleans `_staging`, and republishes leftover pending rows. It throws `std::runtime_error` if the lake can't be opened, for example when another process holds it.
    - `void write(Timeframe, const std::vector<LakeRow>&, const std::vector<std::pair<std::string, TimePoint>>& coverage)`
    - `std::map<std::string, std::vector<Bar>> read(Timeframe, const std::vector<std::string>& tickers, TimePoint start, TimePoint end)`, returning the newest version per (ticker, t), ascending by t.
    - `std::map<std::string, TimePoint> coverage(Timeframe, const std::vector<std::string>& tickers)`
    - `std::size_t compact(Timeframe, std::size_t max_files)`
    - `std::size_t apply_retention(const RetentionPolicy&, TimePoint now)`
    - `std::size_t file_count(Timeframe) const`
    - `const std::filesystem::path& root() const`

- [ ] **Step 1: Add DuckDB to the build.** In `CMakeLists.txt`, after the doctest `FetchContent_Declare`, add:
```cmake
FetchContent_Declare(duckdb_bin
  URL https://github.com/duckdb/duckdb/releases/download/v1.5.6/libduckdb-linux-amd64.zip
  URL_HASH SHA256=b845005f5132a7d8180057c35e14a7626632258782f871a90861b19c1c03841b
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
```
Add `duckdb_bin` to the `FetchContent_MakeAvailable(...)` list. The zip has no CMakeLists.txt, so it is only extracted. After `find_package(OpenMP REQUIRED)`, add:
```cmake
add_library(duckdb SHARED IMPORTED GLOBAL)
set_target_properties(duckdb PROPERTIES
  IMPORTED_LOCATION ${duckdb_bin_SOURCE_DIR}/libduckdb.so
  INTERFACE_INCLUDE_DIRECTORIES ${duckdb_bin_SOURCE_DIR})
```
Then append `duckdb` to fluxcore's `target_link_libraries(... PUBLIC ...)` list.
Run `cmake -S . -B build && cmake --build build -j && ./build/fluxtests`. Expected: SUCCESS, with `build/_deps/duckdb_bin-src/libduckdb.so` present.

- [ ] **Step 2: Write the failing tests** in `tests/test_lake.cpp`:
```cpp
#include <doctest/doctest.h>

#include <filesystem>
#include <fstream>

#include "core/time.hpp"
#include "storage/lake.hpp"
#include "test_util.hpp"

using namespace fx;

namespace {
LakeRow row(const std::string& ticker, TimePoint t, double c) {
  return {ticker, Bar{t, c - 1, c + 1, c - 2, c, 1000 + c, c + 0.5}};
}
}  // namespace

TEST_CASE("lake round-trips rows losslessly and filters by ticker and window") {
  auto dir = test::temp_dir("lake_rt");
  const TimePoint d0 = utc_seconds(2026, 9, 1, 4);
  {
    Lake lake(dir);
    std::vector<LakeRow> rows = {row("AAPL", d0, 0.1 + 0.2), row("AAPL", d0 + 86400, 1.0 / 3.0),
                                 row("BRK.B", d0, 470.25), row("MSFT", d0 + 40 * 86400, 5.0)};
    lake.write(Timeframe::Day, rows, {});
  }
  Lake lake(dir);
  auto got = lake.read(Timeframe::Day, {"AAPL", "BRK.B"}, d0, d0 + 86400);
  REQUIRE(got.size() == 2);
  REQUIRE(got["AAPL"].size() == 2);
  CHECK(got["AAPL"][0].c == 0.1 + 0.2);  // exact
  CHECK(got["AAPL"][1].c == 1.0 / 3.0);
  CHECK(got["AAPL"][1].vw == 1.0 / 3.0 + 0.5);
  CHECK(got["BRK.B"][0].t == d0);
  CHECK(lake.read(Timeframe::Day, {"MSFT"}, d0, d0 + 86400).empty());
  CHECK(lake.read(Timeframe::Day, {"MSFT"}, d0, d0 + 50 * 86400)["MSFT"].size() == 1);
  CHECK(lake.read(Timeframe::Hour, {"AAPL"}, d0, d0 + 86400).empty());
}

TEST_CASE("newest write wins for the same ticker and time") {
  auto dir = test::temp_dir("lake_upsert");
  const TimePoint d0 = utc_seconds(2026, 9, 1, 4);
  Lake lake(dir);
  lake.write(Timeframe::Day, {row("AAPL", d0, 10)}, {});
  lake.write(Timeframe::Day, {row("AAPL", d0, 11)}, {});
  auto got = lake.read(Timeframe::Day, {"AAPL"}, d0, d0);
  REQUIRE(got["AAPL"].size() == 1);
  CHECK(got["AAPL"][0].c == 11);
}

TEST_CASE("coverage only moves earlier and persists across reopen") {
  auto dir = test::temp_dir("lake_cov");
  {
    Lake lake(dir);
    lake.write(Timeframe::Day, {}, {{"AAPL", 500}});
    lake.write(Timeframe::Day, {}, {{"AAPL", 900}, {"NVO", 700}});
  }
  Lake lake(dir);
  auto cov = lake.coverage(Timeframe::Day, {"AAPL", "NVO", "MSFT"});
  CHECK(cov.at("AAPL") == 500);
  CHECK(cov.at("NVO") == 700);
  CHECK_FALSE(cov.count("MSFT"));
  CHECK(lake.coverage(Timeframe::Hour, {"AAPL"}).empty());
}

TEST_CASE("leftover staging is removed on open") {
  auto dir = test::temp_dir("lake_staging");
  { Lake lake(dir); }
  test::write_file(dir / "_staging" / "junk" / "half.parquet", "not parquet");
  Lake lake(dir);
  CHECK_FALSE(std::filesystem::exists(dir / "_staging" / "junk"));
  CHECK(lake.read(Timeframe::Day, {"AAPL"}, 0, 1LL << 40).empty());
}

TEST_CASE("compaction merges partition files and keeps the newest rows") {
  auto dir = test::temp_dir("lake_compact");
  const TimePoint d0 = utc_seconds(2026, 9, 1, 4);
  Lake lake(dir);
  for (int b = 0; b < 5; ++b) lake.write(Timeframe::Day, {row("AAPL", d0, 10 + b), row("F", d0 + b * 86400, 3)}, {});
  CHECK(lake.file_count(Timeframe::Day) == 5);
  CHECK(lake.compact(Timeframe::Day, 2) == 1);
  CHECK(lake.file_count(Timeframe::Day) == 1);
  auto got = lake.read(Timeframe::Day, {"AAPL", "F"}, d0, d0 + 10 * 86400);
  CHECK(got["AAPL"].size() == 1);
  CHECK(got["AAPL"][0].c == 14);
  CHECK(got["F"].size() == 5);
  CHECK(lake.compact(Timeframe::Day, 2) == 0);
}

TEST_CASE("retention drops whole months older than the cutoff only") {
  auto dir = test::temp_dir("lake_ret");
  Lake lake(dir);
  lake.write(Timeframe::Hour, {row("AAPL", utc_seconds(2024, 1, 15, 15), 1), row("AAPL", utc_seconds(2026, 9, 2, 15), 2)}, {});
  lake.write(Timeframe::Day, {row("AAPL", utc_seconds(2020, 1, 15, 4), 1)}, {});
  RetentionPolicy p = RetentionPolicy::defaults();
  p.keep_days[Timeframe::Hour] = 30;
  CHECK(lake.apply_retention(p, utc_seconds(2026, 10, 1)) == 1);
  auto hours = lake.read(Timeframe::Hour, {"AAPL"}, 0, 1LL << 40);
  REQUIRE(hours["AAPL"].size() == 1);
  CHECK(hours["AAPL"][0].c == 2);
  CHECK(lake.read(Timeframe::Day, {"AAPL"}, 0, 1LL << 40)["AAPL"].size() == 1);
}

TEST_CASE("retention policy file") {
  auto dir = test::temp_dir("lake_policy");
  auto d = RetentionPolicy::defaults();
  CHECK(d.keep_days.at(Timeframe::Hour).value() == 730);
  CHECK_FALSE(d.keep_days.at(Timeframe::Day).has_value());
  CHECK(RetentionPolicy::load(dir / "missing.json").keep_days.at(Timeframe::Hour).value() == 730);
  auto p = RetentionPolicy::load(test::write_file(dir / "r.json", R"({"1h": 90, "1w": 3650})"));
  CHECK(p.keep_days.at(Timeframe::Hour).value() == 90);
  CHECK(p.keep_days.at(Timeframe::Week).value() == 3650);
  CHECK_FALSE(p.keep_days.at(Timeframe::Day).has_value());
  CHECK_THROWS_AS(RetentionPolicy::load(test::write_file(dir / "bad.json", "{")), std::runtime_error);
}
```

- [ ] **Step 3: Run the build to verify it fails**

Run: `cmake -S . -B build && cmake --build build -j`
Expected: FAIL with `storage/lake.hpp: No such file or directory`.

- [ ] **Step 4: Implement**

`src/storage/lake.hpp`:
```cpp
#pragma once
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/types.hpp"

namespace fx {

struct LakeRow {
  std::string ticker;
  Bar bar;
};

struct RetentionPolicy {
  std::map<Timeframe, std::optional<int>> keep_days;  // nullopt = keep forever
  static RetentionPolicy defaults();                  // 1h: 730, 1d/1w: forever
  static RetentionPolicy load(const std::filesystem::path& json);  // missing file = defaults
};

// Spec 4.3: Hive-partitioned Parquet bars + a small DuckDB catalog. Single process.
class Lake {
 public:
  explicit Lake(std::filesystem::path root);
  ~Lake();
  Lake(const Lake&) = delete;
  Lake& operator=(const Lake&) = delete;

  // One write batch (new seq). Data becomes visible atomically per file; coverage is applied after.
  void write(Timeframe tf, const std::vector<LakeRow>& rows,
             const std::vector<std::pair<std::string, TimePoint>>& coverage);
  std::map<std::string, std::vector<Bar>> read(Timeframe tf, const std::vector<std::string>& tickers,
                                               TimePoint start, TimePoint end);
  std::map<std::string, TimePoint> coverage(Timeframe tf, const std::vector<std::string>& tickers);
  std::size_t compact(Timeframe tf, std::size_t max_files);
  std::size_t apply_retention(const RetentionPolicy& policy, TimePoint now);
  std::size_t file_count(Timeframe tf) const;
  const std::filesystem::path& root() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace fx
```

`src/storage/lake.cpp`:
```cpp
#include "storage/lake.hpp"

#include <omp.h>

#include <algorithm>
#include <fstream>
#include <nlohmann/json.hpp>
#include <random>
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

  // Moves pending rows into Parquet partitions, then applies pending coverage.
  void publish() {
    const auto n = q("SELECT count(*) FROM pending")->GetValue(0, 0).GetValue<int64_t>();
    if (n > 0) {
      fs::create_directories(root / "_staging");
      const fs::path staging = root / "_staging" / random_id();
      q("COPY (SELECT ticker, t, o, h, l, c, v, vw, seq, tf, year, month FROM pending) TO " +
        sql_str(staging.string()) +
        " (FORMAT parquet, PARTITION_BY (tf, year, month), FILENAME_PATTERN 'part-{uuid}')");
      for (const auto& f : parquet_files(staging)) {
        const fs::path dest = root / "bars" / fs::relative(f, staging);
        fs::create_directories(dest.parent_path());
        fs::rename(f, dest);
      }
      fs::remove_all(staging);
    }
    q("BEGIN TRANSACTION");
    try {
      q("DELETE FROM pending");
      q("INSERT INTO coverage SELECT ticker, tf, min(covered_from) FROM pending_cov GROUP BY ticker, tf "
        "ON CONFLICT DO UPDATE SET covered_from = least(covered_from, excluded.covered_from)");
      q("DELETE FROM pending_cov");
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
  I.q("CREATE TABLE IF NOT EXISTS want(ticker VARCHAR)");
  fs::remove_all(I.root / "_staging");
  I.publish();  // republish anything a crash left in pending
}

Lake::~Lake() = default;

const fs::path& Lake::root() const { return impl_->root; }

void Lake::write(Timeframe tf, const std::vector<LakeRow>& rows,
                 const std::vector<std::pair<std::string, TimePoint>>& coverage) {
  if (rows.empty() && coverage.empty()) return;
  auto& I = *impl_;
  const int64_t seq =
      I.q("UPDATE meta SET value = value + 1 WHERE key = 'seq' RETURNING value")->GetValue(0, 0).GetValue<int64_t>();
  const std::string tfs(to_string(tf));
  {
    duckdb::Appender app(I.con, "pending");
    for (const auto& r : rows) {
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
  I.publish();
}

std::map<std::string, std::vector<Bar>> Lake::read(Timeframe tf, const std::vector<std::string>& tickers,
                                                   TimePoint start, TimePoint end) {
  std::map<std::string, std::vector<Bar>> out;
  auto& I = *impl_;
  const fs::path dir = I.root / "bars" / tf_dir_name(tf);
  if (tickers.empty() || start > end || parquet_files(dir).empty()) return out;
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
  auto r = I.q(sql);
  while (auto chunk = r->Fetch()) {
    chunk->Flatten();
    const std::size_t n = chunk->size();
    auto* tk = duckdb::FlatVector::GetData<duckdb::string_t>(chunk->data[0]);
    auto* t = duckdb::FlatVector::GetData<int64_t>(chunk->data[1]);
    double* cols[6];
    for (int k = 0; k < 6; ++k) cols[k] = duckdb::FlatVector::GetData<double>(chunk->data[2 + k]);
    for (std::size_t i = 0; i < n; ++i)
      out[tk[i].GetString()].push_back({t[i], cols[0][i], cols[1][i], cols[2][i], cols[3][i], cols[4][i], cols[5][i]});
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
      fs::rename(tmp, mdir.path() / ("part-compact-" + random_id() + ".parquet"));
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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cmake --build build -j && ./build/fluxtests -tc="*lake*,*retention*,*compaction*,*newest*,*coverage only*,*staging*"`, then the full `./build/fluxtests`.
Expected: SUCCESS and no warnings (`touch src/storage/lake.cpp && cmake --build build 2>&1 | grep -i warning`).
If a DuckDB call behaves differently from the comments here, adapt it minimally, keep the tested behavior, and say so in the report.

- [ ] **Step 6: Commit**
```bash
git add CMakeLists.txt src/storage tests/test_lake.cpp
git commit -m "feat: DuckDB Hive-partitioned Parquet lake with upserts, coverage, compaction and retention

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: BarStore on the Lake, windowed OHLC panel, CSV migration

**Files:**
- Modify (rewrite): `src/market/bar_store.hpp`, `src/market/bar_store.cpp`
- Modify: `src/market/panel.hpp`, `src/market/panel.cpp`
- Create: `src/storage/csv_migration.hpp`, `src/storage/csv_migration.cpp`
- Test: `tests/test_bar_store.cpp` (rework), `tests/test_csv_migration.cpp` (new), plus any test that asserts on CSV cache files (see Step 1)

**Interfaces:**
- Consumes: `Lake`, `LakeRow` (Task 1).
- Produces:
  - `class fx::BarStore`:
    - `explicit BarStore(std::filesystem::path lake_root)` and `~BarStore()`. The destructor runs a best-effort `flush()` and swallows errors.
    - `void merge(const std::string&, Timeframe, const std::vector<Bar>&)`: an in-memory upsert that also queues the incoming bars.
    - `const std::vector<Bar>& bars(...) const`, `first_time`, `last_time`, `covered_from` (unchanged signatures).
    - `void set_covered_from(...)`: in memory now; persisted on the next `flush()` after that flush's bars.
    - `void save(const std::string&, Timeframe)`: marks that series' queued bars for the next flush. It is no longer `const`.
    - `void flush()`: one `Lake::write` per timeframe.
    - `void load_range(const std::vector<std::string>&, Timeframe, TimePoint start, TimePoint end)` and `void load_all(const std::vector<std::string>&, Timeframe)`. Loads don't queue anything.
    - `Lake& lake()`, opened lazily.
  - `Panel` gains `std::vector<double> open, high, low` (row-major, NaN where missing). The signature becomes `build_panel(const BarStore&, const std::vector<std::string>& tickers, Timeframe, TimePoint start = INT64_MIN, TimePoint end = INT64_MAX)`.
  - `std::size_t fx::migrate_csv_cache(const std::filesystem::path& csv_root, BarStore& store)`. It returns the number of series imported, and tolerates bad rows, bad headers and unreadable files.

- [ ] **Step 1: Find every test that depends on the CSV cache format.** Run `grep -rn "\.csv\"\|\.from\"\|\"1d\" / \|\"1h\" /\|\.tmp" tests`. Expect hits in `tests/test_bar_store.cpp` (garbage row, wrong header, no `.tmp`) and `tests/test_alpaca.cpp` (`dir / "1d" / "AAPL.csv"` existence). Apply the rules below.
  - **Bar-store CSV-format tests** (garbage row, wrong header, no `.tmp`): move them into `tests/test_csv_migration.cpp` as migration tests. Write the CSV files with `test::write_file`, run `migrate_csv_cache(dir, store)`, then assert on `store.bars(...)`.
  - **Any check that a CSV file exists after a sync:** replace it with a reload. Create a new `BarStore` on the same directory after the first one is destroyed, call `load_all`, and assert on the bars. Destroy the first store before opening a second one on the same directory; put it in its own `{}` scope. Two live stores on one lake in the same process is not supported.
  - **The `.from` sidecar tests:** they keep working through `covered_from` after a reload. Remove any assertion on `.from` files.

- [ ] **Step 2: Write the new failing tests.** Append to `tests/test_bar_store.cpp`, or rework its existing round-trip case into these:
```cpp
TEST_CASE("merge-only stores never touch disk") {
  auto dir = test::temp_dir("bars_mem");
  {
    BarStore s(dir / "lake");
    s.merge("AAPL", Timeframe::Day, {{100, 1, 1, 1, 1, 10, 1}});
  }
  CHECK_FALSE(std::filesystem::exists(dir / "lake"));
}

TEST_CASE("saved bars and coverage persist losslessly through the lake") {
  auto dir = test::temp_dir("bars_lake");
  {
    BarStore s(dir);
    s.merge("BRK.B", Timeframe::Hour, {{1759239000, 470.25, 471.5, 469.0, 0.1 + 0.2, 123456, 1.0 / 3.0}});
    s.save("BRK.B", Timeframe::Hour);
    s.set_covered_from("BRK.B", Timeframe::Hour, 1759000000);
    s.flush();
    s.merge("BRK.B", Timeframe::Hour, {{1759239000, 1, 1, 1, 9.5, 1, 1}});  // upsert, saved by destructor
    s.save("BRK.B", Timeframe::Hour);
  }
  BarStore s2(dir);
  s2.load_all({"BRK.B", "MISSING"}, Timeframe::Hour);
  const auto& b = s2.bars("BRK.B", Timeframe::Hour);
  REQUIRE(b.size() == 1);
  CHECK(b[0].c == 9.5);
  CHECK(s2.covered_from("BRK.B", Timeframe::Hour).value() == 1759000000);
  CHECK(s2.bars("MISSING", Timeframe::Hour).empty());
}

TEST_CASE("load_range only loads the window and panel windows and carries OHLC") {
  auto dir = test::temp_dir("bars_window");
  {
    BarStore s(dir);
    s.merge("A", Timeframe::Day, {{100, 9, 12, 8, 10, 5, 10}, {200, 10, 13, 9, 11, 6, 11}, {300, 11, 14, 10, 12, 7, 12}});
    s.save("A", Timeframe::Day);
  }
  BarStore s2(dir);
  s2.load_range({"A"}, Timeframe::Day, 150, 400);
  CHECK(s2.bars("A", Timeframe::Day).size() == 2);
  Panel p = build_panel(s2, {"A"}, Timeframe::Day, 250, 400);
  REQUIRE(p.T() == 1);
  CHECK(p.open[p.idx(0, 0)] == 11);
  CHECK(p.high[p.idx(0, 0)] == 14);
  CHECK(p.low[p.idx(0, 0)] == 10);
  CHECK(p.close[p.idx(0, 0)] == 12);
}
```
Create `tests/test_csv_migration.cpp`. Add the migrated CSV-format cases from Step 1 plus:
```cpp
#include <doctest/doctest.h>

#include "market/bar_store.hpp"
#include "storage/csv_migration.hpp"
#include "test_util.hpp"

using namespace fx;

TEST_CASE("migrate_csv_cache imports bars and .from coverage, skipping bad rows") {
  auto dir = test::temp_dir("migrate");
  test::write_file(dir / "csv" / "1d" / "AAPL.csv", "t,o,h,l,c,v,vw\n100,1,2,0.5,1.5,10,1.4\nbad,row\n200,1,2,0.5,1.75,11,1.6\n");
  test::write_file(dir / "csv" / "1d" / "AAPL.from", "50\n");
  test::write_file(dir / "csv" / "1d" / "JUNK.csv", "wrong,header\n1,2\n");
  {
    BarStore s(dir / "lake");
    CHECK(migrate_csv_cache(dir / "csv", s) == 1);
  }
  BarStore s(dir / "lake");
  s.load_all({"AAPL", "JUNK"}, Timeframe::Day);
  REQUIRE(s.bars("AAPL", Timeframe::Day).size() == 2);
  CHECK(s.bars("AAPL", Timeframe::Day)[1].c == 1.75);
  CHECK(s.covered_from("AAPL", Timeframe::Day).value() == 50);
  CHECK(s.bars("JUNK", Timeframe::Day).empty());
  CHECK(migrate_csv_cache(dir / "nothing", s) == 0);
}
```

- [ ] **Step 3: Run the build to verify it fails**

Run: `cmake -S . -B build && cmake --build build -j`
Expected: FAIL. There is no `flush` or `load_range`, `storage/csv_migration.hpp` is missing, and `Panel` has no `open`.

- [ ] **Step 4: Implement**

Replace `src/market/bar_store.hpp` with:
```cpp
#pragma once
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/types.hpp"
#include "storage/lake.hpp"

namespace fx {

// In-memory bar series with upsert semantics, persisted to a Lake (spec 4.3). Merge-only use
// (synthetic markets, tests) never opens the lake.
class BarStore {
 public:
  explicit BarStore(std::filesystem::path lake_root);
  ~BarStore();
  BarStore(const BarStore&) = delete;
  BarStore& operator=(const BarStore&) = delete;

  void merge(const std::string& ticker, Timeframe tf, const std::vector<Bar>& incoming);
  const std::vector<Bar>& bars(const std::string& ticker, Timeframe tf) const;
  std::optional<TimePoint> last_time(const std::string& ticker, Timeframe tf) const;
  std::optional<TimePoint> first_time(const std::string& ticker, Timeframe tf) const;
  // Earliest start ever requested; only moves earlier; persisted after that flush's bars.
  std::optional<TimePoint> covered_from(const std::string& ticker, Timeframe tf) const;
  void set_covered_from(const std::string& ticker, Timeframe tf, TimePoint t);
  void save(const std::string& ticker, Timeframe tf);  // mark queued bars for the next flush
  void flush();                                        // persist everything saved so far
  void load_range(const std::vector<std::string>& tickers, Timeframe tf, TimePoint start, TimePoint end);
  void load_all(const std::vector<std::string>& tickers, Timeframe tf);
  Lake& lake();

 private:
  using Key = std::pair<std::string, Timeframe>;
  void upsert(const Key& key, const std::vector<Bar>& incoming);
  std::filesystem::path root_;
  std::unique_ptr<Lake> lake_;
  std::map<Key, std::vector<Bar>> series_;
  std::map<Key, TimePoint> covered_;
  std::map<Key, std::vector<Bar>> queued_;    // merged, not yet saved
  std::map<Key, std::vector<Bar>> to_write_;  // saved, awaiting flush
  std::map<Key, TimePoint> cov_to_write_;
};

}  // namespace fx
```

Replace `src/market/bar_store.cpp` with:
```cpp
#include "market/bar_store.hpp"

#include <algorithm>
#include <limits>

namespace fx {

BarStore::BarStore(std::filesystem::path lake_root) : root_(std::move(lake_root)) {}

BarStore::~BarStore() {
  try {
    flush();
  } catch (...) {
  }
}

Lake& BarStore::lake() {
  if (!lake_) lake_ = std::make_unique<Lake>(root_);
  return *lake_;
}

void BarStore::upsert(const Key& key, const std::vector<Bar>& incoming) {
  auto& series = series_[key];
  std::map<TimePoint, Bar> by_time;
  for (const Bar& b : series) by_time[b.t] = b;
  for (const Bar& b : incoming) by_time[b.t] = b;
  series.clear();
  series.reserve(by_time.size());
  for (const auto& [t, b] : by_time) series.push_back(b);
}

void BarStore::merge(const std::string& ticker, Timeframe tf, const std::vector<Bar>& incoming) {
  const Key key{ticker, tf};
  upsert(key, incoming);
  auto& q = queued_[key];
  q.insert(q.end(), incoming.begin(), incoming.end());
}

const std::vector<Bar>& BarStore::bars(const std::string& ticker, Timeframe tf) const {
  static const std::vector<Bar> kEmpty;
  auto it = series_.find({ticker, tf});
  return it == series_.end() ? kEmpty : it->second;
}

std::optional<TimePoint> BarStore::last_time(const std::string& ticker, Timeframe tf) const {
  const auto& b = bars(ticker, tf);
  if (b.empty()) return std::nullopt;
  return b.back().t;
}

std::optional<TimePoint> BarStore::first_time(const std::string& ticker, Timeframe tf) const {
  const auto& b = bars(ticker, tf);
  if (b.empty()) return std::nullopt;
  return b.front().t;
}

std::optional<TimePoint> BarStore::covered_from(const std::string& ticker, Timeframe tf) const {
  auto it = covered_.find({ticker, tf});
  if (it == covered_.end()) return std::nullopt;
  return it->second;
}

void BarStore::set_covered_from(const std::string& ticker, Timeframe tf, TimePoint t) {
  const Key key{ticker, tf};
  auto it = covered_.find(key);
  if (it != covered_.end() && it->second <= t) return;
  covered_[key] = t;
  auto [w, inserted] = cov_to_write_.emplace(key, t);
  if (!inserted) w->second = std::min(w->second, t);
}

void BarStore::save(const std::string& ticker, Timeframe tf) {
  const Key key{ticker, tf};
  auto it = queued_.find(key);
  if (it == queued_.end()) return;
  auto& w = to_write_[key];
  w.insert(w.end(), it->second.begin(), it->second.end());
  queued_.erase(it);
}

void BarStore::flush() {
  if (to_write_.empty() && cov_to_write_.empty()) return;
  for (Timeframe tf : {Timeframe::Hour, Timeframe::Day, Timeframe::Week}) {
    std::vector<LakeRow> rows;
    std::vector<std::pair<std::string, TimePoint>> cov;
    for (const auto& [key, bars] : to_write_)
      if (key.second == tf)
        for (const Bar& b : bars) rows.push_back({key.first, b});
    for (const auto& [key, t] : cov_to_write_)
      if (key.second == tf) cov.emplace_back(key.first, t);
    if (!rows.empty() || !cov.empty()) lake().write(tf, rows, cov);
  }
  to_write_.clear();
  cov_to_write_.clear();
}

void BarStore::load_range(const std::vector<std::string>& tickers, Timeframe tf, TimePoint start,
                          TimePoint end) {
  try {
    for (auto& [ticker, loaded] : lake().read(tf, tickers, start, end)) upsert({ticker, tf}, loaded);
    for (const auto& [ticker, t] : lake().coverage(tf, tickers)) {
      auto it = covered_.find({ticker, tf});
      if (it == covered_.end() || t < it->second) covered_[{ticker, tf}] = t;
    }
  } catch (const std::exception&) {
    // Contract 4: unreadable data never aborts a load; the caller sees missing series.
  }
}

void BarStore::load_all(const std::vector<std::string>& tickers, Timeframe tf) {
  load_range(tickers, tf, std::numeric_limits<TimePoint>::min(), std::numeric_limits<TimePoint>::max());
}

}  // namespace fx
```
Note on `load_range`: catching every exception keeps contract 4 (a corrupt lake must not abort a load). But it also hides an "in use by another process" error. To keep that visible, let the `Lake` construction error escape: call `lake();` once before the `try`.

Panel changes:
- In `src/market/panel.hpp`, add `open, high, low` to the vector list (`std::vector<double> open, high, low, close, volume, vwap;`). Change the declaration to:
```cpp
Panel build_panel(const BarStore& store, const std::vector<std::string>& tickers, Timeframe tf,
                  TimePoint start = std::numeric_limits<TimePoint>::min(),
                  TimePoint end = std::numeric_limits<TimePoint>::max());
```
  and add `#include <limits>`.
- In `src/market/panel.cpp`:
  - Collect times only for bars with `start <= b.t && b.t <= end`.
  - Assign `open`, `high` and `low` with NaN like the others.
  - In the fill loop, skip bars outside the window and set `p.open[k] = b.o; p.high[k] = b.h; p.low[k] = b.l;`.

`src/storage/csv_migration.hpp`:
```cpp
#pragma once
#include <cstddef>
#include <filesystem>

#include "market/bar_store.hpp"

namespace fx {

// One-time import of the milestone-1 CSV cache (<root>/<1h|1d|1w>/<TICKER>.csv + .from sidecars).
std::size_t migrate_csv_cache(const std::filesystem::path& csv_root, BarStore& store);

}  // namespace fx
```

`src/storage/csv_migration.cpp`:
```cpp
#include "storage/csv_migration.hpp"

#include <fstream>

#include "core/csv.hpp"

namespace fx {

std::size_t migrate_csv_cache(const std::filesystem::path& csv_root, BarStore& store) {
  namespace fs = std::filesystem;
  static const std::vector<std::string> kHeader = {"t", "o", "h", "l", "c", "v", "vw"};
  std::size_t imported = 0, since_flush = 0;
  for (Timeframe tf : {Timeframe::Hour, Timeframe::Day, Timeframe::Week}) {
    const fs::path dir = csv_root / std::string(to_string(tf));
    if (!fs::is_directory(dir)) continue;
    for (const auto& e : fs::directory_iterator(dir)) {
      if (!e.is_regular_file()) continue;
      const std::string ticker = e.path().stem().string();
      if (e.path().extension() == ".from") {
        std::ifstream in(e.path());
        long long v = 0;
        if (in >> v) store.set_covered_from(ticker, tf, v);
        continue;
      }
      if (e.path().extension() != ".csv") continue;
      try {
        const CsvRows rows = read_csv_file(e.path());
        if (rows.empty() || rows[0] != kHeader) continue;
        std::vector<Bar> bars;
        for (std::size_t r = 1; r < rows.size(); ++r) {
          const auto& f = rows[r];
          if (f.size() < 7) continue;
          try {
            bars.push_back({std::stoll(f[0]), std::stod(f[1]), std::stod(f[2]), std::stod(f[3]),
                            std::stod(f[4]), std::stod(f[5]), std::stod(f[6])});
          } catch (const std::exception&) {
          }
        }
        if (bars.empty()) continue;
        store.merge(ticker, tf, bars);
        store.save(ticker, tf);
        ++imported;
        if (++since_flush >= 500) {
          store.flush();
          since_flush = 0;
        }
      } catch (const std::exception&) {
      }
    }
  }
  store.flush();
  return imported;
}

}  // namespace fx
```

Update callers of `save` that relied on it being `const` (`market_sync.cpp` only calls it on a non-const store, so no change is needed). Make sure `src/main.cpp` still compiles; Task 4 rewires it.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cmake --build build -j && ./build/fluxtests`
Expected: SUCCESS, no warnings. Every pre-existing test passes, after the Step 1 adjustments.

- [ ] **Step 6: Commit**
```bash
git add src/market/bar_store.* src/market/panel.* src/storage/csv_migration.* tests/test_bar_store.cpp tests/test_csv_migration.cpp tests/test_alpaca.cpp
git commit -m "feat: BarStore persists to the Parquet lake; windowed OHLC panel; CSV migration

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
(Include any other test files you changed in Step 1.)

---

### Task 3: Per-batch commits during sync

**Files:**
- Modify: `src/market/alpaca_client.hpp`, `src/market/alpaca_client.cpp`, `src/market/market_sync.cpp`
- Test: `tests/test_alpaca.cpp`

**Interfaces:**
- Produces:
  - `using fx::BatchCallback = std::function<void(const std::vector<std::string>& batch_symbols, const std::map<std::string, std::vector<Bar>>& batch_bars)>;`
  - `FetchResult AlpacaClient::fetch_bars(const std::vector<std::string>&, std::string_view timeframe, TimePoint start, TimePoint end, const BatchCallback& on_batch = {})`. The callback runs once per successful 100-symbol batch, before the next batch is fetched. `FetchResult.bars` still accumulates everything.
  - `sync_bars` merges, saves, sets coverage and flushes the store inside the callback, so each batch is committed before the next request.

- [ ] **Step 1: Write the failing test.** Append to `tests/test_alpaca.cpp`:
```cpp
TEST_CASE("sync commits each 100-symbol batch before fetching the next") {
  auto dir = test::temp_dir("sync_batches");
  std::vector<std::string> tickers;
  for (int i = 0; i < 150; ++i) tickers.push_back("T" + std::to_string(i));
  int calls = 0;
  {
    BarStore store(dir);
    AlpacaClient client(test_config(), [&](const std::string& path) -> HttpResponse {
      ++calls;
      if (path.find("symbols=T0,") == std::string::npos) return {403, "no"};  // second batch fails
      return {200, R"({"bars":{"T0":[{"t":"2026-09-29T04:00:00Z","o":1,"h":1,"l":1,"c":1,"v":1,"vw":1}]}})"};
    });
    auto stale = sync_bars(client, store, tickers, Timeframe::Day, utc_seconds(2026, 9, 1), utc_seconds(2026, 10, 1));
    CHECK(stale.size() == 50);
  }
  BarStore reloaded(dir);
  reloaded.load_all({"T0", "T120"}, Timeframe::Day);
  CHECK(reloaded.bars("T0", Timeframe::Day).size() == 1);
  CHECK(reloaded.covered_from("T0", Timeframe::Day).has_value());   // batch 1 committed
  CHECK_FALSE(reloaded.covered_from("T120", Timeframe::Day).has_value());  // failed batch not covered
  CHECK(calls == 2);
}

TEST_CASE("fetch_bars reports each successful batch to the callback") {
  std::vector<std::string> symbols;
  for (int i = 0; i < 250; ++i) symbols.push_back("S" + std::to_string(i));
  AlpacaClient client(test_config(), [](const std::string&) { return HttpResponse{200, R"({"bars":{}})"}; });
  std::vector<std::size_t> sizes;
  client.fetch_bars(symbols, "1Day", 0, 1,
                    [&](const std::vector<std::string>& batch, const std::map<std::string, std::vector<Bar>>&) {
                      sizes.push_back(batch.size());
                    });
  CHECK(sizes == std::vector<std::size_t>{100, 100, 50});
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cmake --build build -j`
Expected: FAIL, because `fetch_bars` doesn't take a callback.

- [ ] **Step 3: Implement**
- In `alpaca_client.hpp`, declare `BatchCallback` (add `#include <functional>`) and add the defaulted parameter to `fetch_bars`.
- In `alpaca_client.cpp`, inside the per-batch loop, after a batch's pages have all succeeded and its bars have been moved into `result.bars`, call `if (on_batch) on_batch(batch_symbols, batch_bars);`. Build `batch_symbols` from `symbols[b .. batch_end)`, and keep a copy of the batch's map before moving it into the result. A failed batch never calls the callback.
- In `market_sync.cpp` `fetch_into`, pass a callback that does the following for the batch:
  1. For each ticker in `batch_bars`, call `store.merge(...)` (with `aggregate_session_hours` for Hour) and then `store.save(t, tf)`.
  2. For each symbol in `batch_symbols` that is also in `cover`, call `store.set_covered_from(t, tf, from)`.
  3. Call `store.flush()`.

  Then remove the old after-the-fact merge, save and coverage loop. Keep `stale.insert(r.stale...)`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cmake --build build -j && ./build/fluxtests`
Expected: SUCCESS, no warnings. All earlier sync and back-fill tests still pass.

- [ ] **Step 5: Commit**
```bash
git add src/market/alpaca_client.* src/market/market_sync.cpp tests/test_alpaca.cpp
git commit -m "feat: commit each fetch batch to the lake during sync

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: CLI on the lake — windowed loads, migration, maintenance

**Files:**
- Modify: `src/cli/args.hpp`, `src/cli/args.cpp`, `src/main.cpp`, `.gitignore`, `README.md`
- Test: `tests/test_cli_args.cpp`

**Interfaces:**
- Produces:
  - `CliArgs` gains `bool migrate_cache = false;`, `std::filesystem::path migrate_from = "data/cache";` and `bool maintain = false;`. New flags: `--migrate-cache [DIR]` (the value is optional and used only if the next argument doesn't start with `--`) and `--maintain`.
  - `--data` now defaults to `data`, and the lake lives at `<data>/lake`.

- [ ] **Step 1: Write the failing tests.** Append to `tests/test_cli_args.cpp`:
```cpp
TEST_CASE("storage flags") {
  CliArgs a = parse_cli({"--migrate-cache"});
  CHECK(a.migrate_cache);
  CHECK(a.migrate_from == "data/cache");
  CliArgs b = parse_cli({"--migrate-cache", "/tmp/old", "--maintain"});
  CHECK(b.migrate_from == "/tmp/old");
  CHECK(b.maintain);
}
```

- [ ] **Step 2: Run to verify it fails, then implement**

In `args.cpp`, parse the two flags. For `--migrate-cache`, consume the next argument as `migrate_from` only if one exists and doesn't start with `--`. Add both flags to `cli_usage()`.

Read `src/main.cpp` first; it was modified in the milestone-1.5 fix wave. Then make these changes:
1. **Store root:** `fx::BarStore store(args.data / "lake");` replaces `args.data / "cache"`.
2. **Early exits**, right after parsing (and after `--threads`):
```cpp
    if (args.migrate_cache) {
      fx::BarStore lake_store(args.data / "lake");
      const auto n = fx::migrate_csv_cache(args.migrate_from, lake_store);
      std::cout << "migrated " << n << " series from " << args.migrate_from.string() << " into "
                << (args.data / "lake").string() << "\n";
      return 0;
    }
    if (args.maintain) {
      fx::BarStore lake_store(args.data / "lake");
      const auto policy = fx::RetentionPolicy::load(args.data / "lake" / "retention.json");
      std::size_t compacted = 0;
      for (auto tf : {fx::Timeframe::Hour, fx::Timeframe::Day, fx::Timeframe::Week})
        compacted += lake_store.lake().compact(tf, 8);
      const auto removed = lake_store.lake().apply_retention(policy, now_utc());
      std::cout << "compacted " << compacted << " partitions, removed " << removed << " expired partitions\n";
      return 0;
    }
```
   Add `#include "storage/csv_migration.hpp"` and `#include "storage/lake.hpp"`.
3. **Windowed loads.** Replace every `store.load_all(X, tf)` with `store.load_range(X, tf, window_start, end)`, where:
   - `end = now_utc()` in replay and `now_utc() - 16*60` in alpaca.
   - `window_start = end - lookback_days*86400` for the main load, and `end - 40*86400` for the ranking load in `ensure_snapshot`.
   - `build_panel(store, universe.node_tickers(), args.tf, window_start, end)` uses the same window.
4. **After a successful alpaca sync**, call `store.flush();`, then run the same compaction and retention as `--maintain`, printing to stderr. Retention must never drop data inside the lookback. Warn if `policy.keep_days[tf] < lookback_days` and skip retention for that timeframe in that case.

In `.gitignore`, add `data/lake/`.

In `README.md`, add under `## Run`:
```bash
./build/fluxscape --migrate-cache                           # one-time: import the old data/cache CSVs into data/lake
./build/fluxscape --maintain                                # compact partitions and apply data/lake/retention.json
```
Add one sentence to "How it works", item 4: bars are stored in a DuckDB-managed, Hive-partitioned Parquet lake (`data/lake`), with month partitions and a retention policy.

- [ ] **Step 3: Verify offline**

Run: `cmake --build build -j && ./build/fluxtests` (expect SUCCESS). Then:
```bash
./build/fluxscape --migrate-cache
./build/fluxscape --mode replay --universe sp500 --timeframe 1d --top 5
./build/fluxscape --maintain
./build/fluxscape --mode replay --universe sp500 --timeframe 1d --top 5
```
Expected:
- The migration imports about 500 series.
- Both replays print the same hills as before the migration (data is lossless).
- `--maintain` reports its counts.

Paste all outputs into the report. Don't delete `data/cache`; the user decides when to.

- [ ] **Step 4: Commit**
```bash
git add src/cli src/main.cpp tests/test_cli_args.cpp .gitignore README.md
git commit -m "feat: CLI on the Parquet lake: windowed loads, --migrate-cache, --maintain, post-sync maintenance

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: First live 10K run and evaluation

**Files:**
- Modify: none, unless a defect is found. Fix defects in their own commits with tests.

This task reads market data from Alpaca using the existing `.env`. Never print the `.env` contents.

- [ ] **Step 1: Live universe and sync.** Run with a timeout of 45 minutes:
`./build/fluxscape --mode alpaca --universe-size 10000 --timeframe 1d --top 15 2>&1 | tail -80`
Expected:
- **stderr:** the asset and candidate counts, then "wrote N-ticker universe snapshot" with N near 10000, then the sync, then the maintenance summary.
- **stdout:** `solver: converged`, the floor percentage, hills, valleys and portfolio holdings.

- [ ] **Step 2: Lake sanity.** Run `du -sh data/lake`, `find data/lake/bars -name '*.parquet' | wc -l`, and `ls data/lake/bars/tf=1d | head`. Then rerun Step 1's command (the second run should be incremental and much faster) and report both wall times (`/usr/bin/time -v` or `date` before and after).

- [ ] **Step 3: 10K evaluation.** Run with a timeout of 60 minutes:
`./build/fluxscape --mode replay --timeframe 1d --eval --eval-bars 60 2>&1 | tail -15`
Expected: the 9-row table with ms/frame at N ≈ 10,000.

- [ ] **Step 4: Report.** Paste all outputs, sizes and timings into the report. If any step fails, paste the error and report DONE_WITH_CONCERNS. If the failure is a code defect, report it with your analysis; don't patch it without a test.
