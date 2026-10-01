# Fluxscape Milestone 1 — Core Model (CLI) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A C++20 command-line program that loads the S&P 500 universe plus portfolio extras, gets bars (synthetic, cached replay or Alpaca), builds the market flux graph, solves its damped Markov steady state, computes forecast scores, and prints the hottest and coldest stocks.

**Architecture:** A static library `fluxcore` (market data → flux graph → solver → forecast) plus two executables: `fluxscape` (CLI) and `fluxtests` (doctest). Each stage is a pure, separately tested unit with plain value types at its boundaries (`Bar`, `Panel`, dense flux `std::vector<double>`, `Csr`). Network access is isolated behind an injectable `HttpGet` function, so the Alpaca client is tested without the network.

**Tech Stack:** C++20, GCC 13, CMake 3.28; FetchContent: nlohmann/json 3.11.3, cpp-httplib 0.18.1 (OpenSSL), doctest 2.4.11; Python 3 stdlib for the one universe-refresh script.

**Spec:** `docs/superpowers/specs/2026-10-01-fluxscape-design.md` (this plan implements §3, §4.1–4.4, §5, §5.1 and the M1 tests in §11; holidays in §4.2 are deferred to Milestone 3, where the schedule needs them).

## Global Constraints

- Language standard C++20; must build with GCC 13.3 and CMake ≥ 3.24 (installed: 3.28.3).
- All code is in namespace `fx`. Headers use `#pragma once`.
- Third-party code only via CMake FetchContent: nlohmann/json, cpp-httplib (with `CPPHTTPLIB_OPENSSL_SUPPORT`), doctest. No other dependencies.
- Time is `fx::TimePoint` = `int64_t` seconds since the Unix epoch, UTC. US Eastern time uses the explicit DST rule (2nd Sunday of March 07:00 UTC → 1st Sunday of November 06:00 UTC); no tz database.
- Alpaca: `https://data.alpaca.markets/v2/stocks/bars`, `limit=10000`, `adjustment=all`, default `feed=sip`, `end ≤ now − 16 min`, ≤ 100 symbols per request, headers `APCA-API-KEY-ID` / `APCA-API-SECRET-KEY`, credentials from `.env` or env vars `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY`, optional `APCA_DATA_FEED`.
- Backoff on HTTP 429/5xx: 0.5 s doubling to a max of 30 s, max 6 retries.
- Flux defaults: λ = 1.0, slow half-life 20 bars, fast half-life 3 bars, correlation window 60 bars, top-k = 20, damping α = 0.85, β = 0.5, horizons k ∈ {1, 4, 8}.
- Power iteration: tolerance ‖Δπ‖₁ < 1e-10, max 1000 iterations.
- Hotness `h_i = N·π_i − 1`.
- Missing data is NaN in a `Panel`; it is never zero-filled as a price. A missing return or dollar volume gives pressure 0.
- Build: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j`. Tests: `./build/fluxtests` (one case: `./build/fluxtests -tc="<case name>"`).
- Every commit message ends with the line `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.

## File Structure

```
CMakeLists.txt                       build: fluxcore lib (GLOB src/**/*.cpp minus main), fluxscape, fluxtests
.env.example                         Alpaca credential template
README.md                            build/run instructions
scripts/fetch_sp500.py               refresh data/universe/sp500.csv from Wikipedia
data/universe/sp500.csv              ticker,name,sector (generated)
data/universe/funds.csv              ticker,tracks (VOO,sp500)
data/portfolio.json                  $1M made-up portfolio
src/core/types.hpp/.cpp              TimePoint, Bar, Timeframe
src/core/time.hpp/.cpp               civil-date math, utc_seconds, RFC 3339
src/core/csv.hpp/.cpp                quoted-CSV parsing
src/market/session_calendar.hpp/.cpp US DST, to_eastern, 30Min → session hours
src/market/universe.hpp/.cpp         Security/Fund/Universe, PortfolioSpec, extras
src/market/bar_store.hpp/.cpp        per-ticker bars + CSV disk cache
src/market/panel.hpp/.cpp            time-aligned T×N matrices (close, volume, vwap)
src/market/synthetic_market.hpp/.cpp seeded sector market with planted rotation
src/market/alpaca_client.hpp/.cpp    JSON parse, paging, retry, dotenv, HTTP transport
src/market/market_sync.hpp/.cpp      incremental fetch into BarStore
src/graph/flux_builder.hpp/.cpp      pressure, rolling correlation, per-bar flux, slow/fast accumulators
src/graph/csr.hpp/.cpp               Csr, build_transition (top-k + row-normalize + self-loop), left_multiply
src/graph/markov_solver.hpp/.cpp     damped power iteration, propagate, hotness
src/graph/forecaster.hpp/.cpp        π·P_fast^k and drift → scores
src/pipeline/core_pipeline.hpp/.cpp  Panel bar → Frame
src/main.cpp                         CLI
tests/test_main.cpp                  doctest main
tests/test_util.hpp                  temp-file helper
tests/test_*.cpp                     one per unit
```

---

### Task 1: Build scaffold, core types and time utilities

**Files:**
- Create: `CMakeLists.txt`, `src/core/types.hpp`, `src/core/types.cpp`, `src/core/time.hpp`, `src/core/time.cpp`, `src/main.cpp` (stub), `tests/test_main.cpp`, `tests/test_util.hpp`
- Test: `tests/test_time.cpp`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `fx::TimePoint` (`int64_t`), `struct fx::Bar { TimePoint t; double o, h, l, c, v, vw; }`
  - `enum class fx::Timeframe { Hour, Day, Week }`, `std::string_view fx::to_string(Timeframe)` → `"1h"|"1d"|"1w"`, `Timeframe fx::parse_timeframe(std::string_view)` (throws `std::invalid_argument`), `TimePoint fx::timeframe_seconds(Timeframe)`
  - `struct fx::Civil { int y; unsigned m; unsigned d; }`, `int64_t fx::days_from_civil(int, unsigned, unsigned)`, `Civil fx::civil_from_days(int64_t)`, `unsigned fx::weekday_from_days(int64_t)` (0 = Sunday), `int64_t fx::floor_div(int64_t, int64_t)`
  - `TimePoint fx::utc_seconds(int y, unsigned m, unsigned d, int hh = 0, int mm = 0, int ss = 0)`
  - `TimePoint fx::parse_rfc3339(std::string_view)` (UTC "Z" only; fractional seconds ignored; throws `std::invalid_argument`), `std::string fx::format_rfc3339(TimePoint)`

- [ ] **Step 1: Write the build file, the doctest main, the test helper and a stub main**

`CMakeLists.txt`:
```cmake
cmake_minimum_required(VERSION 3.24)
project(fluxscape LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(FetchContent)
set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_Declare(json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
set(HTTPLIB_REQUIRE_OPENSSL ON CACHE BOOL "" FORCE)
set(HTTPLIB_USE_BROTLI_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_USE_ZLIB_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
FetchContent_Declare(httplib
  GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
  GIT_TAG v0.18.1)
FetchContent_Declare(doctest
  GIT_REPOSITORY https://github.com/doctest/doctest.git
  GIT_TAG v2.4.11)
FetchContent_MakeAvailable(json httplib doctest)
find_package(OpenSSL REQUIRED)

file(GLOB_RECURSE FLUXCORE_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/*.cpp)
list(REMOVE_ITEM FLUXCORE_SOURCES ${CMAKE_SOURCE_DIR}/src/main.cpp)
add_library(fluxcore STATIC ${FLUXCORE_SOURCES})
target_include_directories(fluxcore PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_compile_definitions(fluxcore PUBLIC CPPHTTPLIB_OPENSSL_SUPPORT)
target_compile_options(fluxcore PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(fluxcore PUBLIC nlohmann_json::nlohmann_json httplib::httplib
                      OpenSSL::SSL OpenSSL::Crypto)

add_executable(fluxscape src/main.cpp)
target_link_libraries(fluxscape PRIVATE fluxcore)

file(GLOB FLUX_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/tests/*.cpp)
add_executable(fluxtests ${FLUX_TEST_SOURCES})
target_include_directories(fluxtests PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(fluxtests PRIVATE fluxcore doctest::doctest)

enable_testing()
add_test(NAME fluxtests COMMAND fluxtests)
```

`tests/test_main.cpp`:
```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
```

`tests/test_util.hpp`:
```cpp
#pragma once
#include <unistd.h>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <string>

namespace fx::test {

// Creates a fresh, empty directory under the system temp dir.
inline std::filesystem::path temp_dir(const std::string& tag) {
  static std::atomic<int> counter{0};
  auto dir = std::filesystem::temp_directory_path() /
             ("fluxtest_" + tag + "_" + std::to_string(::getpid()) + "_" +
              std::to_string(counter++));
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  return dir;
}

inline std::filesystem::path write_file(const std::filesystem::path& path,
                                        const std::string& content) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream(path) << content;
  return path;
}

}  // namespace fx::test
```
`src/main.cpp` (stub, replaced in Task 11):
```cpp
int main() { return 0; }
```

- [ ] **Step 2: Write the failing tests**

`tests/test_time.cpp`:
```cpp
#include <doctest/doctest.h>

#include "core/time.hpp"
#include "core/types.hpp"

using namespace fx;

TEST_CASE("civil date round trip and known values") {
  CHECK(days_from_civil(1970, 1, 1) == 0);
  CHECK(utc_seconds(2000, 1, 1) == 946684800);
  for (int64_t z = -800000; z <= 800000; z += 997) {
    Civil c = civil_from_days(z);
    CHECK(days_from_civil(c.y, c.m, c.d) == z);
  }
  CHECK(weekday_from_days(days_from_civil(2026, 10, 1)) == 4);  // Thursday
  CHECK(weekday_from_days(days_from_civil(2026, 3, 8)) == 0);   // Sunday
}

TEST_CASE("floor_div rounds toward negative infinity") {
  CHECK(floor_div(7, 2) == 3);
  CHECK(floor_div(-7, 2) == -4);
  CHECK(floor_div(-8, 2) == -4);
}

TEST_CASE("rfc3339 parse and format") {
  CHECK(parse_rfc3339("2026-09-30T13:30:00Z") == utc_seconds(2026, 9, 30, 13, 30));
  CHECK(parse_rfc3339("2026-09-30T13:30:00.123Z") == utc_seconds(2026, 9, 30, 13, 30));
  CHECK(format_rfc3339(utc_seconds(2026, 1, 2, 3, 4, 5)) == "2026-01-02T03:04:05Z");
  CHECK_THROWS_AS(parse_rfc3339("2026-09-30T13:30:00+02:00"), std::invalid_argument);
  CHECK_THROWS_AS(parse_rfc3339("garbage"), std::invalid_argument);
}

TEST_CASE("timeframe strings") {
  CHECK(parse_timeframe("1h") == Timeframe::Hour);
  CHECK(parse_timeframe("1d") == Timeframe::Day);
  CHECK(parse_timeframe("1w") == Timeframe::Week);
  CHECK(to_string(Timeframe::Week) == "1w");
  CHECK(timeframe_seconds(Timeframe::Day) == 86400);
  CHECK_THROWS_AS(parse_timeframe("5m"), std::invalid_argument);
}
```

- [ ] **Step 3: Run the build to verify it fails**

Run: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j`
Expected: FAIL. CMake reports no sources for `fluxcore`, or the compiler reports `core/time.hpp: No such file or directory`. (The first configure downloads the dependencies, which takes about a minute.)

- [ ] **Step 4: Implement types and time**

`src/core/types.hpp`:
```cpp
#pragma once
#include <cstdint>
#include <string_view>

namespace fx {

using TimePoint = std::int64_t;  // seconds since Unix epoch, UTC

struct Bar {
  TimePoint t = 0;  // bar start
  double o = 0, h = 0, l = 0, c = 0, v = 0, vw = 0;
};

enum class Timeframe { Hour, Day, Week };

std::string_view to_string(Timeframe tf);
Timeframe parse_timeframe(std::string_view s);
TimePoint timeframe_seconds(Timeframe tf);

}  // namespace fx
```

`src/core/types.cpp`:
```cpp
#include "core/types.hpp"

#include <stdexcept>
#include <string>

namespace fx {

std::string_view to_string(Timeframe tf) {
  switch (tf) {
    case Timeframe::Hour: return "1h";
    case Timeframe::Day: return "1d";
    case Timeframe::Week: return "1w";
  }
  return "?";
}

Timeframe parse_timeframe(std::string_view s) {
  if (s == "1h") return Timeframe::Hour;
  if (s == "1d") return Timeframe::Day;
  if (s == "1w") return Timeframe::Week;
  throw std::invalid_argument("unknown timeframe: " + std::string(s));
}

TimePoint timeframe_seconds(Timeframe tf) {
  switch (tf) {
    case Timeframe::Hour: return 3600;
    case Timeframe::Day: return 86400;
    case Timeframe::Week: return 7 * 86400;
  }
  return 0;
}

}  // namespace fx
```

`src/core/time.hpp`:
```cpp
#pragma once
#include <cstdint>
#include <string>
#include <string_view>

#include "core/types.hpp"

namespace fx {

struct Civil {
  int y;
  unsigned m;
  unsigned d;
};

std::int64_t floor_div(std::int64_t a, std::int64_t b);
std::int64_t days_from_civil(int y, unsigned m, unsigned d);
Civil civil_from_days(std::int64_t z);
unsigned weekday_from_days(std::int64_t z);  // 0 = Sunday
TimePoint utc_seconds(int y, unsigned m, unsigned d, int hh = 0, int mm = 0, int ss = 0);
TimePoint parse_rfc3339(std::string_view s);
std::string format_rfc3339(TimePoint t);

}  // namespace fx
```

`src/core/time.cpp`:
```cpp
#include "core/time.hpp"

#include <cstdio>
#include <stdexcept>

namespace fx {

std::int64_t floor_div(std::int64_t a, std::int64_t b) {
  std::int64_t q = a / b;
  if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
  return q;
}

// Howard Hinnant's civil-date algorithms (proleptic Gregorian).
std::int64_t days_from_civil(int y, unsigned m, unsigned d) {
  y -= m <= 2;
  const std::int64_t era = (y >= 0 ? y : y - 399) / 400;
  const unsigned yoe = static_cast<unsigned>(y - era * 400);
  const unsigned doy = (153 * (m > 2 ? m - 3 : m + 9) + 2) / 5 + d - 1;
  const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
  return era * 146097 + static_cast<std::int64_t>(doe) - 719468;
}

Civil civil_from_days(std::int64_t z) {
  z += 719468;
  const std::int64_t era = (z >= 0 ? z : z - 146096) / 146097;
  const unsigned doe = static_cast<unsigned>(z - era * 146097);
  const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
  const std::int64_t y = static_cast<std::int64_t>(yoe) + era * 400;
  const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  const unsigned mp = (5 * doy + 2) / 153;
  const unsigned d = doy - (153 * mp + 2) / 5 + 1;
  const unsigned m = mp < 10 ? mp + 3 : mp - 9;
  return {static_cast<int>(y + (m <= 2)), m, d};
}

unsigned weekday_from_days(std::int64_t z) {
  return static_cast<unsigned>(((z % 7) + 11) % 7);  // 1970-01-01 was a Thursday (4)
}

TimePoint utc_seconds(int y, unsigned m, unsigned d, int hh, int mm, int ss) {
  return days_from_civil(y, m, d) * 86400 + hh * 3600 + mm * 60 + ss;
}

TimePoint parse_rfc3339(std::string_view s) {
  int y = 0, mo = 0, d = 0, hh = 0, mi = 0, ss = 0;
  std::string str(s);
  if (str.empty() || str.back() != 'Z' ||
      std::sscanf(str.c_str(), "%4d-%2d-%2dT%2d:%2d:%2d", &y, &mo, &d, &hh, &mi, &ss) != 6) {
    throw std::invalid_argument("not an RFC 3339 UTC timestamp: " + str);
  }
  return utc_seconds(y, static_cast<unsigned>(mo), static_cast<unsigned>(d), hh, mi, ss);
}

std::string format_rfc3339(TimePoint t) {
  const std::int64_t days = floor_div(t, 86400);
  const std::int64_t secs = t - days * 86400;
  const Civil c = civil_from_days(days);
  char buf[32];
  std::snprintf(buf, sizeof buf, "%04d-%02u-%02uT%02d:%02d:%02dZ", c.y, c.m, c.d,
                static_cast<int>(secs / 3600), static_cast<int>(secs % 3600 / 60),
                static_cast<int>(secs % 60));
  return buf;
}

}  // namespace fx
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cmake --build build -j && ./build/fluxtests`
Expected: `[doctest] Status: SUCCESS!` with 4 test cases.

- [ ] **Step 6: Commit**

```bash
git add CMakeLists.txt src tests
git commit -m "feat: build scaffold, core types and UTC time utilities

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Session calendar (US Eastern DST, session-hour bars)

**Files:**
- Create: `src/market/session_calendar.hpp`, `src/market/session_calendar.cpp`
- Test: `tests/test_session_calendar.cpp`

**Interfaces:**
- Consumes: `Bar`, `TimePoint`, `Civil`, `days_from_civil`, `civil_from_days`, `weekday_from_days`, `floor_div`, `utc_seconds` (Task 1).
- Produces:
  - `bool fx::is_us_dst(TimePoint utc)`
  - `struct fx::EtTime { Civil date; int hour; int minute; unsigned weekday; }`, `EtTime fx::to_eastern(TimePoint utc)`
  - `std::vector<Bar> fx::aggregate_session_hours(const std::vector<Bar>& bars30m)`: input sorted by `t`. Output has one bar per (ET date, session hour). The hours are 09:30–10:30, …, 14:30–15:30, plus 15:30–16:00, so there are 7 buckets. Each output bar's `t` is the bucket start. Bars outside 09:30–16:00 ET are dropped.

- [ ] **Step 1: Write the failing tests**

`tests/test_session_calendar.cpp`:
```cpp
#include <doctest/doctest.h>

#include "core/time.hpp"
#include "market/session_calendar.hpp"

using namespace fx;

TEST_CASE("US DST boundaries 2026") {
  // DST starts Sunday 2026-03-08 02:00 EST = 07:00 UTC
  CHECK_FALSE(is_us_dst(utc_seconds(2026, 3, 8, 6, 59)));
  CHECK(is_us_dst(utc_seconds(2026, 3, 8, 7, 0)));
  // DST ends Sunday 2026-11-01 02:00 EDT = 06:00 UTC
  CHECK(is_us_dst(utc_seconds(2026, 11, 1, 5, 59)));
  CHECK_FALSE(is_us_dst(utc_seconds(2026, 11, 1, 6, 0)));
  CHECK_FALSE(is_us_dst(utc_seconds(2026, 1, 15, 12, 0)));
  CHECK(is_us_dst(utc_seconds(2026, 7, 15, 12, 0)));
}

TEST_CASE("to_eastern converts across DST") {
  EtTime a = to_eastern(utc_seconds(2026, 3, 9, 13, 30));  // EDT
  CHECK(a.hour == 9);
  CHECK(a.minute == 30);
  CHECK(a.date.d == 9);
  CHECK(a.weekday == 1);  // Monday
  EtTime b = to_eastern(utc_seconds(2026, 3, 6, 14, 30));  // EST
  CHECK(b.hour == 9);
  CHECK(b.minute == 30);
  EtTime c = to_eastern(utc_seconds(2026, 3, 7, 2, 0));  // previous ET evening
  CHECK(c.date.d == 6);
  CHECK(c.hour == 21);
}

TEST_CASE("30Min bars aggregate into session-aligned hours") {
  // 2026-09-30 is EDT (UTC-4). ET 09:00 (pre-market), 09:30, 10:00, 10:30, 15:30.
  std::vector<Bar> in = {
      {utc_seconds(2026, 9, 30, 13, 0), 9, 9, 9, 9, 100, 9},
      {utc_seconds(2026, 9, 30, 13, 30), 10, 12, 9.5, 11, 1000, 10.5},
      {utc_seconds(2026, 9, 30, 14, 0), 11, 13, 10, 12, 3000, 12.5},
      {utc_seconds(2026, 9, 30, 14, 30), 12, 12.5, 11.5, 12, 500, 12},
      {utc_seconds(2026, 9, 30, 19, 30), 20, 21, 19, 20.5, 800, 20},
  };
  auto out = aggregate_session_hours(in);
  REQUIRE(out.size() == 3);
  CHECK(out[0].t == utc_seconds(2026, 9, 30, 13, 30));
  CHECK(out[0].o == 10);
  CHECK(out[0].h == 13);
  CHECK(out[0].l == 9.5);
  CHECK(out[0].c == 12);
  CHECK(out[0].v == 4000);
  CHECK(out[0].vw == doctest::Approx((10.5 * 1000 + 12.5 * 3000) / 4000));
  CHECK(out[1].t == utc_seconds(2026, 9, 30, 14, 30));
  CHECK(out[1].v == 500);
  CHECK(out[2].t == utc_seconds(2026, 9, 30, 19, 30));
  CHECK(out[2].c == 20.5);
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cmake --build build -j`
Expected: FAIL with `market/session_calendar.hpp: No such file or directory`.

- [ ] **Step 3: Implement**

`src/market/session_calendar.hpp`:
```cpp
#pragma once
#include <vector>

#include "core/time.hpp"
#include "core/types.hpp"

namespace fx {

bool is_us_dst(TimePoint utc);

struct EtTime {
  Civil date;
  int hour;
  int minute;
  unsigned weekday;  // 0 = Sunday
};

EtTime to_eastern(TimePoint utc);

// Session hours 09:30-10:30 ... 14:30-15:30 and 15:30-16:00 ET (7 buckets).
std::vector<Bar> aggregate_session_hours(const std::vector<Bar>& bars30m);

}  // namespace fx
```

`src/market/session_calendar.cpp`:
```cpp
#include "market/session_calendar.hpp"

#include <algorithm>

namespace fx {
namespace {

// Day of month of the n-th Sunday (n >= 1) of the given month.
unsigned nth_sunday(int y, unsigned m, unsigned n) {
  const unsigned w = weekday_from_days(days_from_civil(y, m, 1));
  return 1 + (7 - w) % 7 + 7 * (n - 1);
}

constexpr int kSessionOpenMin = 9 * 60 + 30;
constexpr int kSessionCloseMin = 16 * 60;

}  // namespace

bool is_us_dst(TimePoint utc) {
  const int y = civil_from_days(floor_div(utc, 86400)).y;
  const TimePoint start = utc_seconds(y, 3, nth_sunday(y, 3, 2), 7, 0);
  const TimePoint end = utc_seconds(y, 11, nth_sunday(y, 11, 1), 6, 0);
  return utc >= start && utc < end;
}

EtTime to_eastern(TimePoint utc) {
  const TimePoint local = utc + (is_us_dst(utc) ? -4 : -5) * 3600;
  const std::int64_t days = floor_div(local, 86400);
  const std::int64_t secs = local - days * 86400;
  return {civil_from_days(days), static_cast<int>(secs / 3600),
          static_cast<int>(secs % 3600 / 60), weekday_from_days(days)};
}

std::vector<Bar> aggregate_session_hours(const std::vector<Bar>& bars30m) {
  std::vector<Bar> out;
  std::int64_t cur_day = 0;
  int cur_bucket = -1;
  double vw_num = 0;
  for (const Bar& b : bars30m) {
    const EtTime et = to_eastern(b.t);
    const int minute_of_day = et.hour * 60 + et.minute;
    if (minute_of_day < kSessionOpenMin || minute_of_day >= kSessionCloseMin) continue;
    const int bucket = (minute_of_day - kSessionOpenMin) / 60;
    const std::int64_t day = days_from_civil(et.date.y, et.date.m, et.date.d);
    const double bar_vw = b.vw > 0 ? b.vw : b.c;
    if (out.empty() || day != cur_day || bucket != cur_bucket) {
      if (!out.empty()) out.back().vw = out.back().v > 0 ? vw_num / out.back().v : out.back().c;
      const int offset_min = minute_of_day - (kSessionOpenMin + bucket * 60);
      out.push_back({b.t - offset_min * 60, b.o, b.h, b.l, b.c, b.v, 0});
      vw_num = bar_vw * b.v;
      cur_day = day;
      cur_bucket = bucket;
    } else {
      Bar& a = out.back();
      a.h = std::max(a.h, b.h);
      a.l = std::min(a.l, b.l);
      a.c = b.c;
      a.v += b.v;
      vw_num += bar_vw * b.v;
    }
  }
  if (!out.empty()) out.back().vw = out.back().v > 0 ? vw_num / out.back().v : out.back().c;
  return out;
}

}  // namespace fx
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cmake --build build -j && ./build/fluxtests`
Expected: `Status: SUCCESS!` (7 test cases).

- [ ] **Step 5: Commit**

```bash
git add src/market/session_calendar.* tests/test_session_calendar.cpp
git commit -m "feat: US Eastern session calendar and session-hour aggregation

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: CSV parsing, universe, portfolio and data files

**Files:**
- Create: `src/core/csv.hpp`, `src/core/csv.cpp`, `src/market/universe.hpp`, `src/market/universe.cpp`, `scripts/fetch_sp500.py`, `data/universe/funds.csv`, `data/portfolio.json`, `data/universe/sp500.csv` (generated by the script)
- Test: `tests/test_universe.cpp`

**Interfaces:**
- Consumes: `tests/test_util.hpp` (Task 1).
- Produces:
  - `using fx::CsvRows = std::vector<std::vector<std::string>>`, `CsvRows fx::parse_csv(std::string_view)`, `CsvRows fx::read_csv_file(const std::filesystem::path&)` (throws `std::runtime_error` if the file can't be opened)
  - `struct fx::Security { std::string ticker, name, sector; }`, `struct fx::Fund { std::string ticker, tracks; }`, `struct fx::Holding { std::string ticker; double weight; }`, `struct fx::PortfolioSpec { double initial_cash; std::string inception; std::vector<Holding> holdings; }`
  - `PortfolioSpec fx::load_portfolio(const std::filesystem::path&)` (throws `std::runtime_error` unless the weights sum to 1 within 1e-6)
  - `class fx::Universe` with `static Universe load(const path& sp500_csv, const path& funds_csv)`, `static Universe from_securities(std::vector<Security>)`, `void add_extras(const PortfolioSpec&)`, `const std::vector<Security>& nodes() const`, `const std::vector<Fund>& funds() const`, `std::optional<std::size_t> index_of(std::string_view) const`, `bool is_fund(std::string_view) const`, `std::vector<std::string> node_tickers() const`, `std::vector<std::string> price_tickers() const` (the node tickers plus the fund tickers)

- [ ] **Step 1: Write the failing tests**

`tests/test_universe.cpp`:
```cpp
#include <doctest/doctest.h>

#include "core/csv.hpp"
#include "market/universe.hpp"
#include "test_util.hpp"

using namespace fx;

TEST_CASE("parse_csv handles quotes, escaped quotes, CRLF and blank lines") {
  auto rows = parse_csv("a,b,c\r\n\"x, y\",\"say \"\"hi\"\"\",z\n\nlast,,\n");
  REQUIRE(rows.size() == 3);
  CHECK(rows[1][0] == "x, y");
  CHECK(rows[1][1] == "say \"hi\"");
  CHECK(rows[2].size() == 3);
  CHECK(rows[2][1] == "");
}

TEST_CASE("universe loads, adds non-fund extras, indexes tickers") {
  auto dir = test::temp_dir("universe");
  auto sp = test::write_file(dir / "sp500.csv",
                             "ticker,name,sector\nAAPL,Apple Inc.,Information Technology\n"
                             "NKE,\"Nike, Inc.\",Consumer Discretionary\nF,Ford Motor Company,"
                             "Consumer Discretionary\n");
  auto fu = test::write_file(dir / "funds.csv", "ticker,tracks\nVOO,sp500\n");
  Universe u = Universe::load(sp, fu);
  CHECK(u.nodes().size() == 3);
  CHECK(u.nodes()[1].name == "Nike, Inc.");

  PortfolioSpec p{1e6, "2025-10-01", {{"AAPL", 0.6}, {"VOO", 0.15}, {"NVO", 0.25}}};
  u.add_extras(p);
  REQUIRE(u.nodes().size() == 4);
  CHECK(u.nodes()[3].ticker == "NVO");
  CHECK(u.nodes()[3].sector == "Extra");
  CHECK(u.index_of("NVO").value() == 3);
  CHECK_FALSE(u.index_of("VOO").has_value());
  CHECK(u.is_fund("VOO"));
  CHECK(u.price_tickers().size() == 5);
  u.add_extras(p);  // idempotent
  CHECK(u.nodes().size() == 4);
}

TEST_CASE("load_portfolio validates weights") {
  auto dir = test::temp_dir("portfolio");
  auto ok = test::write_file(dir / "ok.json",
                             R"({"initial_cash":1000000,"inception":"2025-10-01",
                                 "holdings":[{"ticker":"AAPL","weight":0.6},
                                             {"ticker":"VOO","weight":0.4}]})");
  PortfolioSpec p = load_portfolio(ok);
  CHECK(p.initial_cash == 1000000);
  CHECK(p.holdings.size() == 2);
  auto bad = test::write_file(dir / "bad.json",
                              R"({"initial_cash":1,"inception":"x",
                                  "holdings":[{"ticker":"AAPL","weight":0.6}]})");
  CHECK_THROWS_AS(load_portfolio(bad), std::runtime_error);
}

TEST_CASE("bundled data files are consistent") {
  Universe u = Universe::load(FLUX_SOURCE_DIR "/data/universe/sp500.csv",
                              FLUX_SOURCE_DIR "/data/universe/funds.csv");
  CHECK(u.nodes().size() >= 490);
  CHECK(u.nodes().size() <= 510);
  CHECK(u.index_of("AAPL").has_value());
  PortfolioSpec p = load_portfolio(FLUX_SOURCE_DIR "/data/portfolio.json");
  u.add_extras(p);
  CHECK(u.index_of("NVO").has_value());
  CHECK(u.is_fund("VOO"));
}
```

Add one line to `CMakeLists.txt` after `target_link_libraries(fluxtests ...)`:
```cmake
target_compile_definitions(fluxtests PRIVATE FLUX_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cmake --build build -j`
Expected: FAIL with `core/csv.hpp: No such file or directory`.

- [ ] **Step 3: Implement CSV and universe**

`src/core/csv.hpp`:
```cpp
#pragma once
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace fx {

using CsvRows = std::vector<std::vector<std::string>>;

CsvRows parse_csv(std::string_view text);
CsvRows read_csv_file(const std::filesystem::path& path);

}  // namespace fx
```

`src/core/csv.cpp`:
```cpp
#include "core/csv.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace fx {

CsvRows parse_csv(std::string_view text) {
  CsvRows rows;
  std::vector<std::string> row;
  std::string field;
  bool in_quotes = false;
  bool row_has_content = false;
  auto end_row = [&] {
    if (row_has_content) {
      row.push_back(field);
      rows.push_back(row);
    }
    row.clear();
    field.clear();
    row_has_content = false;
  };
  for (std::size_t i = 0; i < text.size(); ++i) {
    const char ch = text[i];
    if (in_quotes) {
      if (ch == '"' && i + 1 < text.size() && text[i + 1] == '"') {
        field += '"';
        ++i;
      } else if (ch == '"') {
        in_quotes = false;
      } else {
        field += ch;
      }
      continue;
    }
    if (ch == '"') {
      in_quotes = true;
      row_has_content = true;
    } else if (ch == ',') {
      row.push_back(field);
      field.clear();
      row_has_content = true;
    } else if (ch == '\n') {
      end_row();
    } else if (ch != '\r') {
      field += ch;
      row_has_content = true;
    }
  }
  end_row();
  return rows;
}

CsvRows read_csv_file(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path.string());
  std::stringstream ss;
  ss << in.rdbuf();
  return parse_csv(ss.str());
}

}  // namespace fx
```

`src/market/universe.hpp`:
```cpp
#pragma once
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace fx {

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

class Universe {
 public:
  static Universe load(const std::filesystem::path& sp500_csv,
                       const std::filesystem::path& funds_csv);
  static Universe from_securities(std::vector<Security> securities);

  // Adds held single stocks that are not yet nodes (sector "Extra"). Funds are skipped.
  void add_extras(const PortfolioSpec& portfolio);

  const std::vector<Security>& nodes() const { return nodes_; }
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

}  // namespace fx
```

`src/market/universe.cpp`:
```cpp
#include "market/universe.hpp"

#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

#include "core/csv.hpp"

namespace fx {

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

Universe Universe::load(const std::filesystem::path& sp500_csv,
                        const std::filesystem::path& funds_csv) {
  Universe u;
  const CsvRows sp = read_csv_file(sp500_csv);
  for (std::size_t r = 1; r < sp.size(); ++r) {
    if (sp[r].size() < 3) continue;
    u.add_node({sp[r][0], sp[r][1], sp[r][2]});
  }
  const CsvRows fu = read_csv_file(funds_csv);
  for (std::size_t r = 1; r < fu.size(); ++r) {
    if (fu[r].size() < 2) continue;
    u.funds_.push_back({fu[r][0], fu[r][1]});
  }
  return u;
}

Universe Universe::from_securities(std::vector<Security> securities) {
  Universe u;
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
  for (const auto& f : funds_) out.push_back(f.ticker);
  return out;
}

}  // namespace fx
```

- [ ] **Step 4: Write the data files and the refresh script**

`data/universe/funds.csv`:
```
ticker,tracks
VOO,sp500
```

`data/portfolio.json`:
```json
{
  "initial_cash": 1000000,
  "inception": "2025-10-01",
  "holdings": [
    {"ticker": "AAPL", "weight": 0.60},
    {"ticker": "VOO",  "weight": 0.15},
    {"ticker": "NVDA", "weight": 0.07},
    {"ticker": "LLY",  "weight": 0.06},
    {"ticker": "NVO",  "weight": 0.05},
    {"ticker": "NKE",  "weight": 0.035},
    {"ticker": "F",    "weight": 0.035}
  ]
}
```

`scripts/fetch_sp500.py`:
```python
#!/usr/bin/env python3
"""Refresh data/universe/sp500.csv (ticker,name,sector) from Wikipedia's constituents table."""
import csv
import sys
import urllib.request
from html.parser import HTMLParser

URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


class ConstituentsParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.depth = 0  # table nesting depth inside the constituents table
        self.rows, self.row, self.cell = [], None, None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            if self.depth > 0:
                self.depth += 1
            elif dict(attrs).get("id") == "constituents":
                self.depth = 1
        elif self.depth == 1 and tag == "tr":
            self.row = []
        elif self.depth == 1 and tag in ("td", "th") and self.row is not None:
            self.cell = []

    def handle_endtag(self, tag):
        if tag == "table" and self.depth > 0:
            self.depth -= 1
        elif self.depth == 1 and tag in ("td", "th") and self.cell is not None:
            self.row.append("".join(self.cell).strip())
            self.cell = None
        elif self.depth == 1 and tag == "tr" and self.row is not None:
            if self.row:
                self.rows.append(self.row)
            self.row = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)


def main(out_path):
    req = urllib.request.Request(URL, headers={"User-Agent": "fluxscape/0.1 (universe refresh)"})
    html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8")
    parser = ConstituentsParser()
    parser.feed(html)
    header, *body = parser.rows
    i_sym, i_name, i_sector = (header.index("Symbol"), header.index("Security"),
                               header.index("GICS Sector"))
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "name", "sector"])
        for r in body:
            w.writerow([r[i_sym], r[i_name], r[i_sector]])
    print(f"wrote {len(body)} constituents to {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/universe/sp500.csv")
```

Run: `python3 scripts/fetch_sp500.py`
Expected: `wrote 50x constituents to data/universe/sp500.csv` (about 503). Then `head -3 data/universe/sp500.csv` shows `ticker,name,sector` followed by rows such as `MMM,3M,Industrials`.
If there's no network, write `data/universe/sp500.csv` by hand with the same header and at least these rows so the app still runs, and tell the user the bundled-data test will fail until the script is run: `AAPL,Apple Inc.,Information Technology`, `NVDA,Nvidia,Information Technology`, `LLY,Lilly (Eli),Health Care`, `NKE,"Nike, Inc.",Consumer Discretionary`, `F,Ford Motor Company,Consumer Discretionary`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cmake -S . -B build && cmake --build build -j && ./build/fluxtests`
Expected: `Status: SUCCESS!` (11 test cases).

- [ ] **Step 6: Commit**

```bash
git add CMakeLists.txt src/core/csv.* src/market/universe.* scripts data/universe data/portfolio.json tests/test_universe.cpp
git commit -m "feat: universe, portfolio spec and S&P 500 data

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Bar store (disk cache) and time-aligned panel

**Files:**
- Create: `src/market/bar_store.hpp`, `src/market/bar_store.cpp`, `src/market/panel.hpp`, `src/market/panel.cpp`
- Test: `tests/test_bar_store.cpp`

**Interfaces:**
- Consumes: `Bar`, `Timeframe`, `to_string(Timeframe)` (Task 1); `test::temp_dir` (Task 1).
- Produces:
  - `class fx::BarStore` with `explicit BarStore(std::filesystem::path cache_dir)`, `void merge(const std::string& ticker, Timeframe, const std::vector<Bar>&)` (bars are kept sorted with unique `t`, and an incoming bar replaces an existing one with the same `t`), `const std::vector<Bar>& bars(const std::string&, Timeframe) const` (empty if unknown), `std::optional<TimePoint> last_time(const std::string&, Timeframe) const`, `void save(const std::string&, Timeframe) const`, `void load_all(const std::vector<std::string>& tickers, Timeframe)` (files that don't exist are skipped). The cache file is `<cache_dir>/<1h|1d|1w>/<ticker>.csv` with header `t,o,h,l,c,v,vw`.
  - `struct fx::Panel { std::vector<TimePoint> times; std::vector<std::string> tickers; std::vector<double> close, volume, vwap; std::size_t T() const; std::size_t N() const; std::size_t idx(std::size_t t, std::size_t i) const; }`: row-major `[t * N + i]`, NaN where missing.
  - `Panel fx::build_panel(const BarStore&, const std::vector<std::string>& tickers, Timeframe)`: `times` is the sorted union of all the tickers' bar times.

- [ ] **Step 1: Write the failing tests**

`tests/test_bar_store.cpp`:
```cpp
#include <doctest/doctest.h>

#include <cmath>

#include "market/bar_store.hpp"
#include "market/panel.hpp"
#include "test_util.hpp"

using namespace fx;

TEST_CASE("merge keeps bars sorted, unique, newest wins") {
  BarStore s(test::temp_dir("bars_merge"));
  s.merge("AAPL", Timeframe::Day, {{200, 1, 1, 1, 2, 10, 2}, {100, 1, 1, 1, 1, 10, 1}});
  s.merge("AAPL", Timeframe::Day, {{200, 1, 1, 1, 5, 10, 5}, {300, 1, 1, 1, 3, 10, 3}});
  const auto& b = s.bars("AAPL", Timeframe::Day);
  REQUIRE(b.size() == 3);
  CHECK(b[0].t == 100);
  CHECK(b[1].c == 5);
  CHECK(b[2].t == 300);
  CHECK(s.last_time("AAPL", Timeframe::Day).value() == 300);
  CHECK_FALSE(s.last_time("MSFT", Timeframe::Day).has_value());
  CHECK(s.bars("MSFT", Timeframe::Day).empty());
}

TEST_CASE("save and load round trip") {
  auto dir = test::temp_dir("bars_io");
  {
    BarStore s(dir);
    s.merge("BRK.B", Timeframe::Hour, {{1759239000, 470.25, 471.5, 469.0, 470.75, 123456, 470.4}});
    s.save("BRK.B", Timeframe::Hour);
  }
  BarStore s2(dir);
  s2.load_all({"BRK.B", "MISSING"}, Timeframe::Hour);
  const auto& b = s2.bars("BRK.B", Timeframe::Hour);
  REQUIRE(b.size() == 1);
  CHECK(b[0].t == 1759239000);
  CHECK(b[0].c == doctest::Approx(470.75));
  CHECK(b[0].v == doctest::Approx(123456));
  CHECK(b[0].vw == doctest::Approx(470.4));
}

TEST_CASE("panel aligns tickers on the union of times with NaN gaps") {
  BarStore s(test::temp_dir("panel"));
  s.merge("A", Timeframe::Day, {{100, 0, 0, 0, 10, 5, 10}, {200, 0, 0, 0, 11, 6, 11}});
  s.merge("B", Timeframe::Day, {{200, 0, 0, 0, 20, 7, 20}, {300, 0, 0, 0, 21, 8, 21}});
  Panel p = build_panel(s, {"A", "B"}, Timeframe::Day);
  REQUIRE(p.T() == 3);
  REQUIRE(p.N() == 2);
  CHECK(p.times == std::vector<TimePoint>{100, 200, 300});
  CHECK(p.close[p.idx(0, 0)] == 10);
  CHECK(std::isnan(p.close[p.idx(0, 1)]));
  CHECK(p.close[p.idx(1, 1)] == 20);
  CHECK(std::isnan(p.volume[p.idx(2, 0)]));
  CHECK(p.vwap[p.idx(2, 1)] == 21);
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cmake --build build -j`
Expected: FAIL with `market/bar_store.hpp: No such file or directory`.

- [ ] **Step 3: Implement**

`src/market/bar_store.hpp`:
```cpp
#pragma once
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/types.hpp"

namespace fx {

class BarStore {
 public:
  explicit BarStore(std::filesystem::path cache_dir);

  void merge(const std::string& ticker, Timeframe tf, const std::vector<Bar>& incoming);
  const std::vector<Bar>& bars(const std::string& ticker, Timeframe tf) const;
  std::optional<TimePoint> last_time(const std::string& ticker, Timeframe tf) const;
  void save(const std::string& ticker, Timeframe tf) const;
  void load_all(const std::vector<std::string>& tickers, Timeframe tf);

 private:
  std::filesystem::path file_for(const std::string& ticker, Timeframe tf) const;
  std::filesystem::path cache_dir_;
  std::map<std::pair<std::string, Timeframe>, std::vector<Bar>> series_;
};

}  // namespace fx
```

`src/market/bar_store.cpp`:
```cpp
#include "market/bar_store.hpp"

#include <fstream>
#include <iomanip>

#include "core/csv.hpp"

namespace fx {

BarStore::BarStore(std::filesystem::path cache_dir) : cache_dir_(std::move(cache_dir)) {}

void BarStore::merge(const std::string& ticker, Timeframe tf, const std::vector<Bar>& incoming) {
  auto& series = series_[{ticker, tf}];
  std::map<TimePoint, Bar> by_time;
  for (const Bar& b : series) by_time[b.t] = b;
  for (const Bar& b : incoming) by_time[b.t] = b;
  series.clear();
  series.reserve(by_time.size());
  for (const auto& [t, b] : by_time) series.push_back(b);
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

std::filesystem::path BarStore::file_for(const std::string& ticker, Timeframe tf) const {
  return cache_dir_ / std::string(to_string(tf)) / (ticker + ".csv");
}

void BarStore::save(const std::string& ticker, Timeframe tf) const {
  const auto path = file_for(ticker, tf);
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path);
  out << "t,o,h,l,c,v,vw\n" << std::setprecision(15);
  for (const Bar& b : bars(ticker, tf)) {
    out << b.t << ',' << b.o << ',' << b.h << ',' << b.l << ',' << b.c << ',' << b.v << ','
        << b.vw << '\n';
  }
}

void BarStore::load_all(const std::vector<std::string>& tickers, Timeframe tf) {
  for (const auto& ticker : tickers) {
    const auto path = file_for(ticker, tf);
    if (!std::filesystem::exists(path)) continue;
    const CsvRows rows = read_csv_file(path);
    std::vector<Bar> loaded;
    for (std::size_t r = 1; r < rows.size(); ++r) {
      const auto& f = rows[r];
      if (f.size() < 7) continue;
      loaded.push_back({std::stoll(f[0]), std::stod(f[1]), std::stod(f[2]), std::stod(f[3]),
                        std::stod(f[4]), std::stod(f[5]), std::stod(f[6])});
    }
    merge(ticker, tf, loaded);
  }
}

}  // namespace fx
```

`src/market/panel.hpp`:
```cpp
#pragma once
#include <string>
#include <vector>

#include "core/types.hpp"
#include "market/bar_store.hpp"

namespace fx {

struct Panel {
  std::vector<TimePoint> times;
  std::vector<std::string> tickers;
  std::vector<double> close, volume, vwap;  // row-major [t * N + i], NaN = missing

  std::size_t T() const { return times.size(); }
  std::size_t N() const { return tickers.size(); }
  std::size_t idx(std::size_t t, std::size_t i) const { return t * tickers.size() + i; }
};

Panel build_panel(const BarStore& store, const std::vector<std::string>& tickers, Timeframe tf);

}  // namespace fx
```

`src/market/panel.cpp`:
```cpp
#include "market/panel.hpp"

#include <algorithm>
#include <limits>
#include <unordered_map>

namespace fx {

Panel build_panel(const BarStore& store, const std::vector<std::string>& tickers, Timeframe tf) {
  Panel p;
  p.tickers = tickers;
  for (const auto& ticker : tickers)
    for (const Bar& b : store.bars(ticker, tf)) p.times.push_back(b.t);
  std::sort(p.times.begin(), p.times.end());
  p.times.erase(std::unique(p.times.begin(), p.times.end()), p.times.end());

  std::unordered_map<TimePoint, std::size_t> row_of;
  for (std::size_t t = 0; t < p.times.size(); ++t) row_of[p.times[t]] = t;

  const double nan = std::numeric_limits<double>::quiet_NaN();
  const std::size_t cells = p.T() * p.N();
  p.close.assign(cells, nan);
  p.volume.assign(cells, nan);
  p.vwap.assign(cells, nan);
  for (std::size_t i = 0; i < tickers.size(); ++i) {
    for (const Bar& b : store.bars(tickers[i], tf)) {
      const std::size_t k = p.idx(row_of[b.t], i);
      p.close[k] = b.c;
      p.volume[k] = b.v;
      p.vwap[k] = b.vw > 0 ? b.vw : b.c;
    }
  }
  return p;
}

}  // namespace fx
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cmake --build build -j && ./build/fluxtests`
Expected: `Status: SUCCESS!` (14 test cases).

- [ ] **Step 5: Commit**

```bash
git add src/market/bar_store.* src/market/panel.* tests/test_bar_store.cpp
git commit -m "feat: bar store with CSV cache and aligned panel

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Synthetic market with planted rotation

**Files:**
- Create: `src/market/synthetic_market.hpp`, `src/market/synthetic_market.cpp`
- Test: `tests/test_synthetic_market.cpp`

**Interfaces:**
- Consumes: `BarStore`, `Security`, `Timeframe`, `timeframe_seconds`, `utc_seconds`, `build_panel` (Tasks 1, 3, 4).
- Produces:
  - `struct fx::SyntheticConfig { int sectors = 5; int per_sector = 10; int bars = 300; std::uint64_t seed = 42; int rotation_from = 0; int rotation_to = 1; int rotation_start = 150; double rotation_strength = 0.004; Timeframe tf = Timeframe::Day; TimePoint start = utc_seconds(2025, 1, 2, 21, 0); }`
  - `std::vector<Security> fx::generate_synthetic(const SyntheticConfig&, BarStore&)`: tickers are `"S<s>_<kk>"` (e.g. `S1_03`) with sector `"Sector<s>"`.
  - The model: a sector factor `f_s ~ N(0, 0.01)` and noise `ε ~ N(0, 0.008)` give `r = 0.6·f_s + ε`. From bar `rotation_start` onward, sector `rotation_from` gets drift `−strength`, sector `rotation_to` gets drift `+strength`, and both sectors' volume is ×1.5. Volume is `base_i·(1 + 30·|r|)·mult` with `base_i = 1e6·(1 + i % 5)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_synthetic_market.cpp`:
```cpp
#include <doctest/doctest.h>

#include "market/panel.hpp"
#include "market/synthetic_market.hpp"
#include "test_util.hpp"

using namespace fx;

TEST_CASE("synthetic market is deterministic and well formed") {
  SyntheticConfig cfg;
  BarStore a(test::temp_dir("syn_a")), b(test::temp_dir("syn_b"));
  auto sa = generate_synthetic(cfg, a);
  auto sb = generate_synthetic(cfg, b);
  REQUIRE(sa.size() == 50);
  CHECK(sa[13].ticker == "S1_03");
  CHECK(sa[13].sector == "Sector1");
  const auto& ba = a.bars("S1_03", cfg.tf);
  REQUIRE(ba.size() == 300);
  CHECK(ba[299].c == b.bars("S1_03", cfg.tf)[299].c);
  CHECK(ba[1].t - ba[0].t == 86400);
  for (const Bar& bar : ba) {
    CHECK(bar.h >= std::max(bar.o, bar.c));
    CHECK(bar.l <= std::min(bar.o, bar.c));
    CHECK(bar.v > 0);
  }
}

TEST_CASE("planted rotation moves sector returns apart") {
  SyntheticConfig cfg;
  BarStore s(test::temp_dir("syn_rot"));
  auto secs = generate_synthetic(cfg, s);
  std::vector<std::string> tickers;
  for (auto& x : secs) tickers.push_back(x.ticker);
  Panel p = build_panel(s, tickers, cfg.tf);
  double from_sum = 0, to_sum = 0;
  for (std::size_t t = cfg.rotation_start + 1; t < p.T(); ++t) {
    for (int k = 0; k < cfg.per_sector; ++k) {
      auto i0 = static_cast<std::size_t>(cfg.rotation_from * cfg.per_sector + k);
      auto i1 = static_cast<std::size_t>(cfg.rotation_to * cfg.per_sector + k);
      from_sum += p.close[p.idx(t, i0)] / p.close[p.idx(t - 1, i0)] - 1;
      to_sum += p.close[p.idx(t, i1)] / p.close[p.idx(t - 1, i1)] - 1;
    }
  }
  CHECK(to_sum > 0);
  CHECK(from_sum < 0);
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cmake --build build -j`
Expected: FAIL with `market/synthetic_market.hpp: No such file or directory`.

- [ ] **Step 3: Implement**

`src/market/synthetic_market.hpp`:
```cpp
#pragma once
#include <cstdint>
#include <vector>

#include "core/time.hpp"
#include "market/bar_store.hpp"
#include "market/universe.hpp"

namespace fx {

struct SyntheticConfig {
  int sectors = 5;
  int per_sector = 10;
  int bars = 300;
  std::uint64_t seed = 42;
  int rotation_from = 0;
  int rotation_to = 1;
  int rotation_start = 150;
  double rotation_strength = 0.004;
  Timeframe tf = Timeframe::Day;
  TimePoint start = utc_seconds(2025, 1, 2, 21, 0);
};

std::vector<Security> generate_synthetic(const SyntheticConfig& cfg, BarStore& store);

}  // namespace fx
```

`src/market/synthetic_market.cpp`:
```cpp
#include "market/synthetic_market.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>

namespace fx {

std::vector<Security> generate_synthetic(const SyntheticConfig& cfg, BarStore& store) {
  std::mt19937_64 rng(cfg.seed);
  std::normal_distribution<double> factor_dist(0.0, 0.01), idio_dist(0.0, 0.008);
  const int n = cfg.sectors * cfg.per_sector;

  std::vector<Security> secs;
  for (int s = 0; s < cfg.sectors; ++s) {
    for (int k = 0; k < cfg.per_sector; ++k) {
      char ticker[16];
      std::snprintf(ticker, sizeof ticker, "S%d_%02d", s, k);
      secs.push_back({ticker, ticker, "Sector" + std::to_string(s)});
    }
  }

  std::vector<std::vector<Bar>> series(static_cast<std::size_t>(n));
  std::vector<double> price(static_cast<std::size_t>(n), 100.0);
  const TimePoint step = timeframe_seconds(cfg.tf);
  for (int b = 0; b < cfg.bars; ++b) {
    std::vector<double> factor(static_cast<std::size_t>(cfg.sectors));
    for (auto& f : factor) f = factor_dist(rng);
    const bool rotating = b >= cfg.rotation_start;
    for (int i = 0; i < n; ++i) {
      const int s = i / cfg.per_sector;
      double drift = 0.0, mult = 1.0;
      if (rotating && s == cfg.rotation_from) drift = -cfg.rotation_strength, mult = 1.5;
      if (rotating && s == cfg.rotation_to) drift = cfg.rotation_strength, mult = 1.5;
      const double r = 0.6 * factor[static_cast<std::size_t>(s)] + idio_dist(rng) + drift;
      const auto ui = static_cast<std::size_t>(i);
      const double o = price[ui];
      const double c = o * (1.0 + r);
      const double h = std::max(o, c) * 1.002;
      const double l = std::min(o, c) * 0.998;
      const double base = 1e6 * (1 + i % 5);
      const double v = base * (1.0 + 30.0 * std::abs(r)) * mult;
      series[ui].push_back({cfg.start + b * step, o, h, l, c, v, (o + h + l + c) / 4.0});
      price[ui] = c;
    }
  }
  for (int i = 0; i < n; ++i)
    store.merge(secs[static_cast<std::size_t>(i)].ticker, cfg.tf,
                series[static_cast<std::size_t>(i)]);
  return secs;
}

}  // namespace fx
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cmake --build build -j && ./build/fluxtests`
Expected: `Status: SUCCESS!` (16 test cases).

- [ ] **Step 5: Commit**

```bash
git add src/market/synthetic_market.* tests/test_synthetic_market.cpp
git commit -m "feat: seeded synthetic market with planted sector rotation

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Alpaca client (parsing, paging, retry, dotenv) and incremental sync

**Files:**
- Create: `src/market/alpaca_client.hpp`, `src/market/alpaca_client.cpp`, `src/market/market_sync.hpp`, `src/market/market_sync.cpp`, `.env.example`
- Test: `tests/test_alpaca.cpp`

**Interfaces:**
- Consumes: `Bar`, `Timeframe`, `parse_rfc3339`, `format_rfc3339`, `BarStore`, `aggregate_session_hours`, `utc_seconds` (Tasks 1, 2, 4).
- Produces:
  - `struct fx::HttpResponse { int status; std::string body; }`, `using fx::HttpGet = std::function<HttpResponse(const std::string& path_and_query)>`
  - `struct fx::BarsPage { std::map<std::string, std::vector<Bar>> bars; std::optional<std::string> next_page_token; }`, `BarsPage fx::parse_bars_page(const std::string& json)`
  - `struct fx::AlpacaConfig { std::string key_id, secret, feed = "sip", host = "data.alpaca.markets"; int backoff_initial_ms = 500; int backoff_max_ms = 30000; int max_retries = 6; }`
  - `void fx::load_dotenv(const std::filesystem::path&)` (does not override variables that are already set), `std::optional<AlpacaConfig> fx::alpaca_config_from_env()`
  - `std::string fx::url_encode(std::string_view)`, `std::string_view fx::alpaca_timeframe(Timeframe)` → `"30Min"|"1Day"|"1Week"`
  - `class fx::AlpacaClient { AlpacaClient(AlpacaConfig, HttpGet get = {}); std::map<std::string, std::vector<Bar>> fetch_bars(const std::vector<std::string>& symbols, std::string_view timeframe, TimePoint start, TimePoint end); }`: batches of at most 100 symbols, follows `next_page_token`, retries 429/5xx with backoff, and throws `std::runtime_error` on other statuses or when retries run out
  - `void fx::sync_bars(AlpacaClient&, BarStore&, const std::vector<std::string>& tickers, Timeframe, TimePoint start, TimePoint end)`: each ticker's fetch starts at its cached `last_time` (inclusive, so a partial last bar gets refreshed), or at `start` if it has no cache. Hour data is fetched as 30Min bars and aggregated. Every touched ticker is saved.

- [ ] **Step 1: Write the failing tests**

`tests/test_alpaca.cpp`:
```cpp
#include <doctest/doctest.h>

#include <cstdlib>

#include "core/time.hpp"
#include "market/alpaca_client.hpp"
#include "market/market_sync.hpp"
#include "test_util.hpp"

using namespace fx;

namespace {
const char* kPage1 = R"({"bars":{"AAPL":[
  {"t":"2026-09-29T04:00:00Z","o":250.1,"h":252.0,"l":249.5,"c":251.2,"v":41000000,"n":500000,"vw":250.9}],
  "NVO":[{"t":"2026-09-29T04:00:00Z","o":60.0,"h":61.0,"l":59.5,"c":60.5,"v":5000000,"n":40000,"vw":60.4}]},
  "next_page_token":"QUFQTHxE+/="})";
const char* kPage2 = R"({"bars":{"AAPL":[
  {"t":"2026-09-30T04:00:00Z","o":251.2,"h":253.0,"l":250.0,"c":252.8,"v":39000000,"n":480000,"vw":252.1}]},
  "next_page_token":null})";
AlpacaConfig test_config() {
  AlpacaConfig c;
  c.key_id = "k";
  c.secret = "s";
  c.backoff_initial_ms = 0;
  c.backoff_max_ms = 0;
  return c;
}
}  // namespace

TEST_CASE("parse_bars_page reads bars and token") {
  BarsPage p = parse_bars_page(kPage1);
  REQUIRE(p.bars.at("AAPL").size() == 1);
  CHECK(p.bars.at("AAPL")[0].t == utc_seconds(2026, 9, 29, 4, 0));
  CHECK(p.bars.at("AAPL")[0].vw == doctest::Approx(250.9));
  CHECK(p.bars.at("NVO")[0].c == doctest::Approx(60.5));
  CHECK(p.next_page_token.value() == "QUFQTHxE+/=");
  CHECK_FALSE(parse_bars_page(kPage2).next_page_token.has_value());
  CHECK(parse_bars_page(R"({"bars":{}})").bars.empty());
}

TEST_CASE("fetch_bars follows pages, URL-encodes the token and retries 429") {
  std::vector<std::string> paths;
  int calls = 0;
  HttpGet fake = [&](const std::string& path) -> HttpResponse {
    paths.push_back(path);
    ++calls;
    if (calls == 1) return {429, "rate limited"};
    if (path.find("page_token=") == std::string::npos) return {200, kPage1};
    return {200, kPage2};
  };
  AlpacaClient client(test_config(), fake);
  auto bars = client.fetch_bars({"AAPL", "NVO"}, "1Day", utc_seconds(2026, 9, 29),
                                utc_seconds(2026, 10, 1));
  CHECK(bars.at("AAPL").size() == 2);
  CHECK(bars.at("NVO").size() == 1);
  REQUIRE(paths.size() == 3);
  CHECK(paths[0].find("/v2/stocks/bars?symbols=AAPL,NVO&timeframe=1Day") == 0);
  CHECK(paths[0].find("&adjustment=all&feed=sip") != std::string::npos);
  CHECK(paths[0].find("start=2026-09-29T00:00:00Z") != std::string::npos);
  CHECK(paths[2].find("page_token=QUFQTHxE%2B%2F%3D") != std::string::npos);
}

TEST_CASE("fetch_bars throws on 403 and after exhausting retries") {
  AlpacaClient forbidden(test_config(), [](const std::string&) { return HttpResponse{403, "no"}; });
  CHECK_THROWS_AS(forbidden.fetch_bars({"AAPL"}, "1Day", 0, 1), std::runtime_error);
  AlpacaClient down(test_config(), [](const std::string&) { return HttpResponse{503, ""}; });
  CHECK_THROWS_AS(down.fetch_bars({"AAPL"}, "1Day", 0, 1), std::runtime_error);
}

TEST_CASE("fetch_bars batches 100 symbols per request") {
  std::vector<std::string> symbols;
  for (int i = 0; i < 250; ++i) symbols.push_back("T" + std::to_string(i));
  int calls = 0;
  AlpacaClient client(test_config(), [&](const std::string&) {
    ++calls;
    return HttpResponse{200, R"({"bars":{}})"};
  });
  client.fetch_bars(symbols, "1Day", 0, 1);
  CHECK(calls == 3);
}

TEST_CASE("sync_bars fetches incrementally and saves") {
  auto dir = test::temp_dir("sync");
  BarStore store(dir);
  std::vector<std::string> paths;
  AlpacaClient client(test_config(), [&](const std::string& path) {
    paths.push_back(path);
    return HttpResponse{200, path.find("page_token") == std::string::npos ? kPage1 : kPage2};
  });
  sync_bars(client, store, {"AAPL", "NVO"}, Timeframe::Day, utc_seconds(2026, 9, 1),
            utc_seconds(2026, 10, 1));
  CHECK(store.bars("AAPL", Timeframe::Day).size() == 2);
  CHECK(std::filesystem::exists(dir / "1d" / "AAPL.csv"));

  paths.clear();
  sync_bars(client, store, {"AAPL"}, Timeframe::Day, utc_seconds(2026, 9, 1),
            utc_seconds(2026, 10, 2));
  REQUIRE_FALSE(paths.empty());
  CHECK(paths[0].find("start=2026-09-30T04:00:00Z") != std::string::npos);
}

TEST_CASE("load_dotenv sets unset variables only") {
  auto dir = test::temp_dir("dotenv");
  ::setenv("FLUX_TEST_KEEP", "original", 1);
  ::unsetenv("FLUX_TEST_NEW");
  auto path = test::write_file(dir / ".env",
                               "# comment\nFLUX_TEST_NEW=\"hello\"\nFLUX_TEST_KEEP=changed\n");
  load_dotenv(path);
  CHECK(std::string(std::getenv("FLUX_TEST_NEW")) == "hello");
  CHECK(std::string(std::getenv("FLUX_TEST_KEEP")) == "original");
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cmake --build build -j`
Expected: FAIL with `market/alpaca_client.hpp: No such file or directory`.

- [ ] **Step 3: Implement the client**

`src/market/alpaca_client.hpp`:
```cpp
#pragma once
#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/types.hpp"

namespace fx {

struct HttpResponse {
  int status = 0;
  std::string body;
};

using HttpGet = std::function<HttpResponse(const std::string& path_and_query)>;

struct BarsPage {
  std::map<std::string, std::vector<Bar>> bars;
  std::optional<std::string> next_page_token;
};

BarsPage parse_bars_page(const std::string& json);

struct AlpacaConfig {
  std::string key_id, secret;
  std::string feed = "sip";
  std::string host = "data.alpaca.markets";
  int backoff_initial_ms = 500;
  int backoff_max_ms = 30000;
  int max_retries = 6;
};

void load_dotenv(const std::filesystem::path& path);
std::optional<AlpacaConfig> alpaca_config_from_env();
std::string url_encode(std::string_view s);
std::string_view alpaca_timeframe(Timeframe tf);

class AlpacaClient {
 public:
  explicit AlpacaClient(AlpacaConfig config, HttpGet get = {});

  std::map<std::string, std::vector<Bar>> fetch_bars(const std::vector<std::string>& symbols,
                                                     std::string_view timeframe, TimePoint start,
                                                     TimePoint end);

 private:
  HttpResponse get_with_retry(const std::string& path);
  AlpacaConfig config_;
  HttpGet get_;
};

}  // namespace fx
```

`src/market/alpaca_client.cpp`:
```cpp
#include "market/alpaca_client.hpp"

#include <httplib.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>
#include <random>
#include <stdexcept>
#include <thread>

#include "core/time.hpp"

namespace fx {

BarsPage parse_bars_page(const std::string& json) {
  const auto j = nlohmann::json::parse(json);
  BarsPage page;
  if (j.contains("bars") && j["bars"].is_object()) {
    for (const auto& [symbol, arr] : j["bars"].items()) {
      auto& out = page.bars[symbol];
      for (const auto& b : arr) {
        out.push_back({parse_rfc3339(b.at("t").get<std::string>()), b.at("o").get<double>(),
                       b.at("h").get<double>(), b.at("l").get<double>(), b.at("c").get<double>(),
                       b.at("v").get<double>(), b.value("vw", 0.0)});
      }
    }
  }
  if (j.contains("next_page_token") && j["next_page_token"].is_string())
    page.next_page_token = j["next_page_token"].get<std::string>();
  return page;
}

void load_dotenv(const std::filesystem::path& path) {
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    const auto eq = line.find('=');
    if (eq == std::string::npos) continue;
    std::string key = line.substr(0, eq), value = line.substr(eq + 1);
    if (!value.empty() && value.back() == '\r') value.pop_back();
    if (value.size() >= 2 && (value.front() == '"' || value.front() == '\'') &&
        value.back() == value.front())
      value = value.substr(1, value.size() - 2);
    ::setenv(key.c_str(), value.c_str(), 0);
  }
}

std::optional<AlpacaConfig> alpaca_config_from_env() {
  const char* key = std::getenv("APCA_API_KEY_ID");
  const char* secret = std::getenv("APCA_API_SECRET_KEY");
  if (!key || !secret || !*key || !*secret) return std::nullopt;
  AlpacaConfig c;
  c.key_id = key;
  c.secret = secret;
  if (const char* feed = std::getenv("APCA_DATA_FEED"); feed && *feed) c.feed = feed;
  return c;
}

std::string url_encode(std::string_view s) {
  static const char* kHex = "0123456789ABCDEF";
  std::string out;
  for (unsigned char ch : s) {
    if (std::isalnum(ch) || ch == '-' || ch == '_' || ch == '.' || ch == '~') {
      out += static_cast<char>(ch);
    } else {
      out += '%';
      out += kHex[ch >> 4];
      out += kHex[ch & 15];
    }
  }
  return out;
}

std::string_view alpaca_timeframe(Timeframe tf) {
  switch (tf) {
    case Timeframe::Hour: return "30Min";
    case Timeframe::Day: return "1Day";
    case Timeframe::Week: return "1Week";
  }
  return "1Day";
}

AlpacaClient::AlpacaClient(AlpacaConfig config, HttpGet get)
    : config_(std::move(config)), get_(std::move(get)) {
  if (!get_) {
    get_ = [cfg = config_](const std::string& path) -> HttpResponse {
      httplib::SSLClient cli(cfg.host);
      cli.set_connection_timeout(10);
      cli.set_read_timeout(60);
      httplib::Headers headers = {{"APCA-API-KEY-ID", cfg.key_id},
                                  {"APCA-API-SECRET-KEY", cfg.secret}};
      auto res = cli.Get(path, headers);
      if (!res) return {0, httplib::to_string(res.error())};
      return {res->status, res->body};
    };
  }
}

HttpResponse AlpacaClient::get_with_retry(const std::string& path) {
  std::mt19937 jitter_rng(std::random_device{}());
  int delay_ms = config_.backoff_initial_ms;
  for (int attempt = 0;; ++attempt) {
    HttpResponse res = get_(path);
    if (res.status == 200) return res;
    const bool retryable = res.status == 0 || res.status == 429 || res.status >= 500;
    if (!retryable || attempt >= config_.max_retries) {
      throw std::runtime_error("Alpaca GET failed (HTTP " + std::to_string(res.status) +
                               "): " + res.body.substr(0, 200));
    }
    if (delay_ms > 0) {
      std::uniform_int_distribution<int> jitter(0, delay_ms / 2);
      std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms + jitter(jitter_rng)));
    }
    delay_ms = std::min(delay_ms * 2, config_.backoff_max_ms);
  }
}

std::map<std::string, std::vector<Bar>> AlpacaClient::fetch_bars(
    const std::vector<std::string>& symbols, std::string_view timeframe, TimePoint start,
    TimePoint end) {
  std::map<std::string, std::vector<Bar>> out;
  constexpr std::size_t kBatch = 100;
  for (std::size_t b = 0; b < symbols.size(); b += kBatch) {
    std::string joined;
    for (std::size_t i = b; i < std::min(symbols.size(), b + kBatch); ++i) {
      if (!joined.empty()) joined += ',';
      joined += url_encode(symbols[i]);
    }
    const std::string base = "/v2/stocks/bars?symbols=" + joined +
                             "&timeframe=" + std::string(timeframe) +
                             "&start=" + format_rfc3339(start) + "&end=" + format_rfc3339(end) +
                             "&limit=10000&adjustment=all&feed=" + config_.feed;
    std::optional<std::string> token;
    do {
      std::string path = base;
      if (token) path += "&page_token=" + url_encode(*token);
      BarsPage page = parse_bars_page(get_with_retry(path).body);
      for (auto& [sym, bars] : page.bars)
        out[sym].insert(out[sym].end(), bars.begin(), bars.end());
      token = page.next_page_token;
    } while (token);
  }
  return out;
}

}  // namespace fx
```

Note: `url_encode` keeps `.` unchanged, so `BRK.B` passes through as is. Commas between symbols are added after encoding, which is why the test path shows `symbols=AAPL,NVO`.

- [ ] **Step 4: Implement sync and the env template**

`src/market/market_sync.hpp`:
```cpp
#pragma once
#include <string>
#include <vector>

#include "market/alpaca_client.hpp"
#include "market/bar_store.hpp"

namespace fx {

void sync_bars(AlpacaClient& client, BarStore& store, const std::vector<std::string>& tickers,
               Timeframe tf, TimePoint start, TimePoint end);

}  // namespace fx
```

`src/market/market_sync.cpp`:
```cpp
#include "market/market_sync.hpp"

#include <map>

#include "market/session_calendar.hpp"

namespace fx {

void sync_bars(AlpacaClient& client, BarStore& store, const std::vector<std::string>& tickers,
               Timeframe tf, TimePoint start, TimePoint end) {
  std::map<TimePoint, std::vector<std::string>> by_start;
  for (const auto& t : tickers) by_start[store.last_time(t, tf).value_or(start)].push_back(t);

  for (const auto& [from, group] : by_start) {
    if (from >= end) continue;
    auto fetched = client.fetch_bars(group, alpaca_timeframe(tf), from, end);
    for (auto& [ticker, bars] : fetched) {
      store.merge(ticker, tf, tf == Timeframe::Hour ? aggregate_session_hours(bars) : bars);
      store.save(ticker, tf);
    }
  }
}

}  // namespace fx
```

`.env.example`:
```
# Copy to .env and fill in. .env is gitignored.
APCA_API_KEY_ID=
APCA_API_SECRET_KEY=
# sip (default; consolidated, 15-min delayed on free plan) or iex
APCA_DATA_FEED=sip
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cmake --build build -j && ./build/fluxtests`
Expected: `Status: SUCCESS!` (22 test cases).

- [ ] **Step 6: Commit**

```bash
git add src/market/alpaca_client.* src/market/market_sync.* tests/test_alpaca.cpp .env.example
git commit -m "feat: Alpaca bars client with paging, retry and incremental sync

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Flux builder (pressure, rolling correlation, slow/fast accumulators)

**Files:**
- Create: `src/graph/flux_builder.hpp`, `src/graph/flux_builder.cpp`
- Test: `tests/test_flux_builder.cpp`

**Interfaces:**
- Consumes: nothing from earlier tasks beyond the standard library.
- Produces:
  - `struct fx::FluxParams { double lambda = 1.0; double halflife_slow = 20; double halflife_fast = 3; int corr_window = 60; }`
  - `void fx::bar_flux(std::span<const double> pressure, std::span<const double> corr, double lambda, std::vector<double>& out)`. `corr` is n×n, or empty for no affinity. `out` is resized to n×n and overwritten. For each source *i* (p < 0) and sink *j* (p > 0): `out[i*n+j] = |p_i| · p_j a_ij / Σ_k p_k a_ik`, where `a_ij = 1 + λ·max(0, corr_ij)`.
  - `class fx::FluxBuilder { FluxBuilder(std::size_t n, FluxParams); void step(std::span<const double> returns, std::span<const double> dollar_volume); const std::vector<double>& flux_slow() const; const std::vector<double>& flux_fast() const; double correlation(std::size_t i, std::size_t j) const; std::size_t size() const; }`. A NaN input means pressure 0 and a correlation sample of 0. The accumulator update is `F ← 2^(−1/halflife)·F + f`.

- [ ] **Step 1: Write the failing tests**

`tests/test_flux_builder.cpp`:
```cpp
#include <doctest/doctest.h>

#include <cmath>
#include <limits>

#include "graph/flux_builder.hpp"

using namespace fx;

TEST_CASE("bar_flux conserves each source's outflow (no affinity)") {
  std::vector<double> out;
  bar_flux(std::vector<double>{-3, 1, 2}, {}, 0.0, out);
  REQUIRE(out.size() == 9);
  CHECK(out[0 * 3 + 1] == doctest::Approx(1.0));
  CHECK(out[0 * 3 + 2] == doctest::Approx(2.0));
  for (std::size_t j = 0; j < 3; ++j) {
    CHECK(out[1 * 3 + j] == 0.0);  // sinks emit nothing
    CHECK(out[2 * 3 + j] == 0.0);
  }
}

TEST_CASE("bar_flux applies correlation affinity") {
  std::vector<double> corr = {1, 1, 0, 1, 1, 0, 0, 0, 1};
  std::vector<double> out;
  bar_flux(std::vector<double>{-1, 1, 1}, corr, 1.0, out);
  CHECK(out[1] == doctest::Approx(2.0 / 3.0));
  CHECK(out[2] == doctest::Approx(1.0 / 3.0));
  CHECK(out[1] + out[2] == doctest::Approx(1.0));
}

TEST_CASE("bar_flux with no sinks produces no flux") {
  std::vector<double> out;
  bar_flux(std::vector<double>{-1, -2, 0}, {}, 0.0, out);
  for (double v : out) CHECK(v == 0.0);
}

TEST_CASE("accumulators decay by half-life and NaN inputs are inert") {
  FluxParams p;
  p.lambda = 0.0;
  p.halflife_slow = 1.0;
  p.halflife_fast = 1.0;
  FluxBuilder fb(3, p);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> r = {-0.01, 0.01, nan}, dv = {300, 100, 100};
  fb.step(r, dv);
  fb.step(r, dv);
  // per-bar flux 0->1 = |-3| = 3; after two steps with halflife 1: 0.5*3 + 3 = 4.5
  CHECK(fb.flux_slow()[0 * 3 + 1] == doctest::Approx(4.5));
  CHECK(fb.flux_fast()[0 * 3 + 1] == doctest::Approx(4.5));
  CHECK(fb.flux_slow()[0 * 3 + 2] == 0.0);
}

TEST_CASE("rolling correlation tracks co-movement") {
  FluxParams p;
  p.corr_window = 20;
  FluxBuilder fb(3, p);
  std::vector<double> dv = {1, 1, 1};
  for (int t = 0; t < 50; ++t) {
    const double x = std::sin(t * 0.7) * 0.01;
    const double z = std::cos(t * 1.3) * 0.01;
    fb.step(std::vector<double>{x, -x, z}, dv);
  }
  CHECK(fb.correlation(0, 0) == doctest::Approx(1.0));
  CHECK(fb.correlation(0, 1) == doctest::Approx(-1.0).epsilon(1e-6));
  CHECK(std::abs(fb.correlation(0, 2)) < 0.6);
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cmake --build build -j`
Expected: FAIL with `graph/flux_builder.hpp: No such file or directory`.

- [ ] **Step 3: Implement**

`src/graph/flux_builder.hpp`:
```cpp
#pragma once
#include <cstddef>
#include <span>
#include <vector>

namespace fx {

struct FluxParams {
  double lambda = 1.0;
  double halflife_slow = 20;
  double halflife_fast = 3;
  int corr_window = 60;
};

void bar_flux(std::span<const double> pressure, std::span<const double> corr, double lambda,
              std::vector<double>& out);

class FluxBuilder {
 public:
  FluxBuilder(std::size_t n, FluxParams params);

  void step(std::span<const double> returns, std::span<const double> dollar_volume);
  const std::vector<double>& flux_slow() const { return slow_; }
  const std::vector<double>& flux_fast() const { return fast_; }
  double correlation(std::size_t i, std::size_t j) const { return corr_[i * n_ + j]; }
  std::size_t size() const { return n_; }

 private:
  void push_correlation_sample(const std::vector<double>& x);
  void recompute_sums();
  void update_correlation();

  std::size_t n_;
  FluxParams params_;
  std::vector<double> slow_, fast_, bar_, corr_;
  // rolling window of returns: ring buffer [window][n]
  std::vector<double> ring_;
  std::size_t head_ = 0, count_ = 0;
  std::vector<double> sx_, sxx_, sxy_;
};

}  // namespace fx
```

`src/graph/flux_builder.cpp`:
```cpp
#include "graph/flux_builder.hpp"

#include <algorithm>
#include <cmath>

namespace fx {

void bar_flux(std::span<const double> pressure, std::span<const double> corr, double lambda,
              std::vector<double>& out) {
  const std::size_t n = pressure.size();
  out.assign(n * n, 0.0);
  const bool affinity = !corr.empty() && lambda != 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    if (!(pressure[i] < 0)) continue;
    double denom = 0;
    for (std::size_t k = 0; k < n; ++k) {
      if (!(pressure[k] > 0)) continue;
      const double a = affinity ? 1.0 + lambda * std::max(0.0, corr[i * n + k]) : 1.0;
      denom += pressure[k] * a;
    }
    if (denom <= 0) continue;
    const double outflow = -pressure[i];
    for (std::size_t j = 0; j < n; ++j) {
      if (!(pressure[j] > 0)) continue;
      const double a = affinity ? 1.0 + lambda * std::max(0.0, corr[i * n + j]) : 1.0;
      out[i * n + j] = outflow * pressure[j] * a / denom;
    }
  }
}

FluxBuilder::FluxBuilder(std::size_t n, FluxParams params)
    : n_(n),
      params_(params),
      slow_(n * n, 0.0),
      fast_(n * n, 0.0),
      corr_(n * n, 0.0),
      ring_(static_cast<std::size_t>(std::max(params.corr_window, 2)) * n, 0.0),
      sx_(n, 0.0),
      sxx_(n, 0.0),
      sxy_(n * n, 0.0) {
  for (std::size_t i = 0; i < n; ++i) corr_[i * n + i] = 1.0;
}

void FluxBuilder::push_correlation_sample(const std::vector<double>& x) {
  const std::size_t window = ring_.size() / n_;
  double* slot = &ring_[head_ * n_];
  if (count_ == window) {  // evict oldest
    for (std::size_t i = 0; i < n_; ++i) {
      sx_[i] -= slot[i];
      sxx_[i] -= slot[i] * slot[i];
      for (std::size_t j = 0; j < n_; ++j) sxy_[i * n_ + j] -= slot[i] * slot[j];
    }
  } else {
    ++count_;
  }
  for (std::size_t i = 0; i < n_; ++i) {
    slot[i] = x[i];
    sx_[i] += x[i];
    sxx_[i] += x[i] * x[i];
    for (std::size_t j = 0; j < n_; ++j) sxy_[i * n_ + j] += x[i] * x[j];
  }
  head_ = (head_ + 1) % window;
  if (head_ == 0) recompute_sums();  // bound floating-point drift once per window
}

void FluxBuilder::recompute_sums() {
  std::fill(sx_.begin(), sx_.end(), 0.0);
  std::fill(sxx_.begin(), sxx_.end(), 0.0);
  std::fill(sxy_.begin(), sxy_.end(), 0.0);
  for (std::size_t w = 0; w < count_; ++w) {
    const double* row = &ring_[w * n_];
    for (std::size_t i = 0; i < n_; ++i) {
      sx_[i] += row[i];
      sxx_[i] += row[i] * row[i];
      for (std::size_t j = 0; j < n_; ++j) sxy_[i * n_ + j] += row[i] * row[j];
    }
  }
}

void FluxBuilder::update_correlation() {
  const double c = static_cast<double>(count_);
  for (std::size_t i = 0; i < n_; ++i) {
    const double vi = c * sxx_[i] - sx_[i] * sx_[i];
    for (std::size_t j = 0; j < n_; ++j) {
      if (i == j) {
        corr_[i * n_ + j] = 1.0;
        continue;
      }
      const double vj = c * sxx_[j] - sx_[j] * sx_[j];
      const double denom = std::sqrt(std::max(0.0, vi) * std::max(0.0, vj));
      corr_[i * n_ + j] = denom > 1e-18 ? (c * sxy_[i * n_ + j] - sx_[i] * sx_[j]) / denom : 0.0;
    }
  }
}

void FluxBuilder::step(std::span<const double> returns, std::span<const double> dollar_volume) {
  std::vector<double> pressure(n_, 0.0), sample(n_, 0.0);
  for (std::size_t i = 0; i < n_; ++i) {
    const double r = returns[i], dv = dollar_volume[i];
    if (std::isfinite(r)) sample[i] = r;
    if (std::isfinite(r) && std::isfinite(dv)) pressure[i] = r * dv;
  }
  const bool use_corr = params_.lambda != 0.0;
  if (use_corr) {
    push_correlation_sample(sample);
    update_correlation();
  }
  bar_flux(pressure, use_corr ? std::span<const double>(corr_) : std::span<const double>(),
           params_.lambda, bar_);
  const double ds = std::exp2(-1.0 / params_.halflife_slow);
  const double df = std::exp2(-1.0 / params_.halflife_fast);
  for (std::size_t k = 0; k < bar_.size(); ++k) {
    slow_[k] = ds * slow_[k] + bar_[k];
    fast_[k] = df * fast_[k] + bar_[k];
  }
}

}  // namespace fx
```

The correlation test case uses the default `lambda = 1.0`, so correlation is updated there. The decay case sets `lambda = 0` and doesn't depend on it.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cmake --build build -j && ./build/fluxtests`
Expected: `Status: SUCCESS!` (27 test cases).

- [ ] **Step 5: Commit**

```bash
git add src/graph/flux_builder.* tests/test_flux_builder.cpp
git commit -m "feat: flux builder with correlation affinity and slow/fast accumulators

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Sparse transition matrix (top-k prune, row-normalize, self-loops)

**Files:**
- Create: `src/graph/csr.hpp`, `src/graph/csr.cpp`
- Test: `tests/test_csr.cpp`

**Interfaces:**
- Consumes: the dense n×n flux layout from Task 7 (`F[i*n+j]`).
- Produces:
  - `struct fx::Csr { std::size_t n = 0; std::vector<std::size_t> row_ptr; std::vector<std::uint32_t> col; std::vector<double> val; }`. `row_ptr` has n+1 entries, and columns within a row are ascending.
  - `Csr fx::build_transition(std::span<const double> flux, std::size_t n, std::size_t k)`: keeps the k largest positive off-diagonal entries per row and normalizes each row to sum 1. A row with no positive entry becomes `{i: 1.0}`, a self-loop.
  - `std::vector<double> fx::left_multiply(const Csr& P, std::span<const double> x)`: returns y = x·P.

- [ ] **Step 1: Write the failing tests**

`tests/test_csr.cpp`:
```cpp
#include <doctest/doctest.h>

#include "graph/csr.hpp"

using namespace fx;

TEST_CASE("build_transition keeps top-k, normalizes rows, adds self-loops") {
  // row 0: flows 5, 1, 3 to nodes 1, 2, 3; row 1: nothing; row 2: one flow; row 3: nothing
  std::vector<double> F = {0, 5, 1, 3,  //
                           0, 0, 0, 0,  //
                           2, 0, 0, 0,  //
                           0, 0, 0, 0};
  Csr P = build_transition(F, 4, 2);
  REQUIRE(P.row_ptr.size() == 5);
  // row 0 keeps 5 (col 1) and 3 (col 3), ascending columns
  REQUIRE(P.row_ptr[1] - P.row_ptr[0] == 2);
  CHECK(P.col[0] == 1);
  CHECK(P.val[0] == doctest::Approx(5.0 / 8.0));
  CHECK(P.col[1] == 3);
  CHECK(P.val[1] == doctest::Approx(3.0 / 8.0));
  // row 1 is an accumulator: self-loop
  REQUIRE(P.row_ptr[2] - P.row_ptr[1] == 1);
  CHECK(P.col[P.row_ptr[1]] == 1);
  CHECK(P.val[P.row_ptr[1]] == 1.0);
  for (std::size_t i = 0; i < 4; ++i) {
    double s = 0;
    for (auto e = P.row_ptr[i]; e < P.row_ptr[i + 1]; ++e) s += P.val[e];
    CHECK(s == doctest::Approx(1.0));
  }
}

TEST_CASE("k >= n keeps every positive edge") {
  std::vector<double> F = {0, 1, 1, 1, 0, 1, 1, 1, 0};
  Csr P = build_transition(F, 3, 10);
  CHECK(P.val.size() == 6);
}

TEST_CASE("left_multiply computes x * P") {
  std::vector<double> F = {0, 1, 0, 0};  // node0 -> node1, node1 self-loop
  Csr P = build_transition(F, 2, 5);
  auto y = left_multiply(P, std::vector<double>{0.3, 0.7});
  CHECK(y[0] == doctest::Approx(0.0));
  CHECK(y[1] == doctest::Approx(1.0));
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cmake --build build -j`
Expected: FAIL with `graph/csr.hpp: No such file or directory`.

- [ ] **Step 3: Implement**

`src/graph/csr.hpp`:
```cpp
#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace fx {

struct Csr {
  std::size_t n = 0;
  std::vector<std::size_t> row_ptr;
  std::vector<std::uint32_t> col;
  std::vector<double> val;
};

Csr build_transition(std::span<const double> flux, std::size_t n, std::size_t k);
std::vector<double> left_multiply(const Csr& P, std::span<const double> x);

}  // namespace fx
```

`src/graph/csr.cpp`:
```cpp
#include "graph/csr.hpp"

#include <algorithm>
#include <utility>

namespace fx {

Csr build_transition(std::span<const double> flux, std::size_t n, std::size_t k) {
  Csr P;
  P.n = n;
  P.row_ptr.reserve(n + 1);
  P.row_ptr.push_back(0);
  std::vector<std::pair<double, std::uint32_t>> row;
  for (std::size_t i = 0; i < n; ++i) {
    row.clear();
    for (std::size_t j = 0; j < n; ++j) {
      const double w = flux[i * n + j];
      if (j != i && w > 0) row.emplace_back(w, static_cast<std::uint32_t>(j));
    }
    if (row.size() > k) {
      std::partial_sort(row.begin(), row.begin() + static_cast<std::ptrdiff_t>(k), row.end(),
                        [](const auto& a, const auto& b) { return a.first > b.first; });
      row.resize(k);
    }
    if (row.empty()) {
      P.col.push_back(static_cast<std::uint32_t>(i));
      P.val.push_back(1.0);
    } else {
      std::sort(row.begin(), row.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; });
      double sum = 0;
      for (const auto& [w, j] : row) sum += w;
      for (const auto& [w, j] : row) {
        P.col.push_back(j);
        P.val.push_back(w / sum);
      }
    }
    P.row_ptr.push_back(P.col.size());
  }
  return P;
}

std::vector<double> left_multiply(const Csr& P, std::span<const double> x) {
  std::vector<double> y(P.n, 0.0);
  for (std::size_t i = 0; i < P.n; ++i) {
    const double xi = x[i];
    if (xi == 0) continue;
    for (auto e = P.row_ptr[i]; e < P.row_ptr[i + 1]; ++e) y[P.col[e]] += xi * P.val[e];
  }
  return y;
}

}  // namespace fx
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cmake --build build -j && ./build/fluxtests`
Expected: `Status: SUCCESS!` (30 test cases).

- [ ] **Step 5: Commit**

```bash
git add src/graph/csr.* tests/test_csr.cpp
git commit -m "feat: top-k sparse transition matrix with self-loops

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Markov steady-state solver

**Files:**
- Create: `src/graph/markov_solver.hpp`, `src/graph/markov_solver.cpp`
- Test: `tests/test_markov_solver.cpp`

**Interfaces:**
- Consumes: `Csr`, `build_transition`, `left_multiply` (Task 8).
- Produces:
  - `struct fx::SolveResult { std::vector<double> pi; int iterations; double residual; bool converged; }`
  - `std::vector<double> fx::damped_step(const Csr& P, double alpha, std::span<const double> pi)`: computes `α·πP + (1−α)/N`, renormalized to sum 1.
  - `SolveResult fx::stationary(const Csr& P, double alpha, std::span<const double> warm_start, double tol = 1e-10, int max_iter = 1000)`. An empty or wrongly sized warm start means start from uniform.
  - `std::vector<double> fx::propagate(const Csr& P, double alpha, std::span<const double> pi, int k)`: applies `damped_step` k times.
  - `std::vector<double> fx::hotness(std::span<const double> pi)`: returns `N·π_i − 1`.

- [ ] **Step 1: Write the failing tests**

`tests/test_markov_solver.cpp`:
```cpp
#include <doctest/doctest.h>

#include "graph/csr.hpp"
#include "graph/markov_solver.hpp"

using namespace fx;

namespace {
// Builds a Csr directly from a dense row-stochastic matrix (k = n keeps all edges;
// diagonal entries are handled by putting them in explicitly).
Csr dense_to_csr(const std::vector<double>& M, std::size_t n) {
  Csr P;
  P.n = n;
  P.row_ptr.push_back(0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      if (M[i * n + j] != 0) {
        P.col.push_back(static_cast<std::uint32_t>(j));
        P.val.push_back(M[i * n + j]);
      }
    }
    P.row_ptr.push_back(P.col.size());
  }
  return P;
}
}  // namespace

TEST_CASE("undamped 3-state chain matches the analytic stationary distribution") {
  Csr P = dense_to_csr({0.5, 0.5, 0, 0.25, 0.5, 0.25, 0, 0.5, 0.5}, 3);
  SolveResult r = stationary(P, 1.0, {});
  CHECK(r.converged);
  CHECK(r.pi[0] == doctest::Approx(0.25));
  CHECK(r.pi[1] == doctest::Approx(0.5));
  CHECK(r.pi[2] == doctest::Approx(0.25));
}

TEST_CASE("damped 2-state chain matches the closed form") {
  Csr P = dense_to_csr({0.9, 0.1, 0.5, 0.5}, 2);
  SolveResult r = stationary(P, 0.85, {});
  CHECK(r.converged);
  CHECK(r.pi[0] == doctest::Approx(0.5 / (1 - 0.4 * 0.85)));
  CHECK(r.pi[0] + r.pi[1] == doctest::Approx(1.0));
}

TEST_CASE("warm start converges to the same answer in fewer iterations") {
  Csr P = dense_to_csr({0.5, 0.5, 0, 0.25, 0.5, 0.25, 0, 0.5, 0.5}, 3);
  SolveResult cold = stationary(P, 0.85, {});
  SolveResult warm = stationary(P, 0.85, cold.pi);
  CHECK(warm.converged);
  CHECK(warm.iterations < cold.iterations);
  for (int i = 0; i < 3; ++i) CHECK(warm.pi[i] == doctest::Approx(cold.pi[i]));
}

TEST_CASE("damping makes a reducible chain converge (two absorbing nodes)") {
  std::vector<double> F = {0, 1, 1, 0,  //
                           0, 0, 0, 0,  //
                           0, 0, 0, 0,  //
                           0, 1, 1, 0};
  Csr P = build_transition(F, 4, 4);
  SolveResult r = stationary(P, 0.85, {});
  CHECK(r.converged);
  CHECK(r.pi[1] == doctest::Approx(r.pi[2]));
  CHECK(r.pi[1] > r.pi[0]);
  auto h = hotness(r.pi);
  CHECK(h[1] > 0);
  CHECK(h[0] < 0);
  double sum_h = 0;
  for (double x : h) sum_h += x;
  CHECK(sum_h == doctest::Approx(0.0).epsilon(1e-9));
}

TEST_CASE("propagate applies k damped steps") {
  Csr P = dense_to_csr({0.9, 0.1, 0.5, 0.5}, 2);
  auto one = propagate(P, 1.0, std::vector<double>{0.5, 0.5}, 1);
  CHECK(one[0] == doctest::Approx(0.7));
  auto two = propagate(P, 1.0, std::vector<double>{0.5, 0.5}, 2);
  CHECK(two[0] == doctest::Approx(0.7 * 0.9 + 0.3 * 0.5));
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cmake --build build -j`
Expected: FAIL with `graph/markov_solver.hpp: No such file or directory`.

- [ ] **Step 3: Implement**

`src/graph/markov_solver.hpp`:
```cpp
#pragma once
#include <span>
#include <vector>

#include "graph/csr.hpp"

namespace fx {

struct SolveResult {
  std::vector<double> pi;
  int iterations = 0;
  double residual = 0;
  bool converged = false;
};

std::vector<double> damped_step(const Csr& P, double alpha, std::span<const double> pi);
SolveResult stationary(const Csr& P, double alpha, std::span<const double> warm_start,
                       double tol = 1e-10, int max_iter = 1000);
std::vector<double> propagate(const Csr& P, double alpha, std::span<const double> pi, int k);
std::vector<double> hotness(std::span<const double> pi);

}  // namespace fx
```

`src/graph/markov_solver.cpp`:
```cpp
#include "graph/markov_solver.hpp"

#include <cmath>

namespace fx {

std::vector<double> damped_step(const Csr& P, double alpha, std::span<const double> pi) {
  std::vector<double> next = left_multiply(P, pi);
  const double teleport = (1.0 - alpha) / static_cast<double>(P.n);
  double sum = 0;
  for (double& x : next) {
    x = alpha * x + teleport;
    sum += x;
  }
  for (double& x : next) x /= sum;
  return next;
}

SolveResult stationary(const Csr& P, double alpha, std::span<const double> warm_start,
                       double tol, int max_iter) {
  SolveResult r;
  if (warm_start.size() == P.n) {
    r.pi.assign(warm_start.begin(), warm_start.end());
  } else {
    r.pi.assign(P.n, 1.0 / static_cast<double>(P.n));
  }
  for (r.iterations = 1; r.iterations <= max_iter; ++r.iterations) {
    std::vector<double> next = damped_step(P, alpha, r.pi);
    r.residual = 0;
    for (std::size_t i = 0; i < P.n; ++i) r.residual += std::abs(next[i] - r.pi[i]);
    r.pi = std::move(next);
    if (r.residual < tol) {
      r.converged = true;
      return r;
    }
  }
  r.iterations = max_iter;
  return r;
}

std::vector<double> propagate(const Csr& P, double alpha, std::span<const double> pi, int k) {
  std::vector<double> x(pi.begin(), pi.end());
  for (int s = 0; s < k; ++s) x = damped_step(P, alpha, x);
  return x;
}

std::vector<double> hotness(std::span<const double> pi) {
  std::vector<double> h(pi.size());
  const double n = static_cast<double>(pi.size());
  for (std::size_t i = 0; i < pi.size(); ++i) h[i] = n * pi[i] - 1.0;
  return h;
}

}  // namespace fx
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cmake --build build -j && ./build/fluxtests`
Expected: `Status: SUCCESS!` (35 test cases).

- [ ] **Step 5: Commit**

```bash
git add src/graph/markov_solver.* tests/test_markov_solver.cpp
git commit -m "feat: damped power-iteration steady-state solver

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Forecaster (π·P_fast^k plus drift)

**Files:**
- Create: `src/graph/forecaster.hpp`, `src/graph/forecaster.cpp`
- Test: `tests/test_forecaster.cpp`

**Interfaces:**
- Consumes: `Csr`, `propagate`, `stationary` (Task 9).
- Produces:
  - `struct fx::Forecast { int k; std::vector<double> pi_k; std::vector<double> score; }`
  - `Forecast fx::forecast(const Csr& P_fast, double alpha, std::span<const double> pi_now, std::span<const double> pi_prev, int k, double beta)`: `pi_k = propagate(P_fast, alpha, pi_now, k)` and `score_i = N·(pi_k_i − pi_now_i) + β·N·(pi_now_i − pi_prev_i)`. The drift term is 0 when `pi_prev` is empty.

- [ ] **Step 1: Write the failing tests**

`tests/test_forecaster.cpp`:
```cpp
#include <doctest/doctest.h>

#include "graph/forecaster.hpp"
#include "graph/markov_solver.hpp"

using namespace fx;

namespace {
Csr two_state() {
  Csr P;
  P.n = 2;
  P.row_ptr = {0, 2, 4};
  P.col = {0, 1, 0, 1};
  P.val = {0.9, 0.1, 0.5, 0.5};
  return P;
}
}  // namespace

TEST_CASE("forecast score from propagation without drift") {
  Forecast f = forecast(two_state(), 1.0, std::vector<double>{0.5, 0.5}, {}, 1, 0.5);
  CHECK(f.k == 1);
  CHECK(f.pi_k[0] == doctest::Approx(0.7));
  CHECK(f.score[0] == doctest::Approx(2 * (0.7 - 0.5)));
  CHECK(f.score[1] == doctest::Approx(2 * (0.3 - 0.5)));
}

TEST_CASE("forecast of a vector already stationary for P_fast is zero") {
  Csr P = two_state();
  auto pi = stationary(P, 0.85, {}).pi;
  Forecast f = forecast(P, 0.85, pi, {}, 8, 0.5);
  CHECK(f.score[0] == doctest::Approx(0.0).epsilon(1e-8));
}

TEST_CASE("drift term adds beta * N * (pi_now - pi_prev)") {
  Csr P = two_state();
  auto pi = stationary(P, 0.85, {}).pi;
  std::vector<double> prev = {pi[0] - 0.1, pi[1] + 0.1};
  Forecast f = forecast(P, 0.85, pi, prev, 1, 0.5);
  CHECK(f.score[0] == doctest::Approx(0.5 * 2 * 0.1).epsilon(1e-6));
  CHECK(f.score[1] == doctest::Approx(-0.5 * 2 * 0.1).epsilon(1e-6));
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cmake --build build -j`
Expected: FAIL with `graph/forecaster.hpp: No such file or directory`.

- [ ] **Step 3: Implement**

`src/graph/forecaster.hpp`:
```cpp
#pragma once
#include <span>
#include <vector>

#include "graph/csr.hpp"

namespace fx {

struct Forecast {
  int k = 0;
  std::vector<double> pi_k;
  std::vector<double> score;
};

Forecast forecast(const Csr& P_fast, double alpha, std::span<const double> pi_now,
                  std::span<const double> pi_prev, int k, double beta);

}  // namespace fx
```

`src/graph/forecaster.cpp`:
```cpp
#include "graph/forecaster.hpp"

#include "graph/markov_solver.hpp"

namespace fx {

Forecast forecast(const Csr& P_fast, double alpha, std::span<const double> pi_now,
                  std::span<const double> pi_prev, int k, double beta) {
  Forecast f;
  f.k = k;
  f.pi_k = propagate(P_fast, alpha, pi_now, k);
  const std::size_t n = pi_now.size();
  const double N = static_cast<double>(n);
  const bool drift = pi_prev.size() == n;
  f.score.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    f.score[i] = N * (f.pi_k[i] - pi_now[i]);
    if (drift) f.score[i] += beta * N * (pi_now[i] - pi_prev[i]);
  }
  return f;
}

}  // namespace fx
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cmake --build build -j && ./build/fluxtests`
Expected: `Status: SUCCESS!` (38 test cases).

- [ ] **Step 5: Commit**

```bash
git add src/graph/forecaster.* tests/test_forecaster.cpp
git commit -m "feat: fast-flux propagation forecaster with drift

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: Core pipeline, CLI and synthetic-rotation integration test

**Files:**
- Create: `src/pipeline/core_pipeline.hpp`, `src/pipeline/core_pipeline.cpp`, `README.md`
- Modify: `src/main.cpp` (replace the stub)
- Test: `tests/test_core_pipeline.cpp`

**Interfaces:**
- Consumes: `Panel` (Task 4), `FluxBuilder`/`FluxParams` (Task 7), `build_transition`/`Csr` (Task 8), `stationary`/`hotness`/`SolveResult` (Task 9), `forecast`/`Forecast` (Task 10), `generate_synthetic`/`SyntheticConfig` (Task 5), `Universe`, `load_portfolio` (Task 3), `BarStore`, `build_panel` (Task 4), `AlpacaClient`, `load_dotenv`, `alpaca_config_from_env`, `sync_bars` (Task 6).
- Produces (milestone 2 builds on these):
  - `struct fx::CoreParams { FluxParams flux; std::size_t top_k = 20; double alpha = 0.85; double beta = 0.5; std::vector<int> horizons{1, 4, 8}; }`
  - `struct fx::Frame { TimePoint t; std::vector<double> pi, h; SolveResult solve; std::vector<Forecast> forecasts; Csr P; double compute_ms; }`. `forecasts` is parallel to `horizons`, and `P` is the slow transition matrix.
  - `class fx::CorePipeline { CorePipeline(std::size_t n, CoreParams); Frame step(const Panel&, std::size_t t); }`. It requires `t ≥ 1` and keeps the warm start and previous π internally.
  - `Frame fx::run_panel_last(const Panel&, const CoreParams&)`: steps t = 1 … T−1 and returns the last frame. It throws `std::runtime_error` if T < 2.
  - CLI: `fluxscape [--mode synthetic|replay|alpaca] [--timeframe 1h|1d|1w] [--lookback-days N] [--top N] [--data DIR]`

- [ ] **Step 1: Write the failing integration test**

`tests/test_core_pipeline.cpp`:
```cpp
#include <doctest/doctest.h>

#include <map>

#include "market/panel.hpp"
#include "market/synthetic_market.hpp"
#include "pipeline/core_pipeline.hpp"
#include "test_util.hpp"

using namespace fx;

TEST_CASE("planted rotation makes the receiving sector the top hill") {
  SyntheticConfig cfg;  // 5 sectors x 10, rotation Sector0 -> Sector1 from bar 150
  BarStore store(test::temp_dir("pipeline"));
  auto secs = generate_synthetic(cfg, store);
  std::vector<std::string> tickers;
  for (auto& s : secs) tickers.push_back(s.ticker);
  Panel panel = build_panel(store, tickers, cfg.tf);

  Frame f = run_panel_last(panel, CoreParams{});
  CHECK(f.solve.converged);
  REQUIRE(f.h.size() == 50);
  REQUIRE(f.forecasts.size() == 3);
  CHECK(f.forecasts[2].k == 8);
  CHECK(f.t == panel.times.back());

  std::map<std::string, double> sector_mean;
  for (std::size_t i = 0; i < secs.size(); ++i) sector_mean[secs[i].sector] += f.h[i] / 10.0;
  for (const auto& [sector, mean] : sector_mean) {
    if (sector != "Sector1") CHECK(sector_mean["Sector1"] > mean);
  }
  CHECK(sector_mean["Sector1"] > 0);
  CHECK(sector_mean["Sector0"] < sector_mean["Sector1"]);
}

TEST_CASE("pipeline requires at least two bars") {
  Panel p;
  p.times = {100};
  p.tickers = {"A"};
  p.close = {1};
  p.volume = {1};
  p.vwap = {1};
  CHECK_THROWS_AS(run_panel_last(p, CoreParams{}), std::runtime_error);
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cmake --build build -j`
Expected: FAIL with `pipeline/core_pipeline.hpp: No such file or directory`.

- [ ] **Step 3: Implement the pipeline**

`src/pipeline/core_pipeline.hpp`:
```cpp
#pragma once
#include <vector>

#include "core/types.hpp"
#include "graph/csr.hpp"
#include "graph/flux_builder.hpp"
#include "graph/forecaster.hpp"
#include "graph/markov_solver.hpp"
#include "market/panel.hpp"

namespace fx {

struct CoreParams {
  FluxParams flux;
  std::size_t top_k = 20;
  double alpha = 0.85;
  double beta = 0.5;
  std::vector<int> horizons{1, 4, 8};
};

struct Frame {
  TimePoint t = 0;
  std::vector<double> pi, h;
  SolveResult solve;
  std::vector<Forecast> forecasts;  // parallel to CoreParams::horizons
  Csr P;                            // slow (equilibrium) transition matrix
  double compute_ms = 0;
};

class CorePipeline {
 public:
  CorePipeline(std::size_t n, CoreParams params);
  Frame step(const Panel& panel, std::size_t t);

 private:
  std::size_t n_;
  CoreParams params_;
  FluxBuilder flux_;
  std::vector<double> prev_pi_;
};

Frame run_panel_last(const Panel& panel, const CoreParams& params);

}  // namespace fx
```

`src/pipeline/core_pipeline.cpp`:
```cpp
#include "pipeline/core_pipeline.hpp"

#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace fx {

CorePipeline::CorePipeline(std::size_t n, CoreParams params)
    : n_(n), params_(std::move(params)), flux_(n, params_.flux) {}

Frame CorePipeline::step(const Panel& panel, std::size_t t) {
  const auto t0 = std::chrono::steady_clock::now();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> returns(n_, nan), dollar_volume(n_, nan);
  for (std::size_t i = 0; i < n_; ++i) {
    const double c = panel.close[panel.idx(t, i)];
    const double c_prev = panel.close[panel.idx(t - 1, i)];
    if (std::isfinite(c) && std::isfinite(c_prev) && c_prev > 0) returns[i] = c / c_prev - 1.0;
    const double v = panel.volume[panel.idx(t, i)];
    const double vw = panel.vwap[panel.idx(t, i)];
    if (std::isfinite(v) && std::isfinite(vw)) dollar_volume[i] = v * vw;
  }
  flux_.step(returns, dollar_volume);

  Frame f;
  f.t = panel.times[t];
  f.P = build_transition(flux_.flux_slow(), n_, params_.top_k);
  const Csr P_fast = build_transition(flux_.flux_fast(), n_, params_.top_k);
  f.solve = stationary(f.P, params_.alpha, prev_pi_);
  f.pi = f.solve.pi;
  f.h = hotness(f.pi);
  for (int k : params_.horizons)
    f.forecasts.push_back(forecast(P_fast, params_.alpha, f.pi, prev_pi_, k, params_.beta));
  prev_pi_ = f.pi;
  f.compute_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  return f;
}

Frame run_panel_last(const Panel& panel, const CoreParams& params) {
  if (panel.T() < 2) throw std::runtime_error("need at least two bars to build flux");
  CorePipeline pipeline(panel.N(), params);
  Frame last;
  for (std::size_t t = 1; t < panel.T(); ++t) last = pipeline.step(panel, t);
  return last;
}

}  // namespace fx
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cmake --build build -j && ./build/fluxtests`
Expected: `Status: SUCCESS!` (40 test cases). If the rotation test fails, do not loosen its assertions. Print `sector_mean` for every sector and use superpowers:systematic-debugging. The likely causes are a source/sink sign error in `bar_flux`, or `stationary` reading a row as a column.

- [ ] **Step 5: Implement the CLI**

`src/main.cpp`:
```cpp
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
        fx::sync_bars(client, store, universe.price_tickers(), args.tf, start, end);
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

    std::vector<std::size_t> order(panel.N());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](auto a, auto b) { return f.h[a] > f.h[b]; });
    const auto& score = f.forecasts.front().score;
    const std::size_t top = std::min(args.top, order.size());
    std::printf("HILLS (money accumulating)          hotness        pi     score+%d\n",
                f.forecasts.front().k);
    for (std::size_t r = 0; r < top; ++r)
      print_row(r + 1, universe.nodes()[order[r]], f.h[order[r]], f.pi[order[r]], score[order[r]]);
    std::printf("\nVALLEYS (money draining)\n");
    for (std::size_t r = 0; r < top; ++r) {
      const std::size_t i = order[order.size() - 1 - r];
      print_row(order.size() - r, universe.nodes()[i], f.h[i], f.pi[i], score[i]);
    }
    if (portfolio) {
      std::printf("\nPORTFOLIO HOLDINGS\n");
      for (const auto& hld : portfolio->holdings) {
        if (auto i = universe.index_of(hld.ticker)) {
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
```

- [ ] **Step 6: Write the README**

`README.md`:
````markdown
# Fluxscape

Models the market as a flux graph (money leaving net-sold stocks for net-bought ones),
solves its Markov steady state like PageRank, and ranks the hottest and coldest stocks.
Design: `docs/superpowers/specs/2026-10-01-fluxscape-design.md`.

## Build and test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/fluxtests
```

## Run

```bash
./build/fluxscape --mode synthetic                 # no keys needed
cp .env.example .env                               # add Alpaca keys
./build/fluxscape --mode alpaca --timeframe 1d     # fetch and cache, then solve
./build/fluxscape --mode replay --timeframe 1d     # cached data only
python3 scripts/fetch_sp500.py                     # refresh the S&P 500 list
```

Advisory and experimental. The flux is inferred from price and dollar-volume
co-movement, not observed order flow.
````

- [ ] **Step 7: Verify the CLI end to end**

Run: `cmake --build build -j && ./build/fluxtests && ./build/fluxscape --mode synthetic --top 5`
Expected: tests `Status: SUCCESS!`. The CLI prints `solver: converged`, and most of the top-5 HILLS rows are in `Sector1`.

Run: `./build/fluxscape --mode replay`
Expected: exit code 1 with `fluxscape: not enough cached bars; run with --mode alpaca first` (no cache exists yet).

If `.env` has Alpaca keys, run `./build/fluxscape --mode alpaca --timeframe 1d --top 10`.
Expected: about 506 tickers synced into `data/cache/1d/`, `solver: converged`, and a PORTFOLIO HOLDINGS section listing AAPL, NVDA, LLY, NVO, NKE and F with hotness values, plus VOO marked as a fund. If there are no keys, skip this step and say so in the task report.

- [ ] **Step 8: Commit**

```bash
git add src/pipeline src/main.cpp tests/test_core_pipeline.cpp README.md
git commit -m "feat: core pipeline and CLI ranking market hills and valleys

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

## Next plans (written after this milestone lands)

- **Milestone 2:** geometry (force layout, Hungarian lattice with hysteresis, IDW), the HTTP/SSE server, and the deck.gl UI with the left-panel controls. It consumes `Frame`, `CoreParams` and `Universe` from this plan.
- **Milestone 3:** fund look-through, ADMM optimizer, shadow books, strategy versions, backtest, catch-up, the NYSE holiday calendar and the performance panel.
