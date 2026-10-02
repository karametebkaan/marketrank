#pragma once
#include <cstddef>
#include <filesystem>

#include "market/bar_store.hpp"

namespace fx {

// One-time import of the milestone-1 CSV cache (<root>/<1h|1d|1w>/<TICKER>.csv + .from sidecars).
std::size_t migrate_csv_cache(const std::filesystem::path& csv_root, BarStore& store);

}  // namespace fx
