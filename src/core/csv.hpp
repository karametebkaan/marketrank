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
