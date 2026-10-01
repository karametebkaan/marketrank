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
