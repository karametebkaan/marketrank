#include "core/types.hpp"

#include <stdexcept>
#include <string>

namespace mr {

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

}  // namespace mr
