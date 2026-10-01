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
