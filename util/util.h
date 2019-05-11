#pragma once

#include <string>

// Format the count as #G, #M, #K, depending on its magnitude
std::string pretty_print_count(const double count);

uint64_t align_to(uint64_t val, uint64_t align);

