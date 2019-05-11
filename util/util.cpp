#include "util.h"

std::string pretty_print_count(const double count) {
	const double giga = 1000000000;
	const double mega = 1000000;
	const double kilo = 1000;
	if (count > giga) {
		return std::to_string(count / giga) + " G";
	} else if (count > mega) {
		return std::to_string(count / mega) + " M";
	} else if (count > kilo) {
		return std::to_string(count / kilo) + " K";
	}
	return std::to_string(count);
}

uint64_t align_to(uint64_t val, uint64_t align) {
	return ((val + align - 1) / align) * align;
}
