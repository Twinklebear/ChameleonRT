#pragma once

#include <memory>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

class FileMapping {
    void *mapping;
    size_t num_bytes;
#ifdef _WIN32
    HANDLE file;
    HANDLE mapping_handle;
#else
    int file;
#endif

public:
    // Map the file into memory
    FileMapping(const std::string &fname);

    FileMapping(FileMapping &&fm);

    ~FileMapping();

    FileMapping &operator=(FileMapping &&fm);

    FileMapping(const FileMapping &) = delete;

    FileMapping &operator=(const FileMapping &) = delete;

    const uint8_t *data() const;

    size_t nbytes() const;
};
