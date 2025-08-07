#pragma once

#define TIKTOKEN_EXCEPTIONS_ENABLE 0

#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <string_view>

#include "SIMDString/SIMDString.h"

namespace tiktoken
{

#ifndef TIKTOKEN_STL_TYPEDEFS_DEFINED
#define TIKTOKEN_STL_TYPEDEFS_DEFINED 1
namespace tt_stl
{
    using std::getline;
    using std::to_string;

    using string = std::string;
    template <typename T>
    using vector = std::vector<T>;
    template <typename T>
    using unordered_set = std::unordered_set<T>;
    template<typename K, typename V, typename H = std::hash<K>>
    using unordered_map = std::unordered_map<K, V, H>;
}
#endif // TIKTOKEN_STL_TYPEDEFS_DEFINED

struct bpe_encoding_hash_t
{
    std::size_t operator()(const tt_stl::vector<uint8_t> &v) const
    {
        const char *const begin = reinterpret_cast<const char *>(v.data());
        return std::hash<std::string_view> {}(std::string_view(begin, v.size()));
    }
};

using bpe_encoding_t = tt_stl::unordered_map<tt_stl::vector<uint8_t>, int, bpe_encoding_hash_t>;

}
