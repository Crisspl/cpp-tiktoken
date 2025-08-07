/*
 * Copyright (c) 2023 by Mark Tarrabain All rights reserved. Redistribution and use in source and binary forms,
 * with or without modification, are permitted provided that the following conditions are met:
 * Redistributions of source code must retain the above copyright notice, this list of conditions and the following
 * disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
 * disclaimer in the documentation and/or other materials provided with the distribution.
 * Neither the name of the nor the names of its contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include "byte_pair_encoding.h"
#include "pcre2_regex.h"
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <algorithm>

namespace tiktoken
{

BytePairEncodingCore::BytePairEncodingCore(bpe_encoding_t&& byte_pair_ranks,
    tt_stl::unordered_map<tt_stl::string, int>&& special_token_mappings,
    PCRERegex&& pattern_string) :
    byte_pair_ranks_(std::move(byte_pair_ranks)),
    special_token_mappings_(std::move(special_token_mappings)),
    pattern_string_(std::move(pattern_string)) { }

tt_stl::vector<int> BytePairEncodingCore::byte_pair_merge(const tt_stl::vector<uint8_t> &piece,
    const bpe_encoding_t &ranks,
    const std::function<int(int, int)> &f)
{
    tt_stl::vector<std::pair<int, int>> partitions(piece.size() + 1);
    for (size_t i = 0; i <= piece.size(); ++i) {
        partitions[i] = { static_cast<int>(i), std::numeric_limits<int>::max() };
    }
    auto get_rank = [&piece, &partitions, &ranks](size_t idx, int skip) -> std::optional<int> {
        if (idx + skip + 2 >= partitions.size()) {
            return std::nullopt;
        }
        tt_stl::vector<uint8_t> key(piece.begin() + partitions[idx].first, piece.begin() + partitions[idx + skip + 2].first);
        auto rank_iter = ranks.find(key);
        return (rank_iter != ranks.end()) ? std::optional<int>(rank_iter->second) : std::nullopt;
    };
    for (size_t i = 0; i < partitions.size() - 2; ++i) {
        auto rank = get_rank(i, 0);
        if (rank.has_value()) {
            partitions[i].second = rank.value();
        }
    }
    while (partitions.size() > 1) {
        int min_rank = std::numeric_limits<int>::max();
        size_t min_rank_idx = 0;
        for (size_t i = 0; i < partitions.size() - 1; ++i) {
            if (partitions[i].second < min_rank) {
                min_rank = partitions[i].second;
                min_rank_idx = i;
            }
        }
        if (min_rank != std::numeric_limits<int>::max()) {
            partitions[min_rank_idx].second = get_rank(min_rank_idx, 1).value_or(std::numeric_limits<int>::max());

            if (min_rank_idx > 0) {
                partitions[min_rank_idx - 1].second = get_rank(min_rank_idx - 1, 1).value_or(std::numeric_limits<int>::max());
            }
            partitions.erase(partitions.begin() + static_cast<long long>(min_rank_idx) + 1);
        } else {
            break;
        }
    }
    tt_stl::vector<int> output;
    output.reserve(partitions.size() - 1);
    for (size_t i = 0; i < partitions.size() - 1; ++i) {
        output.push_back(f(partitions[i].first, partitions[i + 1].first));
    }
    return output;
}


tt_stl::vector<tt_stl::string> BytePairEncodingCore::break_into_specials(tt_stl::string const& line_to_encode, const tt_stl::unordered_set<tt_stl::string> &allowed_special) {
    tt_stl::vector<std::pair<size_t, size_t>> separator_offsets;
    tt_stl::string::size_type pos = 0;
    for (auto& sep: special_token_mappings_) {
        if (!sep.first.empty()) {
            while ((pos = line_to_encode.find(sep.first, pos)) != tt_stl::string::npos) {
                separator_offsets.push_back({ pos, pos + sep.first.size() });
                pos += sep.first.size();
            }
            pos = 0;
        } else if (allowed_special.count("")) {
            separator_offsets.push_back({ 0, 0 });
        }
    }
    std::sort(separator_offsets.begin(), separator_offsets.end());
    tt_stl::vector<tt_stl::string> lines;
    for (auto [begin, end]: separator_offsets) {
        lines.push_back(line_to_encode.substr(pos, begin - pos));
        lines.push_back(line_to_encode.substr(begin, end - begin));
        pos = end;
    }
    lines.push_back(line_to_encode.substr(pos, line_to_encode.size() - pos));
    return lines;
}

std::pair<tt_stl::vector<int>, tt_stl::vector<int>> BytePairEncodingCore::encode_native(const tt_stl::string &line_to_encode,
    const tt_stl::unordered_set<tt_stl::string> &allowed_special)
{
    tt_stl::vector<int> tokens;
    tt_stl::vector<int> segment_ids;
    auto lines = break_into_specials(line_to_encode, allowed_special);
    for(auto line:lines) {
        auto special_mapping = special_token_mappings_.find(line);
        if (special_mapping != special_token_mappings_.end() && allowed_special.count(line) > 0) {
            tokens.push_back(special_mapping->second);
            segment_ids.push_back(0);
        } else {
            auto matches = pattern_string_.get_all_matches(line);
            for (auto token: matches) {
                auto special_mapping = special_token_mappings_.find(token);
                if (special_mapping != special_token_mappings_.end() && allowed_special.count(token) > 0) {
                    if (!token.empty()) {
                        tokens.push_back(special_mapping->second);
                        segment_ids.push_back(0);
                    }
                } else {
                    tt_stl::vector<uint8_t> utf8_encoded(token.begin(), token.end());
                    if (utf8_encoded.size() == 1) {
                        auto rank_iter = byte_pair_ranks_.find(utf8_encoded);
                        if (rank_iter != byte_pair_ranks_.end()) {
                            tokens.push_back(rank_iter->second);
                            segment_ids.push_back(0);
                        }
                    } else {
                        auto byte_pairs = byte_pair_merge(utf8_encoded, byte_pair_ranks_, [&](int start, int end) {
                            tt_stl::vector<uint8_t> key(utf8_encoded.begin() + start, utf8_encoded.begin() + end);
                            return byte_pair_ranks_[key];
                        });
                        tokens.insert(tokens.end(), byte_pairs.begin(), byte_pairs.end());
                        segment_ids.insert(segment_ids.end(), byte_pairs.size(), 0);
                    }
                }
            }
        }
    }
    return std::make_pair(tokens, segment_ids);
}

tt_stl::string BytePairEncodingCore::decode_native(const tt_stl::vector<int> &input_tokens_to_decode)
{
    tt_stl::string decoded_string;
    for (const int token_id: input_tokens_to_decode) {
        auto special_token = std::find_if(special_token_mappings_.begin(), special_token_mappings_.end(),
            [token_id](const auto &pair) { return pair.second == token_id; });
        if (special_token != special_token_mappings_.end()) {
            decoded_string += special_token->first;
        } else {
            for (const auto &byte_pair: byte_pair_ranks_) {
                if (byte_pair.second == token_id) {
                    decoded_string += tt_stl::string(byte_pair.first.begin(), byte_pair.first.end());
                    break;
                }
            }
        }
    }
    return decoded_string;
}

}