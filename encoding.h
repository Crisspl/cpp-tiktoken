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
#pragma once
#include "common.h"
#include "byte_pair_encoding.h"
#include "modelparams.h"
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tiktoken
{

class IResourceReader;

class GptEncoding {
    int n_words;
    BytePairEncodingCore byte_pair_encoding_core_processor_;

    GptEncoding(tt_stl::string&& pattern_string, bpe_encoding_t&& byte_pair_ranks,
        tt_stl::unordered_map<tt_stl::string, int>&& special_token_mappings, int explicit_n_vocab);

    GptEncoding(const GptEncoding&) = delete;
    GptEncoding &operator=(const GptEncoding&) = delete;

public:
    GptEncoding(GptEncoding &&) = default;
    GptEncoding &operator=(GptEncoding &&) = default;

    static GptEncoding get_encoding(ModelParams&& params);
    static GptEncoding get_encoding_llama3(ModelParams&& params);
    static GptEncoding get_encoding_llama3_1(ModelParams&& params);

    static GptEncoding get_encoding(LanguageModel model, IResourceReader *resource_reader = nullptr, const char *resource_name = nullptr);
    static GptEncoding get_encoding_llama3(LanguageModel model, IResourceReader *resource_reader = nullptr, const char *resource_name = nullptr);
    static GptEncoding get_encoding_llama3_1(LanguageModel model, IResourceReader *resource_reader = nullptr, const char *resource_name = nullptr);
    tt_stl::vector<int> encode(const tt_stl::string &line_to_encode, const tt_stl::unordered_set<tt_stl::string> &allowed_special = {},
        const tt_stl::unordered_set<tt_stl::string> &disallowed_special = { "all" });
    tt_stl::string decode(const tt_stl::vector<int> &input_tokens_to_decode);

    [[nodiscard]] const bpe_encoding_t& get_byte_pair_token_map() const;
};

}
