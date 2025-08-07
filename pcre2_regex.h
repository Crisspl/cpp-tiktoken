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
#include <cstring>
#include <string>
#include <vector>

namespace tiktoken
{

class PCRERegex {
    class Impl;
public:
    explicit PCRERegex(const tt_stl::string &pattern, int flags = 0);
    PCRERegex(PCRERegex&&);
    PCRERegex& operator=(PCRERegex&&);
    PCRERegex& operator=(const PCRERegex&) = delete;
    PCRERegex(const PCRERegex &) = delete;
    ~PCRERegex();

    [[nodiscard]] tt_stl::vector<tt_stl::string> get_all_matches(const tt_stl::string &text) const;
    void replace_all(tt_stl::string &text, const tt_stl::string &replacement) const;
    [[nodiscard]] bool contains(const tt_stl::string& text) const;
    [[nodiscard]] tt_stl::vector<std::pair<tt_stl::string::size_type, tt_stl::string::size_type>> all_matches(const tt_stl::string &text) const;

private:
    void* impl_state_;
};

}
