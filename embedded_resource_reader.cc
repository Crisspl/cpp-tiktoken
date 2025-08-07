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
#include "embedded_resource_reader.h"
#include "encoding_utils.h"

#include <cstdlib>
#include <stdexcept>

#ifndef TIKTOKEN_EMBEDDED_RESOURCES
#include <filesystem>
#include <fstream>
#ifdef _WIN32
#include <windows.h>
#else
#include <limits.h>
#include <unistd.h>
#endif
#endif

#if defined(TIKTOKEN_EMBEDDED_RESOURCES)
extern std::pair<const unsigned char *, size_t> get_resource_cl100k_base();
extern std::pair<const unsigned char *, size_t> get_resource_o200k_base();
extern std::pair<const unsigned char *, size_t> get_resource_p50k_base();
extern std::pair<const unsigned char *, size_t> get_resource_r50k_base();
#endif

namespace tiktoken
{

namespace 
{
#ifndef TIKTOKEN_EMBEDDED_RESOURCES
    std::filesystem::path get_exe_parent_path_intern() {
        std::filesystem::path path;
#ifdef _WIN32
        wchar_t result[MAX_PATH] = {0};
        GetModuleFileNameW(nullptr, result, MAX_PATH);
        path = std::filesystem::path(result);
#else
        char result[PATH_MAX];
        ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
        path = std::filesystem::path(tt_stl::string(result, count > 0 ? count : 0));
#endif
        return path.parent_path();
    }

    static const std::filesystem::path g_exe_parent_path = get_exe_parent_path_intern();
#endif

    class EmbeddedResourceReader: public IResourceReader {
    public:
        tt_stl::vector<tt_stl::string> readLines(std::string_view resourceName) override;
    };

    tt_stl::vector<tt_stl::string> EmbeddedResourceReader::readLines(std::string_view resourceName)
    {
#ifndef TIKTOKEN_EMBEDDED_RESOURCES
        std::filesystem::path resource_path = g_exe_parent_path / "tokenizers" / resourceName;
        std::ifstream file(resource_path);
        if (!file.is_open()) {
#if TIKTOKEN_EXCEPTIONS_ENABLE
            throw std::runtime_error("Embedded resource '" + resource_path.string() + "' not found.");
#else
            return {};
#endif
        }

        tt_stl::string line;
        tt_stl::vector<tt_stl::string> lines;
        while (tt_stl::getline(file, line)) {
            lines.push_back(line);
        }

        return lines;
#else
        auto readLinesFromMem = [](std::pair<const unsigned char *, size_t> mem) {
            struct membuf: std::streambuf {
                membuf(char *base, std::ptrdiff_t n)
                {
                    this->setg(base, base, base + n);
                }
            };

            membuf sbuf(const_cast<char *>((const char*) mem.first), (ptrdiff_t) mem.second);
            std::istream file(&sbuf);

            tt_stl::string line;
            tt_stl::vector<tt_stl::string> lines;
            while (tt_stl::getline(file, line))
                lines.push_back(line);

            return lines;
        };

        if (resourceName == "o200k_base.tiktoken")
        {
            return readLinesFromMem(get_resource_o200k_base());
        }
        else if (resourceName == "cl100k_base.tiktoken") 
        {
            return readLinesFromMem(get_resource_cl100k_base());
        }
        else if (resourceName == "r50k_base.tiktoken") 
        {
            return readLinesFromMem(get_resource_r50k_base());
        }
        else if (resourceName == "p50k_base.tiktoken") 
        {
            return readLinesFromMem(get_resource_p50k_base());
        } 
        else
        {
#if TIKTOKEN_EXCEPTIONS_ENABLE
            throw std::runtime_error("Embedded resource '" + (tt_stl::string) resourceName + "' not found.");
#else
            return {};
#endif
        }
        
#endif
    }
}

EmbeddedResourceLoader::EmbeddedResourceLoader(const tt_stl::string& dataSourceName, IResourceReader* reader)
    : resourceReader_(reader)
    , dataSourceName_(dataSourceName)
{
}

tt_stl::vector<tt_stl::string> EmbeddedResourceLoader::readEmbeddedResourceAsLines() {
    if (!!resourceReader_) {
        return resourceReader_->readLines(dataSourceName_);
    }
    return EmbeddedResourceReader().readLines(dataSourceName_);
}

bpe_encoding_t
EmbeddedResourceLoader::loadTokenBytePairEncoding()
{
    auto lines = readEmbeddedResourceAsLines();
    bpe_encoding_t token_byte_pair_encoding;

    for (const auto &line: lines) {
        if (!line.empty()) {
            const char* whitespace_chars = " \f\n\r\t\v";

            const size_t b64str_end_offset = line.find_first_of(whitespace_chars);
            const size_t rank_offset = line.find_first_not_of(whitespace_chars, b64str_end_offset);

            const auto decoded = base64::decode(std::string_view(line.c_str(), b64str_end_offset));
            const int rank = std::strtol(line.c_str() + rank_offset, nullptr, 10);
            token_byte_pair_encoding.insert({std::move(decoded), rank});
        }
    }

    return token_byte_pair_encoding;
}

}