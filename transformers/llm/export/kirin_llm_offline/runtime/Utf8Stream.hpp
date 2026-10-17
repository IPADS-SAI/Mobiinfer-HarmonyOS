#pragma once
#include <cstdint>
#include <string>

namespace KirinFullNpu {
// Decode byte-level tokenizer output incrementally. Keep incomplete characters
// between tokens, replace malformed bytes, and flush a truncated tail at EOS.
inline std::string takeUtf8(std::string& pending, bool finish = false) {
    std::string output;
    size_t at = 0;
    while (at < pending.size()) {
        auto first = static_cast<unsigned char>(pending[at]);
        if (first < 128) { output += pending[at++]; continue; }
        int width = first >= 0xc2 && first <= 0xdf ? 2 : first >= 0xe0 && first <= 0xef ? 3 :
                    first >= 0xf0 && first <= 0xf4 ? 4 : 0;
        if (!width) { output += "\xef\xbf\xbd"; ++at; continue; }
        size_t count = 1;
        while (count < size_t(width) && at + count < pending.size()) {
            auto next = static_cast<unsigned char>(pending[at + count]);
            if (next < 0x80 || next > 0xbf || (count == 1 &&
                ((first == 0xe0 && next < 0xa0) || (first == 0xed && next > 0x9f) ||
                 (first == 0xf0 && next < 0x90) || (first == 0xf4 && next > 0x8f)))) break;
            ++count;
        }
        if (count == size_t(width)) { output.append(pending, at, count); at += count; }
        else if (at + count == pending.size() && !finish) break;
        else { output += "\xef\xbf\xbd"; at += count; }
    }
    pending.erase(0, at);
    return output;
}
} // namespace KirinFullNpu
