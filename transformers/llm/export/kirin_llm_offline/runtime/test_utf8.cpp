#include "Utf8Stream.hpp"
#include <cassert>
#include <iostream>

int main() {
    using KirinFullNpu::takeUtf8;
    const std::string text = "中文🙂ASCII";
    std::string pending, result;
    for (char c : text) { pending += c; result += takeUtf8(pending); }
    assert(result == text && pending.empty());
    pending = "\xe4\xb8";
    assert(takeUtf8(pending).empty());
    assert(takeUtf8(pending, true) == "\xef\xbf\xbd" && pending.empty());
    pending = "\xe4" "X";
    assert(takeUtf8(pending) == "\xef\xbf\xbdX");
    pending = "\x80\xc0\xaf";
    assert(takeUtf8(pending) == std::string("\xef\xbf\xbd") + "\xef\xbf\xbd\xef\xbf\xbd");
    pending = "\xed\xa0\x80"; // UTF-16 surrogate code point is not valid UTF-8.
    assert(takeUtf8(pending) == std::string("\xef\xbf\xbd") + "\xef\xbf\xbd\xef\xbf\xbd");
    pending = "\xf4\x90\x80\x80"; // Beyond Unicode's maximum code point.
    assert(takeUtf8(pending) == std::string("\xef\xbf\xbd") + "\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd");
    std::cout << "PASS: UTF8 token splits, invalid bytes and incomplete EOS tails\n";
}
