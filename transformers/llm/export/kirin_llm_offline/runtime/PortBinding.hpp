#pragma once
#include "FullNpuRuntime.hpp"
#include <algorithm>

namespace KirinFullNpu {
inline std::string normalized(std::string name) {
    if (name.size() > 2 && name.substr(name.size() - 2) == ":0") name.resize(name.size() - 2);
    return name;
}
inline bool named(const std::string& actual, const std::string& expected) {
    auto a = normalized(actual), e = normalized(expected);
    return a == e || (a.size() > e.size() && a.compare(a.size() - e.size(), e.size(), e) == 0 &&
                      (a[a.size() - e.size() - 1] == '/' || a[a.size() - e.size() - 1] == ':'));
}
inline std::vector<int> nonsingleton(std::vector<int> shape) {
    shape.erase(std::remove(shape.begin(), shape.end(), 1), shape.end()); return shape;
}
inline bool compatible(const Port& expected, const Port& actual) {
    bool dtype = expected.dtype == actual.dtype ||
        ((expected.dtype == DType::F32 || expected.dtype == DType::F16) &&
         (actual.dtype == DType::F32 || actual.dtype == DType::F16));
    return dtype && expected.elements() == actual.elements() && nonsingleton(expected.shape) == nonsingleton(actual.shape);
}
inline std::string description(const Port& port) {
    std::string text = "name=" + port.name + " dtype=";
    text += port.dtype == DType::F32 ? "float32" : port.dtype == DType::F16 ? "float16" : port.dtype == DType::I8 ? "int8" : "int32";
    text += " shape=[";
    for (size_t i = 0; i < port.shape.size(); ++i) text += (i ? "," : "") + std::to_string(port.shape[i]);
    return text + "]";
}
inline bool bindPorts(const std::vector<Port>& expected, const std::vector<Port>& actual,
          const std::map<std::string, std::string>& aliases, std::vector<size_t>& mapping, std::string& error) {
    std::vector<bool> consumed(actual.size(), false);
    for (const auto& p : expected) {
        std::string name = p.name;
        auto alias = aliases.find(name);
        std::string canonical = alias == aliases.end() ? name : alias->second;
        std::vector<std::string> equivalents{canonical};
        for (const auto& pair : aliases) if (pair.second == canonical) equivalents.push_back(pair.first);
        bool present = std::any_of(actual.begin(), actual.end(), [&](const Port& t) { return named(t.name, name); });
        int found = -1;
        for (size_t i = 0; i < actual.size(); ++i) {
            bool match = present ? named(actual[i].name, name) :
                std::any_of(equivalents.begin(), equivalents.end(), [&](const std::string& e) { return named(actual[i].name, e); });
            if (!match) continue;
            if (found != -1) { error = "Ambiguous device port name: " + name; return false; }
            found = int(i);
        }
        // Shapes alone cannot distinguish K/V, rotary cos/sin or scales. Never
        // silently use positional binding for these ambiguous ports.
        if (found < 0 || !compatible(p, actual[found])) {
            error = "Device port missing or incompatible: " + description(p); return false;
        }
        if (consumed[found] && equivalents.size() == 1) { error = "Duplicate device port binding"; return false; }
        consumed[found] = true; mapping.push_back(size_t(found));
    }
    if (std::find(consumed.begin(), consumed.end(), false) != consumed.end()) { error = "Unexpected device port"; return false; }
    return true;
}

} // namespace KirinFullNpu
