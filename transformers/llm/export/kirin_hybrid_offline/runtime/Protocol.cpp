#include "OfflineNpu2.hpp"
#include <algorithm>
#include <cstring>
#include <cmath>

namespace OfflineNpu2 {
namespace {
constexpr int H = 2048, Rows = 19, Cols = 8, ImageTokens = Rows * Cols;
}
bool parseImages(const std::string& input, Prompt& out, std::string& error) {
    out = {}; error.clear();
    size_t start = 0;
    while (true) {
        auto begin = input.find("<img>", start);
        if (begin == std::string::npos) { out.text += input.substr(start); break; }
        out.text += input.substr(start, begin - start);
        auto end = input.find("</img>", begin + 5);
        if (end == std::string::npos) { error = "Unclosed <img> tag"; return false; }
        auto path = input.substr(begin + 5, end - begin - 5);
        auto hw = path.find("<hw>");
        if (hw != std::string::npos) {
            auto close = path.find("</hw>", hw);
            if (close == std::string::npos) { error = "Unclosed <hw> tag"; return false; }
            path.erase(hw, close + 5 - hw); // fixed compiled grid, independent of UI dimensions
        }
        if (path.empty() || path.find('<') != std::string::npos || path.find('\0') != std::string::npos || out.images.size() >= 16) {
            error = "Invalid image path/tag, or more than 16 images"; return false;
        }
        out.images.push_back(path);
        out.text += "<|vision_start|>";
        for (int i = 0; i < ImageTokens; ++i) out.text += "<|image_pad|>";
        out.text += "<|vision_end|>";
        start = end + 6;
    }
    if (out.text.find("<img") != std::string::npos || out.text.find("</img>") != std::string::npos ||
        out.text.find("<video>") != std::string::npos || out.text.find("<audio>") != std::string::npos) {
        error = "Offline NPU supports text and <img> file paths only"; return false;
    }
    return true;
}
bool makePositions(const std::vector<int>& ids, Prompt& p, int base, std::string& error) {
    p.positions.assign(3 * ids.size(), 0); p.imageOffsets.clear();
    int pos = base;
    for (size_t i = 0; i < ids.size();) {
        if (ids[i] == 151655) {
            if (i == 0 || ids[i-1] != 151652 || i + ImageTokens >= ids.size() ||
                ids[i+ImageTokens] != 151653 ||
                !std::all_of(ids.begin()+i, ids.begin()+i+ImageTokens, [](int t) { return t == 151655; })) {
                error = "Image token protocol mismatch (expected 152 image tokens)"; return false;
            }
            p.imageOffsets.push_back(i);
            for (int j = 0; j < ImageTokens; ++j) {
                p.positions[i+j] = pos;
                p.positions[ids.size()+i+j] = pos + j / Cols;
                p.positions[2*ids.size()+i+j] = pos + j % Cols;
            }
            pos += Rows; i += ImageTokens;
        } else {
            for (int axis = 0; axis < 3; ++axis) p.positions[axis*ids.size()+i] = pos;
            ++pos; ++i;
        }
    }
    p.nextPosition = pos;
    if (p.imageOffsets.size() != p.images.size()) { error = "Image paths and token spans disagree"; return false; }
    return true;
}
bool injectFeatures(const Prompt& p, const std::vector<Features>& features, size_t tokens,
                    float* embeddings, std::vector<float>& deep, std::string& error) {
    if (!embeddings || features.size() != p.imageOffsets.size()) { error = "Missing image embeddings"; return false; }
    deep.assign(3 * tokens * H, 0);
    for (size_t i = 0; i < features.size(); ++i) {
        const auto& f = features[i]; size_t offset = p.imageOffsets[i];
        if (offset + ImageTokens > tokens || f.image.size() != size_t(ImageTokens)*H ||
            f.deepstack.size() != size_t(3)*ImageTokens*H) { error = "Image/DeepStack dimensions disagree"; return false; }
        for (const auto* v : {&f.image, &f.deepstack})
            for (float x : *v) if (!std::isfinite(x)) { error = "Nonfinite visual feature"; return false; }
        std::memcpy(embeddings + offset*H, f.image.data(), f.image.size()*sizeof(float));
        // Captures 5/11/17 have already passed their individual mergers in vision_full.
        // These are residuals for language layers 0/1/2, in that order.
        for (int layer = 0; layer < 3; ++layer)
            std::memcpy(deep.data() + (layer*tokens+offset)*H,
                        f.deepstack.data() + size_t(layer)*ImageTokens*H, size_t(ImageTokens)*H*sizeof(float));
    }
    return true;
}
} // namespace OfflineNpu2
