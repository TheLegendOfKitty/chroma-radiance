#pragma once
#include "darts.h"
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <regex>
#include <memory>
#include <cassert>

// Minimal JSON parser for T5 tokenizer vocab extraction
// Only handles the specific structure we need
struct T5TokenizerVocab {
    std::vector<std::pair<std::string, float>> pieces;
    int unk_id = 2;
    std::string replacement = "\xe2\x96\x81"; // ▁ (U+2581)
    bool add_prefix_space = true;
};

static T5TokenizerVocab parse_tokenizer_json(const std::string& path) {
    T5TokenizerVocab vocab;

    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "Failed to open tokenizer file: %s\n", path.c_str());
        exit(1);
    }
    std::string json((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    f.close();

    // Find "vocab" array
    size_t vocab_pos = json.find("\"vocab\"");
    if (vocab_pos == std::string::npos) {
        fprintf(stderr, "No vocab found in tokenizer JSON\n");
        exit(1);
    }

    // Find the [ that starts the vocab array
    size_t arr_start = json.find('[', vocab_pos);
    if (arr_start == std::string::npos) {
        fprintf(stderr, "Malformed vocab in tokenizer JSON\n");
        exit(1);
    }

    // Parse vocab entries: [["piece", score], ...]
    size_t pos = arr_start + 1;
    auto skip_ws = [&]() {
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\n' || json[pos] == '\r' || json[pos] == '\t'))
            pos++;
    };

    while (pos < json.size()) {
        skip_ws();
        if (json[pos] == ']') break;
        if (json[pos] == ',') { pos++; continue; }

        // Expect [
        if (json[pos] != '[') break;
        pos++;
        skip_ws();

        // Parse string
        if (json[pos] != '"') break;
        pos++;
        std::string piece;
        while (pos < json.size() && json[pos] != '"') {
            if (json[pos] == '\\') {
                pos++;
                if (pos < json.size()) {
                    switch (json[pos]) {
                        case '"': piece += '"'; break;
                        case '\\': piece += '\\'; break;
                        case '/': piece += '/'; break;
                        case 'n': piece += '\n'; break;
                        case 'r': piece += '\r'; break;
                        case 't': piece += '\t'; break;
                        case 'u': {
                            // Parse unicode escape \uXXXX
                            if (pos + 4 < json.size()) {
                                std::string hex = json.substr(pos + 1, 4);
                                unsigned int cp = std::stoul(hex, nullptr, 16);
                                pos += 4;
                                // Convert to UTF-8
                                if (cp < 0x80) {
                                    piece += (char)cp;
                                } else if (cp < 0x800) {
                                    piece += (char)(0xC0 | (cp >> 6));
                                    piece += (char)(0x80 | (cp & 0x3F));
                                } else {
                                    piece += (char)(0xE0 | (cp >> 12));
                                    piece += (char)(0x80 | ((cp >> 6) & 0x3F));
                                    piece += (char)(0x80 | (cp & 0x3F));
                                }
                            }
                            break;
                        }
                        default: piece += json[pos]; break;
                    }
                    pos++;
                }
            } else {
                piece += json[pos++];
            }
        }
        pos++; // closing "

        skip_ws();
        if (json[pos] == ',') pos++;
        skip_ws();

        // Parse score (float)
        std::string num_str;
        while (pos < json.size() && (json[pos] == '-' || json[pos] == '.' ||
               (json[pos] >= '0' && json[pos] <= '9') || json[pos] == 'e' || json[pos] == 'E' || json[pos] == '+')) {
            num_str += json[pos++];
        }
        float score = 0.0f;
        if (!num_str.empty()) score = std::stof(num_str);

        skip_ws();
        if (json[pos] == ']') pos++; // closing ]

        if (piece.empty()) piece = "<empty_token>";
        vocab.pieces.emplace_back(piece, score);
    }

    // Find unk_id
    size_t unk_pos = json.find("\"unk_id\"");
    if (unk_pos != std::string::npos) {
        size_t colon = json.find(':', unk_pos);
        if (colon != std::string::npos) {
            size_t p = colon + 1;
            while (p < json.size() && json[p] == ' ') p++;
            vocab.unk_id = std::stoi(json.substr(p));
        }
    }

    // Find pre_tokenizer replacement
    size_t pt_pos = json.find("\"pre_tokenizer\"");
    if (pt_pos != std::string::npos) {
        size_t rep_pos = json.find("\"replacement\"", pt_pos);
        if (rep_pos != std::string::npos) {
            size_t colon = json.find(':', rep_pos);
            size_t quote1 = json.find('"', colon + 1);
            size_t quote2 = json.find('"', quote1 + 1);
            if (quote1 != std::string::npos && quote2 != std::string::npos) {
                // Handle unicode escapes in replacement
                std::string rep_raw = json.substr(quote1 + 1, quote2 - quote1 - 1);
                if (rep_raw.find("\\u") != std::string::npos) {
                    // Parse \u2581
                    std::string rep;
                    for (size_t i = 0; i < rep_raw.size(); i++) {
                        if (rep_raw[i] == '\\' && i + 1 < rep_raw.size() && rep_raw[i+1] == 'u') {
                            std::string hex = rep_raw.substr(i + 2, 4);
                            unsigned int cp = std::stoul(hex, nullptr, 16);
                            if (cp < 0x80) {
                                rep += (char)cp;
                            } else if (cp < 0x800) {
                                rep += (char)(0xC0 | (cp >> 6));
                                rep += (char)(0x80 | (cp & 0x3F));
                            } else {
                                rep += (char)(0xE0 | (cp >> 12));
                                rep += (char)(0x80 | ((cp >> 6) & 0x3F));
                                rep += (char)(0x80 | (cp & 0x3F));
                            }
                            i += 5;
                        } else {
                            rep += rep_raw[i];
                        }
                    }
                    vocab.replacement = rep;
                } else {
                    vocab.replacement = rep_raw;
                }
            }
        }

        size_t aps_pos = json.find("\"add_prefix_space\"", pt_pos);
        if (aps_pos != std::string::npos) {
            size_t colon = json.find(':', aps_pos);
            std::string val = json.substr(colon + 1, 10);
            vocab.add_prefix_space = (val.find("true") != std::string::npos);
        }
    }

    printf("T5 tokenizer: loaded %zu vocab entries\n", vocab.pieces.size());
    return vocab;
}

class T5Tokenizer {
    std::vector<std::pair<std::string, float>> piece_score_pairs;
    float min_score_ = FLT_MAX;
    float max_score_ = -FLT_MAX;
    std::unique_ptr<Darts::DoubleArray> trie_;
    int trie_results_size_ = 0;
    int unk_id_ = 2;
    int eos_id_ = 1;
    int pad_id_ = 0;
    float kUnkPenalty = 10.0f;
    std::string replacement_;
    bool add_prefix_space_ = true;

    size_t one_char_len(const char* src) const {
        return "\1\1\1\1\1\1\1\1\1\1\1\1\2\2\3\4"[(*src & 0xFF) >> 4];
    }

public:
    T5Tokenizer() = default;

    void load(const std::string& json_path) {
        auto vocab = parse_tokenizer_json(json_path);
        piece_score_pairs = vocab.pieces;
        unk_id_ = vocab.unk_id;
        replacement_ = vocab.replacement;
        add_prefix_space_ = vocab.add_prefix_space;

        std::vector<std::pair<std::string, int>> pieces;
        for (int i = 0; i < (int)piece_score_pairs.size(); i++) {
            min_score_ = std::min(min_score_, piece_score_pairs[i].second);
            max_score_ = std::max(max_score_, piece_score_pairs[i].second);
            pieces.emplace_back(piece_score_pairs[i].first, i);
        }

        // Build trie
        std::sort(pieces.begin(), pieces.end());
        std::vector<const char*> keys(pieces.size());
        std::vector<int> values(pieces.size());
        for (size_t i = 0; i < pieces.size(); i++) {
            keys[i] = pieces[i].first.c_str();
            values[i] = pieces[i].second;
        }

        trie_ = std::make_unique<Darts::DoubleArray>();
        if (trie_->build(keys.size(), const_cast<char**>(keys.data()), nullptr, values.data()) != 0) {
            fprintf(stderr, "Failed to build trie\n");
            exit(1);
        }

        // Compute max trie results size
        const int kMax = 1024;
        std::vector<Darts::DoubleArray::result_pair_type> results(kMax);
        trie_results_size_ = 0;
        for (const auto& p : pieces) {
            size_t n = trie_->commonPrefixSearch(p.first.c_str(), results.data(), results.size(), p.first.size());
            trie_results_size_ = std::max(trie_results_size_, (int)n);
        }
    }

    std::string pre_tokenize(const std::string& input) const {
        std::string out;
        if (add_prefix_space_) out += replacement_;

        bool first = true;
        std::istringstream ss(input);
        std::string token;
        while (std::getline(ss, token, ' ')) {
            if (!first) out += replacement_ + token;
            else out += token;
            first = false;
        }
        return out;
    }

    std::string normalize(const std::string& input) const {
        return std::regex_replace(input, std::regex(" {2,}"), " ");
    }

    // Viterbi tokenization
    std::vector<std::pair<std::string, int>> encode_optimized(const std::string& normalized) const {
        if (normalized.empty()) return {};

        struct BestPathNode {
            int id = -1;
            float best_path_score = 0;
            int starts_at = -1;
        };

        const int size = (int)normalized.size();
        const float unk_score = min_score_ - kUnkPenalty;
        std::vector<BestPathNode> best_path_ends_at(size + 1);

        int starts_at = 0;
        while (starts_at < size) {
            std::size_t node_pos = 0;
            std::size_t key_pos = starts_at;
            const auto best_score_here = best_path_ends_at[starts_at].best_path_score;
            bool has_single_node = false;
            const int mblen = std::min<int>((int)one_char_len(normalized.data() + starts_at), size - starts_at);

            while (key_pos < (size_t)size) {
                const int ret = trie_->traverse(normalized.data(), node_pos, key_pos, key_pos + 1);
                if (ret == -2) break;
                if (ret >= 0) {
                    auto& target = best_path_ends_at[key_pos];
                    const auto length = (int)(key_pos - starts_at);
                    const auto score = piece_score_pairs[ret].second;
                    const auto candidate = score + best_score_here;
                    if (target.starts_at == -1 || candidate > target.best_path_score) {
                        target.best_path_score = candidate;
                        target.starts_at = starts_at;
                        target.id = ret;
                    }
                    if (!has_single_node && length == mblen) has_single_node = true;
                }
            }
            if (!has_single_node) {
                auto& target = best_path_ends_at[starts_at + mblen];
                const auto candidate = unk_score + best_score_here;
                if (target.starts_at == -1 || candidate > target.best_path_score) {
                    target.best_path_score = candidate;
                    target.starts_at = starts_at;
                    target.id = unk_id_;
                }
            }
            starts_at += mblen;
        }

        // Backtrack
        std::vector<std::pair<std::string, int>> results;
        int ends_at = size;
        while (ends_at > 0) {
            const auto& node = best_path_ends_at[ends_at];
            results.emplace_back(normalized.substr(node.starts_at, ends_at - node.starts_at), node.id);
            ends_at = node.starts_at;
        }
        std::reverse(results.begin(), results.end());
        return results;
    }

    std::vector<int> encode(const std::string& input) const {
        std::string normalized = normalize(input);
        normalized = pre_tokenize(normalized);
        auto result = encode_optimized(normalized);

        std::vector<int> tokens;
        for (auto& p : result) tokens.push_back(p.second);

        // Append EOS
        if (tokens.empty() || tokens.back() != eos_id_) {
            tokens.push_back(eos_id_);
        }
        return tokens;
    }

    // Pad to max_length with pad_id
    void pad(std::vector<int>& tokens, int max_length) const {
        if ((int)tokens.size() < max_length) {
            tokens.resize(max_length, pad_id_);
        }
    }
};
