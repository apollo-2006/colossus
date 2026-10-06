#include "gltf.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace {

// json: just enough for gltf.
struct json {
    enum kind_t { null, boolean, number, string, array, object } kind = null;
    double num = 0;
    bool b = false;
    std::string str;
    std::vector<json> items;
    std::map<std::string, json> fields;

    const json& operator[](const std::string& k) const {
        static const json none;
        auto it = fields.find(k);
        return it == fields.end() ? none : it->second;
    }
    const json& operator[](size_t i) const {
        static const json none;
        return i < items.size() ? items[i] : none;
    }
    bool has(const std::string& k) const { return fields.count(k) != 0; }
    size_t size() const { return items.size(); }
    double num_or(double d) const { return kind == number ? num : d; }
    size_t index() const {
        if (kind != number || num < 0 || num != std::floor(num)) throw std::runtime_error("glTF: bad index");
        return size_t(num);
    }
};

struct json_parser {
    const std::string& s;
    size_t at = 0;
    int depth = 0;

    [[noreturn]] void fail(const char* what) { throw std::runtime_error(std::string("glTF json: ") + what); }
    void space() {
        while (at < s.size() && (s[at] == ' ' || s[at] == '\n' || s[at] == '\r' || s[at] == '\t')) ++at;
    }
    bool take(char c) {
        space();
        if (at < s.size() && s[at] == c) { ++at; return true; }
        return false;
    }
    std::string text() {
        if (!take('"')) fail("want a string");
        std::string out;
        while (at < s.size() && s[at] != '"') {
            char c = s[at++];
            if (c == '\\') {
                if (at >= s.size()) fail("ends in a string");
                const char e = s[at++];
                if (e == 'u') {  // kept as utf-8; names only
                    if (at + 4 > s.size()) fail("bad escape");
                    const unsigned cp = std::stoul(s.substr(at, 4), nullptr, 16);
                    at += 4;
                    if (cp < 0x80) out += char(cp);
                    else if (cp < 0x800) { out += char(0xc0 | cp >> 6); out += char(0x80 | (cp & 63)); }
                    else { out += char(0xe0 | cp >> 12); out += char(0x80 | (cp >> 6 & 63)); out += char(0x80 | (cp & 63)); }
                    continue;
                }
                c = e == 'n' ? '\n' : e == 't' ? '\t' : e == 'r' ? '\r' : e == 'b' ? '\b' : e == 'f' ? '\f' : e;
            }
            out += c;
        }
        if (!take('"')) fail("unterminated string");
        return out;
    }
    json value() {
        if (++depth > 64) fail("nested too deep");
        space();
        if (at >= s.size()) fail("ends early");
        json v;
        const char c = s[at];
        if (c == '{') {
            ++at;
            v.kind = json::object;
            if (!take('}')) {
                do {
                    std::string k = text();
                    if (!take(':')) fail("want :");
                    v.fields[k] = value();
                } while (take(','));
                if (!take('}')) fail("want }");
            }
        } else if (c == '[') {
            ++at;
            v.kind = json::array;
            if (!take(']')) {
                do v.items.push_back(value());
                while (take(','));
                if (!take(']')) fail("want ]");
            }
        } else if (c == '"') {
            v.kind = json::string;
            v.str = text();
        } else if (s.compare(at, 4, "true") == 0) { v.kind = json::boolean; v.b = true; at += 4; }
        else if (s.compare(at, 5, "false") == 0) { v.kind = json::boolean; at += 5; }
        else if (s.compare(at, 4, "null") == 0) at += 4;
        else {
            char* end;
            v.num = std::strtod(s.c_str() + at, &end);
            if (end == s.c_str() + at) fail("unexpected character");
            v.kind = json::number;
            at = size_t(end - s.c_str());
        }
        --depth;
        return v;
    }
};

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::string base64(const std::string& in) {
    std::string out;
    uint32_t acc = 0;
    int bits = 0;
    for (char c : in) {
        int v = c >= 'A' && c <= 'Z' ? c - 'A' : c >= 'a' && c <= 'z' ? c - 'a' + 26 : c >= '0' && c <= '9' ? c - '0' + 52
                : c == '+' ? 62 : c == '/' ? 63 : -1;
        if (v < 0) continue;
        acc = acc << 6 | uint32_t(v);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out += char(acc >> bits & 255);
        }
    }
    return out;
}

struct gltf_file {
    json root;
    std::vector<std::string> buffers;
    std::string dir;

    // an accessor's elements as floats (normalized integers to [0, 1] or [-1, 1]).
    std::vector<float> floats(size_t index, int& width) const {
        const json& a = root["accessors"][index];
        if (a.has("sparse")) throw std::runtime_error("glTF: sparse accessors are not supported");
        static const std::map<std::string, int> widths = {{"SCALAR", 1}, {"VEC2", 2}, {"VEC3", 3}, {"VEC4", 4}, {"MAT4", 16}};
        auto w = widths.find(a["type"].str);
        if (w == widths.end()) throw std::runtime_error("glTF: unsupported accessor type " + a["type"].str);
        width = w->second;
        const size_t count = a["count"].index();
        const int type = int(a["componentType"].num);
        const size_t component = type == 5126 || type == 5125 ? 4 : type == 5123 || type == 5122 ? 2 : 1;
        const bool normalized = a["normalized"].b;
        std::vector<float> out(count * width, 0.0f);
        if (!a.has("bufferView")) return out;  // all zero
        const json& view = root["bufferViews"][a["bufferView"].index()];
        const std::string& buf = buffers.at(view["buffer"].index());
        const size_t stride = view.has("byteStride") ? view["byteStride"].index() : component * width;
        const size_t base = size_t(view["byteOffset"].num_or(0)) + size_t(a["byteOffset"].num_or(0));
        if (count > 0 && base + (count - 1) * stride + component * width > buf.size()) throw std::runtime_error("glTF: accessor past its buffer");
        for (size_t i = 0; i < count; ++i)
            for (int k = 0; k < width; ++k) {
                const char* p = buf.data() + base + i * stride + k * component;
                float v;
                switch (type) {
                    case 5126: std::memcpy(&v, p, 4); break;
                    case 5125: { uint32_t x; std::memcpy(&x, p, 4); v = float(x); break; }
                    case 5123: { uint16_t x; std::memcpy(&x, p, 2); v = normalized ? x / 65535.0f : float(x); break; }
                    case 5122: { int16_t x; std::memcpy(&x, p, 2); v = normalized ? std::max(x / 32767.0f, -1.0f) : float(x); break; }
                    case 5121: { uint8_t x = uint8_t(*p); v = normalized ? x / 255.0f : float(x); break; }
                    case 5120: { int8_t x = int8_t(*p); v = normalized ? std::max(x / 127.0f, -1.0f) : float(x); break; }
                    default: throw std::runtime_error("glTF: unsupported component type");
                }
                out[i * width + k] = v;
            }
        return out;
    }
};

// column major 4x4.
struct mat4c {
    float m[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
};
mat4c mul(const mat4c& a, const mat4c& b) {
    mat4c r;
    for (int c = 0; c < 4; ++c)
        for (int row = 0; row < 4; ++row) {
            float s = 0;
            for (int k = 0; k < 4; ++k) s += a.m[k * 4 + row] * b.m[c * 4 + k];
            r.m[c * 4 + row] = s;
        }
    return r;
}
mat4c trs(const float* t, const float* q, const float* s) {
    const float x = q[0], y = q[1], z = q[2], w = q[3];
    mat4c r;
    const float rot[9] = {1 - 2 * (y * y + z * z), 2 * (x * y + z * w),     2 * (x * z - y * w),
                          2 * (x * y - z * w),     1 - 2 * (x * x + z * z), 2 * (y * z + x * w),
                          2 * (x * z + y * w),     2 * (y * z - x * w),     1 - 2 * (x * x + y * y)};
    for (int c = 0; c < 3; ++c)
        for (int row = 0; row < 3; ++row) r.m[c * 4 + row] = rot[c * 3 + row] * s[c];
    r.m[12] = t[0]; r.m[13] = t[1]; r.m[14] = t[2];
    return r;
}
mat4c node_local(const json& n) {
    if (n.has("matrix")) {
        mat4c r;
        for (int i = 0; i < 16; ++i) r.m[i] = float(n["matrix"][i].num_or(i % 5 == 0));
        return r;
    }
    float t[3] = {0, 0, 0}, q[4] = {0, 0, 0, 1}, s[3] = {1, 1, 1};
    for (int i = 0; i < 3; ++i) t[i] = float(n["translation"][i].num_or(0));
    for (int i = 0; i < 4; ++i) q[i] = float(n["rotation"][i].num_or(i == 3));
    for (int i = 0; i < 3; ++i) s[i] = float(n["scale"][i].num_or(1));
    return trs(t, q, s);
}

}  // namespace

mesh load_gltf(const std::string& path, skeleton* skin, const std::string& only_node) {
    gltf_file g;
    g.dir = path.substr(0, path.find_last_of('/') + 1);
    std::string data = read_file(path);
    std::string glb_bin;
    if (data.size() >= 12 && data.compare(0, 4, "glTF") == 0) {  // binary: json chunk, then bin chunk
        size_t at = 12;
        std::string text;
        while (at + 8 <= data.size()) {
            uint32_t len, type;
            std::memcpy(&len, &data[at], 4);
            std::memcpy(&type, &data[at + 4], 4);
            if (at + 8 + len > data.size()) throw std::runtime_error("glTF: chunk past the file");
            if (type == 0x4e4f534a) text = data.substr(at + 8, len);
            else if (type == 0x004e4942) glb_bin = data.substr(at + 8, len);
            at += 8 + ((len + 3) & ~3u);
        }
        data = std::move(text);
    }
    json_parser parser{data};
    g.root = parser.value();
    for (size_t i = 0; i < g.root["buffers"].size(); ++i) {
        const json& b = g.root["buffers"][i];
        if (!b.has("uri")) g.buffers.push_back(glb_bin);
        else if (b["uri"].str.compare(0, 5, "data:") == 0) g.buffers.push_back(base64(b["uri"].str.substr(b["uri"].str.find(',') + 1)));
        else g.buffers.push_back(read_file(g.dir + b["uri"].str));
    }

    // node parents and world transforms.
    const json& nodes = g.root["nodes"];
    std::vector<int> parent(nodes.size(), -1);
    for (size_t i = 0; i < nodes.size(); ++i)
        for (size_t k = 0; k < nodes[i]["children"].size(); ++k) {
            const size_t c = nodes[i]["children"][k].index();
            if (c >= nodes.size() || parent[c] != -1) throw std::runtime_error("glTF: bad node tree");
            parent[c] = int(i);
        }
    std::vector<mat4c> world(nodes.size());
    std::vector<int> done(nodes.size(), 0);
    auto world_of = [&](auto&& self, size_t i) -> const mat4c& {
        if (done[i] == 1) return world[i];
        if (done[i] == 2) throw std::runtime_error("glTF: node cycle");
        done[i] = 2;
        world[i] = parent[i] < 0 ? node_local(nodes[i]) : mul(self(self, size_t(parent[i])), node_local(nodes[i]));
        done[i] = 1;
        return world[i];
    };

    mesh m;
    int skin_index = -1;
    std::vector<vec2> uvs;
    std::vector<uint8_t> joints, weights, vertex_material;
    // materials: the gltf's, in order, plus a default for primitives without one. textured if
    // any has a base colour texture.
    const json& mats = g.root["materials"];
    bool textured = false;
    for (size_t i = 0; i < mats.size(); ++i) textured |= mats[i]["pbrMetallicRoughness"]["baseColorTexture"].has("index");
    for (size_t i = 0; i <= mats.size(); ++i) {
        material mat;
        if (i < mats.size()) {
            const json& pbr = mats[i]["pbrMetallicRoughness"];
            for (int k = 0; k < 3; ++k) mat.color[k] = float(pbr["baseColorFactor"][size_t(k)].num_or(1));
            mat.double_sided = mats[i]["doubleSided"].b;
            // a texture's image file, if it has one (embedded data isn't read).
            auto image_of = [&](const json& t) {
                if (!t.has("index")) return std::string();
                const json& img = g.root["images"][g.root["textures"][t["index"].index()]["source"].index()];
                return img.has("uri") && img["uri"].str.compare(0, 5, "data:") != 0 ? g.dir + img["uri"].str : std::string();
            };
            mat.normal_map = image_of(mats[i]["normalTexture"]);
            mat.roughness_map = image_of(pbr["metallicRoughnessTexture"]);
            const json& t = pbr["baseColorTexture"];
            if (t.has("index")) {
                const json& tex = g.root["textures"][t["index"].index()];
                mat.texture = image_of(t);
                const json& smp = tex.has("sampler") ? g.root["samplers"][tex["sampler"].index()] : json();
                mat.repeat = smp["wrapS"].num_or(10497) != 33071 || smp["wrapT"].num_or(10497) != 33071;  // anything but clamp
            }
        }
        m.materials.push_back(mat);
        if (m.materials.size() > 255) throw std::runtime_error("glTF: more than 255 materials");
    }
    // a material's texture transform (KHR_texture_transform): offset, rotation, scale.
    auto transform_uv = [&](int mi, vec2 uv) {
        if (mi < 0 || size_t(mi) >= mats.size()) return uv;
        const json& x = mats[size_t(mi)]["pbrMetallicRoughness"]["baseColorTexture"]["extensions"]["KHR_texture_transform"];
        if (x.kind != json::object) return uv;
        const float sx = float(x["scale"][0].num_or(1)), sy = float(x["scale"][1].num_or(1)), r = float(x["rotation"].num_or(0));
        const float c = std::cos(r), s = std::sin(r), u = uv.x * sx, v = uv.y * sy;
        return vec2{c * u + s * v + float(x["offset"][0].num_or(0)), -s * u + c * v + float(x["offset"][1].num_or(0))};
    };
    for (size_t ni = 0; ni < nodes.size(); ++ni) {
        const json& n = nodes[ni];
        if (!n.has("mesh")) continue;
        if (!only_node.empty() && n["name"].str != only_node) continue;
        const bool skinned = n.has("skin");
        if (skinned) {
            if (skin_index >= 0 && int(n["skin"].index()) != skin_index) throw std::runtime_error("glTF: more than one skin");
            skin_index = int(n["skin"].index());
        }
        const mat4c& w = world_of(world_of, ni);
        const json& prims = g.root["meshes"][n["mesh"].index()]["primitives"];
        for (size_t pi = 0; pi < prims.size(); ++pi) {
            const json& p = prims[pi];
            if (p["mode"].num_or(4) != 4) continue;  // triangles only
            int width;
            const std::vector<float> pos = g.floats(p["attributes"]["POSITION"].index(), width);
            if (width != 3) throw std::runtime_error("glTF: POSITION is not vec3");
            const uint32_t first = uint32_t(m.positions.size());
            const size_t count = pos.size() / 3;
            for (size_t i = 0; i < count; ++i) {
                vec3 v(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
                if (!skinned) {
                    v = vec3(w.m[0] * v.x + w.m[4] * v.y + w.m[8] * v.z + w.m[12], w.m[1] * v.x + w.m[5] * v.y + w.m[9] * v.z + w.m[13],
                             w.m[2] * v.x + w.m[6] * v.y + w.m[10] * v.z + w.m[14]);
                }
                m.positions.push_back(v);
            }
            if (p["attributes"].has("TEXCOORD_0")) {
                const std::vector<float> t = g.floats(p["attributes"]["TEXCOORD_0"].index(), width);
                if (width != 2 || t.size() / 2 != count) throw std::runtime_error("glTF: bad TEXCOORD_0");
                const int mi = p.has("material") ? int(p["material"].index()) : -1;
                for (size_t i = 0; i < count; ++i) uvs.push_back(transform_uv(mi, {t[2 * i], t[2 * i + 1]}));  // gltf's v already runs down
            } else {
                uvs.resize(uvs.size() + count);  // a flat colour, or a texture's corner
            }
            if (skinned) {
                if (!p["attributes"].has("JOINTS_0") || !p["attributes"].has("WEIGHTS_0"))
                    throw std::runtime_error("glTF: a skinned mesh without JOINTS_0 and WEIGHTS_0");
                const std::vector<float> j = g.floats(p["attributes"]["JOINTS_0"].index(), width);
                if (width != 4 || j.size() / 4 != count) throw std::runtime_error("glTF: bad JOINTS_0");
                const std::vector<float> wt = g.floats(p["attributes"]["WEIGHTS_0"].index(), width);
                if (width != 4 || wt.size() / 4 != count) throw std::runtime_error("glTF: bad WEIGHTS_0");
                for (size_t i = 0; i < count; ++i) {
                    // weights to 8 bits summing to 255: rounded, the remainder on the largest.
                    float total = 0;
                    for (int k = 0; k < 4; ++k) total += std::max(wt[4 * i + k], 0.0f);
                    if (total <= 0) throw std::runtime_error("glTF: a vertex with no weight");
                    uint8_t q[4];
                    int sum = 0, largest = 0;
                    for (int k = 0; k < 4; ++k) {
                        q[k] = uint8_t(std::lround(std::max(wt[4 * i + k], 0.0f) / total * 255));
                        sum += q[k];
                        if (wt[4 * i + k] > wt[4 * i + largest]) largest = k;
                    }
                    q[largest] = uint8_t(q[largest] + 255 - sum);
                    for (int k = 0; k < 4; ++k) {
                        if (j[4 * i + k] > 255) throw std::runtime_error("glTF: more than 256 joints");
                        joints.push_back(q[k] ? uint8_t(j[4 * i + k]) : 0);
                        weights.push_back(q[k]);
                    }
                }
            } else if (skin_index >= 0 || !joints.empty()) {
                throw std::runtime_error("glTF: skinned and unskinned meshes together are not supported");
            }
            const int mi = p.has("material") ? int(p["material"].index()) : int(mats.size());
            vertex_material.insert(vertex_material.end(), count, uint8_t(mi));
            if (p.has("indices")) {
                const std::vector<float> idx = g.floats(p["indices"].index(), width);
                for (float v : idx) {
                    if (v < 0 || v >= float(count)) throw std::runtime_error("glTF: index out of range");
                    m.indices.push_back(first + uint32_t(v));
                }
                if (idx.size() % 3) throw std::runtime_error("glTF: index count not a multiple of 3");
            } else {
                if (count % 3) throw std::runtime_error("glTF: vertex count not a multiple of 3");
                for (size_t i = 0; i < count; ++i) m.indices.push_back(first + uint32_t(i));
            }
            if (!skinned && skin_index >= 0) throw std::runtime_error("glTF: skinned and unskinned meshes together are not supported");
        }
    }
    if (m.indices.empty()) throw std::runtime_error(path + (only_node.empty() ? "" : " node " + only_node) + ": no triangles");
    // the default material, last, only if a primitive has none.
    if (std::find(vertex_material.begin(), vertex_material.end(), uint8_t(mats.size())) == vertex_material.end()) m.materials.pop_back();
    if (textured) {  // a wedge per vertex; weld() merges equal ones
        m.wedge_uvs = uvs;
        m.wedge_vertex.resize(uvs.size());
        for (uint32_t i = 0; i < uvs.size(); ++i) m.wedge_vertex[i] = i;
        m.wedge_material = vertex_material;
        m.corners = m.indices;
    } else {
        m.materials.clear();
    }
    m.skin_joints = std::move(joints);
    m.skin_weights = std::move(weights);

    if (skin_index >= 0 && skin) {
        const json& s = g.root["skins"][size_t(skin_index)];
        skeleton& out = *skin;
        out = skeleton();
        if (s["joints"].size() > 64) throw std::runtime_error("glTF: more than 64 joints");
        for (size_t i = 0; i < nodes.size(); ++i) {
            skeleton_node sn;
            sn.parent = parent[i];
            const json& n = nodes[i];
            if (n.has("matrix")) {
                // kept as trs: translation and the columns' lengths and directions.
                const mat4c l = node_local(n);
                for (int k = 0; k < 3; ++k) sn.translation[k] = l.m[12 + k];
                float r[9];
                for (int c = 0; c < 3; ++c) {
                    sn.scale[c] = std::sqrt(l.m[c * 4] * l.m[c * 4] + l.m[c * 4 + 1] * l.m[c * 4 + 1] + l.m[c * 4 + 2] * l.m[c * 4 + 2]);
                    for (int k = 0; k < 3; ++k) r[c * 3 + k] = l.m[c * 4 + k] / std::max(sn.scale[c], 1e-30f);
                }
                // rotation matrix (columns) to a quaternion.
                const float tr = r[0] + r[4] + r[8];
                float* q = sn.rotation;
                if (tr > 0) {
                    const float k = std::sqrt(tr + 1) * 2;
                    q[3] = 0.25f * k; q[0] = (r[5] - r[7]) / k; q[1] = (r[6] - r[2]) / k; q[2] = (r[1] - r[3]) / k;
                } else if (r[0] > r[4] && r[0] > r[8]) {
                    const float k = std::sqrt(1 + r[0] - r[4] - r[8]) * 2;
                    q[3] = (r[5] - r[7]) / k; q[0] = 0.25f * k; q[1] = (r[3] + r[1]) / k; q[2] = (r[6] + r[2]) / k;
                } else if (r[4] > r[8]) {
                    const float k = std::sqrt(1 + r[4] - r[0] - r[8]) * 2;
                    q[3] = (r[6] - r[2]) / k; q[0] = (r[3] + r[1]) / k; q[1] = 0.25f * k; q[2] = (r[7] + r[5]) / k;
                } else {
                    const float k = std::sqrt(1 + r[8] - r[0] - r[4]) * 2;
                    q[3] = (r[1] - r[3]) / k; q[0] = (r[6] + r[2]) / k; q[1] = (r[7] + r[5]) / k; q[2] = 0.25f * k;
                }
            } else {
                for (int k = 0; k < 3; ++k) sn.translation[k] = float(n["translation"][k].num_or(0));
                for (int k = 0; k < 4; ++k) sn.rotation[k] = float(n["rotation"][k].num_or(k == 3));
                for (int k = 0; k < 3; ++k) sn.scale[k] = float(n["scale"][k].num_or(1));
            }
            out.nodes.push_back(sn);
        }
        for (size_t i = 0; i < s["joints"].size(); ++i) {
            const size_t j = s["joints"][i].index();
            if (j >= nodes.size()) throw std::runtime_error("glTF: joint out of range");
            out.joints.push_back(uint32_t(j));
        }
        if (s.has("inverseBindMatrices")) {
            int width;
            out.inverse_bind = g.floats(s["inverseBindMatrices"].index(), width);
            if (width != 16 || out.inverse_bind.size() != 16 * out.joints.size()) throw std::runtime_error("glTF: bad inverseBindMatrices");
        } else {
            out.inverse_bind.assign(16 * out.joints.size(), 0.0f);
            for (size_t j = 0; j < out.joints.size(); ++j)
                for (int k = 0; k < 4; ++k) out.inverse_bind[16 * j + 5 * k] = 1;
        }
        for (uint8_t j : m.skin_joints)
            if (j >= out.joints.size()) throw std::runtime_error("glTF: vertex joint out of range");
        const json& anims = g.root["animations"];
        for (size_t a = 0; a < anims.size(); ++a) {
            animation an;
            an.name = anims[a]["name"].kind == json::string ? anims[a]["name"].str : "animation " + std::to_string(a);
            for (size_t c = 0; c < anims[a]["channels"].size(); ++c) {
                const json& ch = anims[a]["channels"][c];
                const std::string& p = ch["target"]["path"].str;
                if (p != "translation" && p != "rotation" && p != "scale") continue;  // weights: morph targets, unsupported
                const json& sm = anims[a]["samplers"][ch["sampler"].index()];
                animation_channel out_ch;
                out_ch.node = uint32_t(ch["target"]["node"].index());
                if (out_ch.node >= nodes.size()) throw std::runtime_error("glTF: animated node out of range");
                out_ch.path = p == "translation" ? 0 : p == "rotation" ? 1 : 2;
                const std::string interp = sm["interpolation"].kind == json::string ? sm["interpolation"].str : "LINEAR";
                out_ch.step = interp == "STEP";
                int w;
                out_ch.times = g.floats(sm["input"].index(), w);
                std::vector<float> values = g.floats(sm["output"].index(), w);
                const int want = out_ch.path == 1 ? 4 : 3;
                if (w != want) throw std::runtime_error("glTF: animation values of the wrong width");
                if (interp == "CUBICSPLINE") {  // in-tangent, value, out-tangent per key: values kept
                    std::vector<float> v;
                    for (size_t k = 0; k < out_ch.times.size(); ++k)
                        v.insert(v.end(), values.begin() + (3 * k + 1) * want, values.begin() + (3 * k + 2) * want);
                    values = std::move(v);
                }
                if (values.size() != out_ch.times.size() * want || out_ch.times.empty()) throw std::runtime_error("glTF: animation keys and values disagree");
                out_ch.values = std::move(values);
                an.duration = std::max(an.duration, out_ch.times.back());
                an.channels.push_back(std::move(out_ch));
            }
            if (!an.channels.empty()) out.animations.push_back(std::move(an));
        }
    }
    return m;
}
