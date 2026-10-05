#include "mesh.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace {

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    f.seekg(0, std::ios::end);
    std::string data(static_cast<size_t>(f.tellg()), '\0');
    f.seekg(0);
    f.read(data.data(), static_cast<std::streamsize>(data.size()));
    return data;
}

bool ends_with(const std::string& s, const std::string& suffix) {
    if (s.size() < suffix.size()) return false;
    for (size_t i = 0; i < suffix.size(); ++i)
        if (std::tolower(static_cast<unsigned char>(s[s.size() - suffix.size() + i])) != suffix[i]) return false;
    return true;
}

void add_polygon(mesh& m, const uint32_t* v, size_t n, const uint32_t* w = nullptr) {
    for (size_t k = 2; k < n; ++k) {
        m.indices.push_back(v[0]);
        m.indices.push_back(v[k - 1]);
        m.indices.push_back(v[k]);
        if (w) m.corners.insert(m.corners.end(), {w[0], w[k - 1], w[k]});
    }
}

// ply: a text header of elements and properties, then text or binary data.
// keeps vertex x, y, z and face index lists; skips the rest by size.
struct ply_property {
    std::string name, type, count_type;  // count_type set for lists
    bool list = false;
};
struct ply_element {
    std::string name;
    size_t count = 0;
    std::vector<ply_property> properties;
};

size_t type_size(const std::string& t) {
    if (t == "char" || t == "uchar" || t == "int8" || t == "uint8") return 1;
    if (t == "short" || t == "ushort" || t == "int16" || t == "uint16") return 2;
    if (t == "int" || t == "uint" || t == "float" || t == "int32" || t == "uint32" || t == "float32") return 4;
    if (t == "double" || t == "float64") return 8;
    throw std::runtime_error("PLY: unknown type " + t);
}

struct binary_reader {
    const unsigned char* p;
    const unsigned char* end;
    bool big_endian;

    void need(size_t n) const {
        if (static_cast<size_t>(end - p) < n) throw std::runtime_error("PLY: file ends early");
    }
    template <class T>
    T raw() {
        need(sizeof(T));
        unsigned char b[sizeof(T)];
        std::memcpy(b, p, sizeof(T));
        if (big_endian) std::reverse(b, b + sizeof(T));
        p += sizeof(T);
        T v;
        std::memcpy(&v, b, sizeof(T));
        return v;
    }
    double read(const std::string& t) {
        if (t == "char" || t == "int8") return raw<int8_t>();
        if (t == "uchar" || t == "uint8") return raw<uint8_t>();
        if (t == "short" || t == "int16") return raw<int16_t>();
        if (t == "ushort" || t == "uint16") return raw<uint16_t>();
        if (t == "int" || t == "int32") return raw<int32_t>();
        if (t == "uint" || t == "uint32") return raw<uint32_t>();
        if (t == "float" || t == "float32") return raw<float>();
        if (t == "double" || t == "float64") return raw<double>();
        throw std::runtime_error("PLY: unknown type " + t);
    }
};

mesh load_ply(const std::string& data) {
    const size_t header_end = data.find("end_header");
    if (data.compare(0, 3, "ply") != 0 || header_end == std::string::npos) throw std::runtime_error("PLY: no header");
    size_t body = data.find('\n', header_end);
    if (body == std::string::npos) throw std::runtime_error("PLY: no body");
    ++body;

    std::istringstream header(data.substr(0, header_end));
    std::string line, format;
    std::vector<ply_element> elements;
    while (std::getline(header, line)) {
        std::istringstream ls(line);
        std::string word;
        ls >> word;
        if (word == "format") ls >> format;
        else if (word == "element") {
            ply_element e;
            ls >> e.name >> e.count;
            elements.push_back(e);
        } else if (word == "property" && !elements.empty()) {
            ply_property p;
            ls >> p.type;
            if (p.type == "list") {
                p.list = true;
                ls >> p.count_type >> p.type;
            }
            ls >> p.name;
            elements.back().properties.push_back(p);
        }
    }

    mesh m;
    std::vector<uint32_t> poly;
    if (format == "ascii") {
        std::istringstream in(data.substr(body));
        for (const ply_element& e : elements)
            for (size_t i = 0; i < e.count; ++i) {
                vec3 p;
                for (const ply_property& prop : e.properties) {
                    if (prop.list) {
                        size_t n;
                        in >> n;
                        poly.resize(n);
                        for (size_t k = 0; k < n; ++k) {
                            double v;
                            in >> v;
                            poly[k] = static_cast<uint32_t>(v);
                        }
                        if (e.name == "face" && (prop.name == "vertex_indices" || prop.name == "vertex_index"))
                            add_polygon(m, poly.data(), n);
                    } else {
                        double v;
                        in >> v;
                        if (prop.name == "x") p.x = static_cast<float>(v);
                        if (prop.name == "y") p.y = static_cast<float>(v);
                        if (prop.name == "z") p.z = static_cast<float>(v);
                    }
                }
                if (e.name == "vertex") m.positions.push_back(p);
                if (!in) throw std::runtime_error("PLY: bad ASCII data");
            }
    } else if (format == "binary_big_endian" || format == "binary_little_endian") {
        binary_reader r{reinterpret_cast<const unsigned char*>(data.data()) + body,
                        reinterpret_cast<const unsigned char*>(data.data()) + data.size(), format == "binary_big_endian"};
        for (const ply_element& e : elements) {
            // fast path for the common layout: float x y z, faces of a
            // uchar-counted int list.
            const bool xyz = e.name == "vertex" && e.properties.size() >= 3 && e.properties[0].name == "x" &&
                             e.properties[1].name == "y" && e.properties[2].name == "z" &&
                             e.properties[0].type == "float" && e.properties[1].type == "float" && e.properties[2].type == "float";
            if (e.name == "vertex") m.positions.reserve(e.count);
            if (e.name == "face") m.indices.reserve(3 * e.count);
            for (size_t i = 0; i < e.count; ++i) {
                vec3 p;
                size_t first = 0;
                if (xyz) {
                    p.x = r.raw<float>();
                    p.y = r.raw<float>();
                    p.z = r.raw<float>();
                    first = 3;
                }
                for (size_t k = first; k < e.properties.size(); ++k) {
                    const ply_property& prop = e.properties[k];
                    if (prop.list) {
                        const size_t n = static_cast<size_t>(r.read(prop.count_type));
                        poly.resize(n);
                        for (size_t j = 0; j < n; ++j) poly[j] = static_cast<uint32_t>(r.read(prop.type));
                        if (e.name == "face" && (prop.name == "vertex_indices" || prop.name == "vertex_index"))
                            add_polygon(m, poly.data(), n);
                    } else if (e.name == "vertex" && (prop.name == "x" || prop.name == "y" || prop.name == "z")) {
                        const float v = static_cast<float>(r.read(prop.type));
                        (prop.name == "x" ? p.x : prop.name == "y" ? p.y : p.z) = v;
                    } else {
                        r.need(type_size(prop.type));
                        r.p += type_size(prop.type);
                    }
                }
                if (e.name == "vertex") m.positions.push_back(p);
            }
        }
    } else {
        throw std::runtime_error("PLY: unknown format " + format);
    }
    for (uint32_t i : m.indices)
        if (i >= m.positions.size()) throw std::runtime_error("PLY: face index out of range");
    return m;
}

// obj: v, vt and f (v, v/vt, v/vt/vn or v//vn; negative counts back). with vt the mesh is
// textured, a wedge per vt (split where one serves two positions); weld() merges equal ones.
// mtllib's map_Kd names the texture, relative to the obj.
mesh load_obj(const std::string& data, const std::string& path) {
    mesh m;
    std::vector<vec2>& uvs = m.wedge_uvs;
    std::vector<uint32_t> poly, poly_wedges;
    std::string mtllib;
    bool any_uv = false, any_without = false;
    size_t pos = 0;
    while (pos < data.size()) {
        size_t eol = data.find('\n', pos);
        if (eol == std::string::npos) eol = data.size();
        const char* s = data.c_str() + pos;
        const char* end = data.c_str() + eol;
        pos = eol + 1;
        while (s < end && (*s == ' ' || *s == '\t')) ++s;
        if (end - s > 2 && s[0] == 'v' && s[1] == ' ') {
            char* next;
            vec3 p;
            p.x = std::strtof(s + 2, &next);
            p.y = std::strtof(next, &next);
            p.z = std::strtof(next, &next);
            m.positions.push_back(p);
        } else if (end - s > 3 && s[0] == 'v' && s[1] == 't' && s[2] == ' ') {
            char* next;
            vec2 t;
            t.x = std::strtof(s + 3, &next);
            t.y = 1 - std::strtof(next, &next);  // obj's v runs up, textures' rows down
            uvs.push_back(t);
        } else if (end - s > 7 && std::strncmp(s, "mtllib ", 7) == 0) {
            mtllib.assign(s + 7, end);
            while (!mtllib.empty() && (mtllib.back() == '\r' || mtllib.back() == ' ')) mtllib.pop_back();
        } else if (end - s > 2 && s[0] == 'f' && s[1] == ' ') {
            poly.clear();
            poly_wedges.clear();
            const char* q = s + 2;
            bool with_uv = true;
            while (q < end) {
                char* next;
                const long v = std::strtol(q, &next, 10);
                if (next == q) break;
                // negative indices count back from the last vertex.
                const long index = v < 0 ? static_cast<long>(m.positions.size()) + v : v - 1;
                if (index < 0 || index >= static_cast<long>(m.positions.size()))
                    throw std::runtime_error("OBJ: face index out of range");
                poly.push_back(static_cast<uint32_t>(index));
                q = next;
                long t = 0;
                if (q < end && *q == '/' && q + 1 < end && q[1] != '/') {
                    t = std::strtol(q + 1, &next, 10);
                    q = next;
                }
                if (t == 0) with_uv = false;
                else {
                    const long ti = t < 0 ? static_cast<long>(uvs.size()) + t : t - 1;
                    if (ti < 0 || ti >= static_cast<long>(uvs.size())) throw std::runtime_error("OBJ: texture index out of range");
                    poly_wedges.push_back(uint32_t(ti));
                }
                while (q < end && *q != ' ' && *q != '\t') ++q;  // skip /vn
                while (q < end && (*q == ' ' || *q == '\t' || *q == '\r')) ++q;
            }
            any_uv |= with_uv;
            any_without |= !with_uv;
            add_polygon(m, poly.data(), poly.size(), with_uv ? poly_wedges.data() : nullptr);
        }
    }
    if (any_uv && any_without) throw std::runtime_error("OBJ: some faces have texture coordinates and some not");
    if (!any_uv) {
        m.wedge_uvs.clear();
        return m;
    }
    // a wedge per vt, at the position its corners use; a vt at two positions splits.
    m.wedge_vertex.assign(uvs.size(), UINT32_MAX);
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> split;
    for (size_t i = 0; i < m.corners.size(); ++i) {
        uint32_t& w = m.corners[i];
        const uint32_t v = m.indices[i];
        if (m.wedge_vertex[w] == UINT32_MAX) m.wedge_vertex[w] = v;
        else if (m.wedge_vertex[w] != v) {
            auto [it, added] = split.emplace(std::make_pair(w, v), uint32_t(uvs.size()));
            if (added) {
                uvs.push_back(uvs[w]);
                m.wedge_vertex.push_back(v);
            }
            w = it->second;
        }
    }
    // the texture, from the material library.
    if (!mtllib.empty()) {
        const std::string dir = path.substr(0, path.find_last_of('/') + 1);
        std::ifstream f(dir + mtllib);
        std::string line;
        while (std::getline(f, line)) {
            size_t a = line.find_first_not_of(" \t");
            if (a == std::string::npos || line.compare(a, 7, "map_Kd ") != 0) continue;
            std::string name = line.substr(a + 7);
            while (!name.empty() && (name.back() == '\r' || name.back() == ' ')) name.pop_back();
            m.texture = dir + name.substr(name.find_last_of(' ') == std::string::npos ? 0 : name.find_last_of(' ') + 1);
            break;
        }
    }
    return m;
}

}  // namespace

mesh load_mesh(const std::string& path) {
    const std::string data = read_file(path);
    if (ends_with(path, ".ply")) return load_ply(data);
    if (ends_with(path, ".obj")) return load_obj(data, path);
    throw std::runtime_error("unknown mesh format: " + path + " (want .ply or .obj)");
}

void weld(mesh& m) {
    struct key_hash {
        size_t operator()(const vec3& p) const {
            uint32_t b[3];
            std::memcpy(b, &p, sizeof b);
            return (b[0] * 73856093u) ^ (b[1] * 19349663u) ^ (b[2] * 83492791u);
        }
    };
    struct key_eq {
        bool operator()(const vec3& a, const vec3& b) const { return a.x == b.x && a.y == b.y && a.z == b.z; }
    };
    std::unordered_map<vec3, uint32_t, key_hash, key_eq> index;
    index.reserve(m.positions.size());
    std::vector<uint32_t> remap(m.positions.size());
    std::vector<vec3> positions;
    for (size_t i = 0; i < m.positions.size(); ++i) {
        auto [it, added] = index.emplace(m.positions[i], static_cast<uint32_t>(positions.size()));
        if (added) positions.push_back(m.positions[i]);
        remap[i] = it->second;
    }

    // drop degenerate and repeated triangles (same vertices, same winding).
    struct tri_hash {
        size_t operator()(const std::array<uint32_t, 3>& t) const {
            return (t[0] * 2654435761u) ^ (t[1] * 40503u) ^ (t[2] * 2246822519u);
        }
    };
    std::unordered_map<std::array<uint32_t, 3>, char, tri_hash> seen;
    seen.reserve(m.triangle_count());
    std::vector<uint32_t> indices, corners;
    indices.reserve(m.indices.size());
    for (size_t t = 0; t < m.triangle_count(); ++t) {
        uint32_t a = remap[m.indices[3 * t]], b = remap[m.indices[3 * t + 1]], c = remap[m.indices[3 * t + 2]];
        if (a == b || b == c || a == c) continue;
        // smallest index first, winding kept.
        std::array<uint32_t, 3> k = {a, b, c};
        while (k[0] != std::min({a, b, c})) std::rotate(k.begin(), k.begin() + 1, k.end());
        if (!seen.emplace(k, 0).second) continue;
        indices.insert(indices.end(), {a, b, c});
        if (m.textured()) corners.insert(corners.end(), &m.corners[3 * t], &m.corners[3 * t + 3]);
    }

    // drop unused vertices.
    std::vector<uint32_t> used(positions.size(), UINT32_MAX);
    m.positions.clear();
    for (uint32_t& i : indices) {
        if (used[i] == UINT32_MAX) {
            used[i] = static_cast<uint32_t>(m.positions.size());
            m.positions.push_back(positions[i]);
        }
        i = used[i];
    }
    m.indices = std::move(indices);

    // wedges: one per welded vertex and texture coordinate, only those still used. each
    // vertex's corners in a row (counting sort), equal coordinates merged within it.
    if (!m.textured()) return;
    std::vector<uint32_t> start(m.positions.size() + 1, 0), order(corners.size());
    for (uint32_t v : m.indices) ++start[v + 1];
    for (size_t v = 0; v < m.positions.size(); ++v) start[v + 1] += start[v];
    {
        std::vector<uint32_t> fill(start.begin(), start.end() - 1);
        for (size_t i = 0; i < corners.size(); ++i) order[fill[m.indices[i]]++] = uint32_t(i);
    }
    std::vector<vec2> uvs;
    std::vector<uint32_t> wedge_vertex;
    for (size_t v = 0; v < m.positions.size(); ++v) {
        const size_t first_wedge = uvs.size();
        for (uint32_t k = start[v]; k < start[v + 1]; ++k) {
            const uint32_t i = order[k];
            const vec2 uv = m.wedge_uvs[corners[i]];
            size_t w = first_wedge;
            while (w < uvs.size() && (uvs[w].x != uv.x || uvs[w].y != uv.y)) ++w;
            if (w == uvs.size()) {
                uvs.push_back(uv);
                wedge_vertex.push_back(uint32_t(v));
            }
            corners[i] = uint32_t(w);
        }
    }
    m.corners = std::move(corners);
    m.wedge_uvs = std::move(uvs);
    m.wedge_vertex = std::move(wedge_vertex);
}

void normalize_placement(mesh& m, bool up_z) {
    if (m.positions.empty()) return;
    if (up_z)
        for (vec3& p : m.positions) p = {p.x, p.z, -p.y};
    // wind counterclockwise from outside (shadow rays skip back faces). the
    // signed volume decides, holes or not.
    double volume = 0;
    for (size_t t = 0; t < m.triangle_count(); ++t) {
        const vec3 a = m.positions[m.indices[3 * t]], b = m.positions[m.indices[3 * t + 1]], c = m.positions[m.indices[3 * t + 2]];
        volume += dot(a, cross(b, c));
    }
    if (volume < 0)
        for (size_t t = 0; t < m.triangle_count(); ++t) {
            std::swap(m.indices[3 * t + 1], m.indices[3 * t + 2]);
            if (m.textured()) std::swap(m.corners[3 * t + 1], m.corners[3 * t + 2]);
        }
    vec3 lo = m.positions[0], hi = lo;
    for (const vec3& p : m.positions) { lo = min(lo, p); hi = max(hi, p); }
    const vec3 size = hi - lo;
    const float scale = 1 / std::max({size.x, size.y, size.z});
    const vec3 base = {0.5f * (lo.x + hi.x), lo.y, 0.5f * (lo.z + hi.z)};
    for (vec3& p : m.positions) p = (p - base) * scale;
}

std::vector<vec3> vertex_normals(const mesh& m) {
    std::vector<vec3> n(m.positions.size());
    for (size_t t = 0; t < m.triangle_count(); ++t) {
        const uint32_t* i = &m.indices[3 * t];
        // cross product length is twice the area: an area weight.
        const vec3 fn = cross(m.positions[i[1]] - m.positions[i[0]], m.positions[i[2]] - m.positions[i[0]]);
        for (int c = 0; c < 3; ++c) n[i[c]] += fn;
    }
    for (vec3& v : n) v = normalize(v);
    return n;
}
