#pragma once
#include <cmath>

// --- Scalar Functions ---

inline float atan(float y, float x) {
    return std::atan2(y, x);
}

inline float fract(float x) {
    return x - std::floor(x);
}

inline float sign(float x) {
    return (x > 0.0f) - (x < 0.0f);
}


inline float mix(float x, float y, float a) {
    return x * (1.0f - a) + y * a;
}

inline float clamp(float x, float minVal, float maxVal) {
    return std::fmax(minVal, std::fmin(maxVal, x));
}

inline float mod(float x, float y) {
    return x - y * std::floor(x / y);
}

inline float radians(float degrees) {
    return degrees * (3.14159265358979323846f / 180.0f);
}

inline float degrees(float radians) {
    return radians * (180.0f / 3.14159265358979323846f);
}
// smoothstep
inline float smoothstep(float edge0, float edge1, float x) {
    float t = clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}
inline float length(float x) {
    return std::abs(x);
}



// --- Vector Classes ---
class vec2;
class vec3;
struct vec2_proxy {
    float* a;
    float* b;
    vec2_proxy() : a(nullptr), b(nullptr) {}
    vec2_proxy(float& a_, float& b_) : a(&a_), b(&b_) {}
    operator vec2() const;
    vec2_proxy& operator=(const vec2& v);
};

struct vec3_proxy {
    float* a;
    float* b;
    float* c;

    vec3_proxy() : a(nullptr), b(nullptr), c(nullptr) {}
    vec3_proxy(float& a_, float& b_, float & c_) : a(&a_), b(&b_), c(&c_) {}

    operator vec3() const;
    vec3_proxy& operator=(const vec3& v);
};

// vec2
class vec2 {
public:
    float x, y;

    vec3_proxy xyy,yyx,yxy,xxx;
    vec2() : vec2(0.0f, 0.0f) {}
    vec2(float s_ ) : vec2(s_, s_) {}
    vec2(float x_, float y_) : x(x_), y(y_), xyy(x,y,y),yyx(y,y,x),yxy(y,x,y),xxx(x,x,x) {}

    // Arithmetic
    vec2 operator+(const vec2& v) const { return vec2(x + v.x, y + v.y); }
    vec2 operator-(const vec2& v) const { return vec2(x - v.x, y - v.y); }
    vec2 operator*(const vec2& v) const { return vec2(x * v.x, y * v.y); }
    vec2 operator/(const vec2& v) const { return vec2(x / v.x, y / v.y); }
    vec2 operator*(float s) const { return vec2(x * s, y * s); }
    vec2 operator/(float s) const { return vec2(x / s, y / s); }
    vec2 operator+(float s) const { return vec2(x + s, y + s); }
    vec2 operator-(float s) const { return vec2(x - s, y - s); }
    // += etc.
    vec2& operator+=(const vec2& v) { x += v.x; y += v.y; return *this; }
    vec2& operator-=(const vec2& v) { x -= v.x; y -= v.y; return *this; }
    vec2& operator*=(const vec2& v) { x *= v.x; y *= v.y; return *this; }
    vec2& operator/=(const vec2& v) { x /= v.x; y /= v.y; return *this; }

    friend vec2 operator+(float s, const vec2& v) { return vec2(s + v.x, s + v.y); }
    friend vec2 operator-(float s, const vec2& v) { return vec2(s - v.x, s - v.y); }
    friend vec2 operator*(float s, const vec2& v) { return vec2(s * v.x, s * v.y); }
    friend vec2 operator/(float s, const vec2& v) { return vec2(s / v.x, s / v.y); }

    // Indexing
    float& operator[](int i) { return (i == 0) ? x : y; }
    float operator[](int i) const { return (i == 0) ? x : y; }

    // Math functions
    friend vec2 floor(const vec2& v) {
        return vec2(std::floor(v.x), std::floor(v.y));
    }
    friend vec2 fract(const vec2& v) {
        return vec2(fract(v.x), fract(v.y));
    }
    // Sine and Cosine
    friend vec2 sin(const vec2& v) {
        return vec2(std::sin(v.x), std::sin(v.y));
    }
    friend vec2 cos(const vec2& v) {
        return vec2(std::cos(v.x), std::cos(v.y));
    }
    friend vec2 mod(const vec2& v, float s) {
        return vec2(mod(v.x, s), mod(v.y, s));
    }
    friend vec2 mix(const vec2& a, const vec2& b, float t) {
        return a * (1.0f - t) + b * t;
    }
    friend float dot(const vec2& a, const vec2& b) {
        return a.x * b.x + a.y * b.y;
    }
    friend float length(const vec2& v) {
        return std::sqrt(dot(v, v));
    }
    friend vec2 normalize(const vec2& v) {
        float len = length(v);
        return len > 0.0f ? v / len : vec2(0.0f, 0.0f);
    }
};

// vec2_proxy implementation
float length(const vec2_proxy& v) { return length(vec2(v)); }
float dot(const vec2_proxy& a, const vec2_proxy& b) { return dot(vec2(a), vec2(b)); }
vec2 operator+(const vec2& a, const vec2_proxy& b) { return a + vec2(b); }
vec2 operator/(const vec2& a, const vec2_proxy& b) { return a / vec2(b); }
vec2 operator*(const vec2& a, const vec2_proxy& b) { return a * vec2(b); }
vec2 operator-(const vec2& a, const vec2_proxy& b) { return a - vec2(b); }
vec2 operator+(const vec2_proxy& a, const vec2& b) { return vec2(a) + b; }
vec2 operator/(const vec2_proxy& a, const vec2& b) { return vec2(a) / b; }
vec2 operator*(const vec2_proxy& a, const vec2& b) { return vec2(a) * b; }
vec2 operator-(const vec2_proxy& a, const vec2& b) { return vec2(a) - b; }
vec2 operator+(const vec2_proxy& a, const vec2_proxy& b) { return vec2(a) + vec2(b); }
vec2 operator/(const vec2_proxy& a, const vec2_proxy& b) { return vec2(a) / vec2(b); }
vec2 operator*(const vec2_proxy& a, const vec2_proxy& b) { return vec2(a) * vec2(b); }
vec2 operator-(const vec2_proxy& a, const vec2_proxy& b) { return vec2(a) - vec2(b); }
vec2 operator+(const vec2_proxy& a, float s) { return vec2(a) + s; }
vec2 operator-(const vec2_proxy& a, float s) { return vec2(a) - s; }
vec2 operator*(const vec2_proxy& a, float s) { return vec2(a) * s; }
vec2 operator/(const vec2_proxy& a, float s) { return vec2(a) / s; }
vec2 operator+(float s, const vec2_proxy& v) { return s + vec2(v); }
vec2 operator-(float s, const vec2_proxy& v) { return s - vec2(v); }
vec2 operator*(float s, const vec2_proxy& v) { return s * vec2(v); }
vec2 operator/(float s, const vec2_proxy& v) { return s / vec2(v); }
// += etc.
inline vec2_proxy operator+=(vec2_proxy& a, const vec2_proxy& b) { return a = vec2(a) + vec2(b); }
inline vec2_proxy operator+=(vec2_proxy& a, const vec2& b) { return a = vec2(a) + vec2(b); }



// vec3
class vec3 {
public:
    float x, y, z;


    vec2_proxy xy,yz,xz,zx; // This will be assigned properly in constructor

    vec3() : x(0), y(0), z(0), xy(x,y), yz(y, z), xz(x,z),zx(z,x) {}
    vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_), xy(x,y), yz(y, z), xz(x,z),zx(z,x) {}
    vec3(float s_) : x(s_), y(s_), z(s_), xy(x,y), yz(y, z), xz(x,z),zx(z,x) {}
    vec3(vec2 xy_, float z_) : x(xy_.x), y(xy_.y), z(z_), xy(x,y), yz(y, z), xz(x,z),zx(z,x) {}


    // Arithmetic
    vec3 operator+(const vec3& v) const { return vec3(x + v.x, y + v.y, z + v.z); }
    vec3 operator-(const vec3& v) const { return vec3(x - v.x, y - v.y, z - v.z); }
    vec3 operator*(const vec3& v) const { return vec3(x * v.x, y * v.y, z * v.z); }
    vec3 operator/(const vec3& v) const { return vec3(x / v.x, y / v.y, z / v.z); }
    vec3 operator*(float s) const { return vec3(x * s, y * s, z * s); }
    vec3 operator/(float s) const { return vec3(x / s, y / s, z / s); }
    vec3 operator+(float s) const { return vec3(x + s, y + s, z + s); }
    vec3 operator-(float s) const { return vec3(x - s, y - s, z - s); }
    // +=
    vec3& operator+=(const vec3& v) { x += v.x; y += v.y; z += v.z; return *this; }
    vec3& operator-=(const vec3& v) { x -= v.x; y -= v.y; z -= v.z; return *this; }
    vec3& operator*=(const vec3& v) { x *= v.x; y *= v.y; z *= v.z; return *this; }

    friend vec3 operator+(float s, const vec3& v) { return vec3(s + v.x, s + v.y, s + v.z); }
    friend vec3 operator-(float s, const vec3& v) { return vec3(s - v.x, s - v.y, s - v.z); }
    friend vec3 operator*(float s, const vec3& v) { return vec3(s * v.x, s * v.y, s * v.z); }
    friend vec3 operator/(float s, const vec3& v) { return vec3(s / v.x, s / v.y, s / v.z); }

    // Indexing
    float& operator[](int i) { return (i == 0) ? x : (i == 1) ? y : z; }
    float operator[](int i) const { return (i == 0) ? x : (i == 1) ? y : z; }

    // Math functions
    friend vec3 floor(const vec3& v) {
        return vec3(std::floor(v.x), std::floor(v.y), std::floor(v.z));
    }
    friend vec3 abs(const vec3& v) {
        return vec3(std::fabs(v.x), std::fabs(v.y), std::fabs(v.z));
    }
    friend vec3 fract(const vec3& v) {
        return vec3(fract(v.x), fract(v.y), fract(v.z));
    }
    // Sine and Cosine
    friend vec3 sin(const vec3& v) {
        return vec3(std::sin(v.x), std::sin(v.y), std::sin(v.z));
    }
    friend vec3 cos(const vec3& v) {
        return vec3(std::cos(v.x), std::cos(v.y), std::cos(v.z));
    }
    friend vec3 mod(const vec3& v, float s) {
        return vec3(mod(v.x, s), mod(v.y, s), mod(v.z, s));
    }
    friend vec3 mix(const vec3& a, const vec3& b, float t) {
        return a * (1.0f - t) + b * t;
    }
    friend float dot(const vec3& a, const vec3& b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }
    friend float length(const vec3& v) {
        return std::sqrt(dot(v, v));
    }
    friend vec3 normalize(const vec3& v) {
        float len = length(v);
        return len > 0.0f ? v / len : vec3(0.0f, 0.0f, 0.0f);
    }
    friend vec3 cross(const vec3& a, const vec3& b) {
        return vec3(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
        );
    }


};

inline vec3 operator-(const vec3& v) {
    return vec3(-v.x, -v.y, -v.z);
}

template <typename Proxy>
inline auto operator*(const Proxy& p, float s)
    -> std::enable_if_t<std::is_class_v<Proxy>, decltype(vec3(p) * s)>
{
    return vec3(p) * s;
}

// vec4
class vec4 {
public:
    float x, y, z, w;


    vec2_proxy xy,yz,xz,zx; // This will be assigned properly in constructor
    vec3_proxy xyz,yzw,yxz; // This will be assigned properly in constructor

    vec4() : vec4(0.0f, 0.0f, 0.0f, 0.0f) {}
    vec4(float s_) : vec4(s_, s_, s_, s_) {}
    vec4(vec3 xyz_, float w_): vec4(xyz_.x, xyz_.y, xyz_.z, w_) {}

    vec4(float x_, float y_, float z_, float w_)
        : x(x_), y(y_), z(z_), w(w_),
        xy(x, y), yz(y, z), xz(x, z), zx(z, x),
          xyz(x, y, z), yzw(y, z, w), yxz(y, z, x) {}

    // Arithmetic
    vec4 operator+(const vec4& v) const { return vec4(x + v.x, y + v.y, z + v.z, w + v.w); }
    vec4 operator-(const vec4& v) const { return vec4(x - v.x, y - v.y, z - v.z, w - v.w); }
    vec4 operator*(const vec4& v) const { return vec4(x * v.x, y * v.y, z * v.z, w * v.w); }
    vec4 operator/(const vec4& v) const { return vec4(x / v.x, y / v.y, z / v.z, w / v.w); }
    vec4 operator*(float s) const { return vec4(x * s, y * s, z * s, w * s); }
    vec4 operator/(float s) const { return vec4(x / s, y / s, z / s, w / s); }
    vec4 operator+(float s) const { return vec4(x + s, y + s, z + s, w + s); }
    vec4 operator-(float s) const { return vec4(x - s, y - s, z - s, w - s); }

    friend vec4 operator+(float s, const vec4& v) { return vec4(s + v.x, s + v.y, s + v.z, s + v.w); }
    friend vec4 operator-(float s, const vec4& v) { return vec4(s - v.x, s - v.y, s - v.z, s - v.w); }
    friend vec4 operator*(float s, const vec4& v) { return vec4(s * v.x, s * v.y, s * v.z, s * v.w); }
    friend vec4 operator/(float s, const vec4& v) { return vec4(s / v.x, s / v.y, s / v.z, s / v.w); }

    // Indexing
    float& operator[](int i) {
        if (i == 0) return x;
        else if (i == 1) return y;
        else if (i == 2) return z;
        else return w;
    }
    float operator[](int i) const {
        if (i == 0) return x;
        else if (i == 1) return y;
        else if (i == 2) return z;
        else return w;
    }

    // Math functions
    friend vec4 floor(const vec4& v) {
        return vec4(std::floor(v.x), std::floor(v.y), std::floor(v.z), std::floor(v.w));
    }
    friend vec4 fract(const vec4& v) {
        return vec4(fract(v.x), fract(v.y), fract(v.z), fract(v.w));
    }
    friend vec4 mod(const vec4& v, float s) {
        return vec4(mod(v.x, s), mod(v.y, s), mod(v.z, s), mod(v.w, s));
    }
    friend vec4 mix(const vec4& a, const vec4& b, float t) {
        return a * (1.0f - t) + b * t;
    }
    friend float dot(const vec4& a, const vec4& b) {
        return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    friend float length(const vec4& v) {
        return std::sqrt(dot(v, v));
    }
    friend vec4 normalize(const vec4& v) {
        float len = length(v);
        return len > 0.0f ? v / len : vec4(0.0f, 0.0f, 0.0f, 0.0f);
    }
};

// distance
inline float distance(const vec2& a, const vec2& b) {
    return length(a - b);
}

inline vec2 pow(const vec2& a, const vec2& b) {
    using std::pow;
    return vec2(pow(a.x, b.x), pow(a.y, b.y));
}

inline vec2 pow(const vec2& a, float b) {
    using std::pow;
    return vec2(pow(a.x, b), pow(a.y, b));
}

inline vec3 pow(const vec3& a, const vec3& b) {
    using std::pow;
    return vec3(pow(a.x, b.x), pow(a.y, b.y), pow(a.z, b.z));
}

inline vec3 pow(const vec3& a, float b) {
    using std::pow;
    return vec3(pow(a.x, b), pow(a.y, b), pow(a.z, b));
}

inline vec4 pow(const vec4& a, const vec4& b) {
    using std::pow;
    return vec4(pow(a.x, b.x), pow(a.y, b.y), pow(a.z, b.z), pow(a.w, b.w));
}

inline vec4 pow(const vec4& a, float b) {
    using std::pow;
    return vec4(pow(a.x, b), pow(a.y, b), pow(a.z, b), pow(a.w, b));
}

// reflect
inline vec2 reflect(const vec2& v, const vec2& n) {
    return v - 2.0f * dot(v, n) * n;
}
// reflect for vec3
inline vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2.0f * dot(v, n) * n;
}


class mat2 {
public:
    float m[2][2]; // column-major: m[column][row]

    // Identity matrix
    mat2() {
        m[0][0] = 1.0f; m[0][1] = 0.0f;
        m[1][0] = 0.0f; m[1][1] = 1.0f;
    }

    // Constructor with elements: column-major order
    // | a c |
    // | b d |
    mat2(float a, float b, float c, float d) {
        m[0][0] = a; m[0][1] = b;
        m[1][0] = c; m[1][1] = d;
    }

    // Matrix-vector multiplication (GLSL-style column-major)
    vec2 operator*(const vec2& v) const {
        return vec2(
            m[0][0] * v.x + m[1][0] * v.y,
            m[0][1] * v.x + m[1][1] * v.y
        );
    }

    // Matrix-matrix multiplication
    mat2 operator*(const mat2& n) const {
        return mat2(
            // Column 0
            m[0][0] * n.m[0][0] + m[1][0] * n.m[0][1],
            m[0][1] * n.m[0][0] + m[1][1] * n.m[0][1],
            // Column 1
            m[0][0] * n.m[1][0] + m[1][0] * n.m[1][1],
            m[0][1] * n.m[1][0] + m[1][1] * n.m[1][1]
        );
    }

    // Matrix-scalar multiplication
    mat2 operator*(float s) const {
        return mat2(
            m[0][0] * s, m[0][1] * s,
            m[1][0] * s, m[1][1] * s
        );
    }

    mat2 operator/(float s) const {
        return mat2(
            m[0][0] / s, m[0][1] / s,
            m[1][0] / s, m[1][1] / s
        );
    }

    mat2 operator+(float s) const {
        return mat2(
            m[0][0] + s, m[0][1] + s,
            m[1][0] + s, m[1][1] + s
        );
    }

    mat2 operator-(float s) const {
        return mat2(
            m[0][0] - s, m[0][1] - s,
            m[1][0] - s, m[1][1] - s
        );
    }

    // Matrix-matrix elementwise addition and subtraction
    mat2 operator+(const mat2& n) const {
        return mat2(
            m[0][0] + n.m[0][0], m[0][1] + n.m[0][1],
            m[1][0] + n.m[1][0], m[1][1] + n.m[1][1]
        );
    }

    mat2 operator-(const mat2& n) const {
        return mat2(
            m[0][0] - n.m[0][0], m[0][1] - n.m[0][1],
            m[1][0] - n.m[1][0], m[1][1] - n.m[1][1]
        );
    }

    // Scalar left-hand side operations
    friend mat2 operator*(float s, const mat2& m) {
        return m * s;
    }

    friend mat2 operator/(float s, const mat2& m) {
        return mat2(
            s / m.m[0][0], s / m.m[0][1],
            s / m.m[1][0], s / m.m[1][1]
        );
    }

    friend mat2 operator+(float s, const mat2& m) {
        return m + s;
    }

    friend mat2 operator-(float s, const mat2& m) {
        return mat2(
            s - m.m[0][0], s - m.m[0][1],
            s - m.m[1][0], s - m.m[1][1]
        );
    }

    // Math utilities
    static mat2 rotation(float angle) {
        float c = std::cos(angle);
        float s = std::sin(angle);
        return mat2(c, s, -s, c); // Column-major: [c -s; s c]
    }

    friend mat2 transpose(const mat2& m) {
        return mat2(
            m.m[0][0], m.m[1][0],
            m.m[0][1], m.m[1][1]
        );
    }

    friend float determinant(const mat2& m) {
        return m.m[0][0] * m.m[1][1] - m.m[1][0] * m.m[0][1];
    }

    friend mat2 inverse(const mat2& m) {
        float det = determinant(m);
        if (det == 0.0f) {
            return mat2(0.0f, 0.0f, 0.0f, 0.0f); // Non-invertible matrix
        }
        float invDet = 1.0f / det;
        return mat2(
             m.m[1][1] * invDet, -m.m[1][0] * invDet,
            -m.m[0][1] * invDet,  m.m[0][0] * invDet
        );
    }

    // Element access (column, row)
    float& operator()(int col, int row) {
        return m[col][row];
    }

    float operator()(int col, int row) const {
        return m[col][row];
    }
};



class sampler2D {};
// no-op texture function
vec4 texture(sampler2D sa, vec2 uv) {
    return vec4(0.0f, 0.0f, 0.0f, 1.0f); // Placeholder for texture sampling
}
vec4 texture(sampler2D sa, vec2 uv, float lod) {
    return vec4(0.0f, 0.0f, 0.0f, 1.0f); // Placeholder for texture sampling
}
// no-op textureLod
template <typename T>
T textureLod(sampler2D sa, T uv, float lod) {
  return T(0.0f); // Placeholder for texture sampling with LOD
}



#pragma once
#include <cmath>

// Scalar min
inline float min(float a, float b) {
    return std::fmin(a, b);
}

// Vector min (componentwise)
inline vec2 min(const vec2& a, const vec2& b) {
    return vec2(min(a.x, b.x), min(a.y, b.y));
}

inline vec3 min(const vec3& a, const vec3& b) {
    return vec3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}

inline vec4 min(const vec4& a, const vec4& b) {
    return vec4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w));
}

// Scalar max
inline float max(float a, float b) {
    return std::fmax(a, b);
}

// Vector max (componentwise)
inline vec2 max(const vec2& a, const vec2& b) {
    return vec2(max(a.x, b.x), max(a.y, b.y));
}

inline vec3 max(const vec3& a, const vec3& b) {
    return vec3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}

inline vec4 max(const vec4& a, const vec4& b) {
    return vec4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w));
}

inline vec2 clamp(const vec2& v, float minVal, float maxVal) {
    return vec2(clamp(v.x, minVal, maxVal), clamp(v.y, minVal, maxVal));
}

inline vec3 clamp(const vec3& v, float minVal, float maxVal) {
    return vec3(clamp(v.x, minVal, maxVal), clamp(v.y, minVal, maxVal), clamp(v.z, minVal, maxVal));
}

inline vec4 clamp(const vec4& v, float minVal, float maxVal) {
    return vec4(clamp(v.x, minVal, maxVal), clamp(v.y, minVal, maxVal), clamp(v.z, minVal, maxVal), clamp(v.w, minVal, maxVal));
}



class mat3 {
public:
    float m[3][3]; // column-major: m[column][row]

    // Identity matrix
    mat3() {
        m[0][0] = 1.0f; m[0][1] = 0.0f; m[0][2] = 0.0f;
        m[1][0] = 0.0f; m[1][1] = 1.0f; m[1][2] = 0.0f;
        m[2][0] = 0.0f; m[2][1] = 0.0f; m[2][2] = 1.0f;
    }

    // Constructor with elements: column-major order
    // | a d g |
    // | b e h |
    // | c f i |
    mat3(float a, float b, float c,
         float d, float e, float f,
         float g, float h, float i) {
        m[0][0] = a; m[0][1] = b; m[0][2] = c;
        m[1][0] = d; m[1][1] = e; m[1][2] = f;
        m[2][0] = g; m[2][1] = h; m[2][2] = i;
    }

    // Matrix-vector multiplication (GLSL-style column-major)
    vec3 operator*(const vec3& v) const {
        return vec3(
            m[0][0] * v.x + m[1][0] * v.y + m[2][0] * v.z,
            m[0][1] * v.x + m[1][1] * v.y + m[2][1] * v.z,
            m[0][2] * v.x + m[1][2] * v.y + m[2][2] * v.z
        );
    }

    // Matrix-matrix multiplication
    mat3 operator*(const mat3& n) const {
        return mat3(
            // Column 0
            m[0][0] * n.m[0][0] + m[1][0] * n.m[0][1] + m[2][0] * n.m[0][2],
            m[0][1] * n.m[0][0] + m[1][1] * n.m[0][1] + m[2][1] * n.m[0][2],
            m[0][2] * n.m[0][0] + m[1][2] * n.m[0][1] + m[2][2] * n.m[0][2],
            // Column 1
            m[0][0] * n.m[1][0] + m[1][0] * n.m[1][1] + m[2][0] * n.m[1][2],
            m[0][1] * n.m[1][0] + m[1][1] * n.m[1][1] + m[2][1] * n.m[1][2],
            m[0][2] * n.m[1][0] + m[1][2] * n.m[1][1] + m[2][2] * n.m[1][2],
            // Column 2
            m[0][0] * n.m[2][0] + m[1][0] * n.m[2][1] + m[2][0] * n.m[2][2],
            m[0][1] * n.m[2][0] + m[1][1] * n.m[2][1] + m[2][1] * n.m[2][2],
            m[0][2] * n.m[2][0] + m[1][2] * n.m[2][1] + m[2][2] * n.m[2][2]
        );
    }

    // Matrix-scalar multiplication
    mat3 operator*(float s) const {
        return mat3(
            m[0][0] * s, m[0][1] * s, m[0][2] * s,
            m[1][0] * s, m[1][1] * s, m[1][2] * s,
            m[2][0] * s, m[2][1] * s, m[2][2] * s
        );
    }

    mat3 operator/(float s) const {
        return mat3(
            m[0][0] / s, m[0][1] / s, m[0][2] / s,
            m[1][0] / s, m[1][1] / s, m[1][2] / s,
            m[2][0] / s, m[2][1] / s, m[2][2] / s
        );
    }

    mat3 operator+(float s) const {
        return mat3(
            m[0][0] + s, m[0][1] + s, m[0][2] + s,
            m[1][0] + s, m[1][1] + s, m[1][2] + s,
            m[2][0] + s, m[2][1] + s, m[2][2] + s
        );
    }

    mat3 operator-(float s) const {
        return mat3(
            m[0][0] - s, m[0][1] - s, m[0][2] - s,
            m[1][0] - s, m[1][1] - s, m[1][2] - s,
            m[2][0] - s, m[2][1] - s, m[2][2] - s
        );
    }

    // Matrix-matrix elementwise addition and subtraction
    mat3 operator+(const mat3& n) const {
        return mat3(
            m[0][0] + n.m[0][0], m[0][1] + n.m[0][1], m[0][2] + n.m[0][2],
            m[1][0] + n.m[1][0], m[1][1] + n.m[1][1], m[1][2] + n.m[1][2],
            m[2][0] + n.m[2][0], m[2][1] + n.m[2][1], m[2][2] + n.m[2][2]
        );
    }

    mat3 operator-(const mat3& n) const {
        return mat3(
            m[0][0] - n.m[0][0], m[0][1] - n.m[0][1], m[0][2] - n.m[0][2],
            m[1][0] - n.m[1][0], m[1][1] - n.m[1][1], m[1][2] - n.m[1][2],
            m[2][0] - n.m[2][0], m[2][1] - n.m[2][1], m[2][2] - n.m[2][2]
        );
    }

    // Scalar left-hand side operations
    friend mat3 operator*(float s, const mat3& m) {
        return m * s;
    }

    friend mat3 operator/(float s, const mat3& m) {
        return mat3(
            s / m.m[0][0], s / m.m[0][1], s / m.m[0][2],
            s / m.m[1][0], s / m.m[1][1], s / m.m[1][2],
            s / m.m[2][0], s / m.m[2][1], s / m.m[2][2]
        );
    }

    friend mat3 operator+(float s, const mat3& m) {
        return m + s;
    }

    friend mat3 operator-(float s, const mat3& m) {
        return mat3(
            s - m.m[0][0], s - m.m[0][1], s - m.m[0][2],
            s - m.m[1][0], s - m.m[1][1], s - m.m[1][2],
            s - m.m[2][0], s - m.m[2][1], s - m.m[2][2]
        );
    }

    // Math utilities
    friend mat3 transpose(const mat3& m) {
        return mat3(
            m.m[0][0], m.m[1][0], m.m[2][0],
            m.m[0][1], m.m[1][1], m.m[2][1],
            m.m[0][2], m.m[1][2], m.m[2][2]
        );
    }

    friend float determinant(const mat3& m) {
        return 
              m.m[0][0] * (m.m[1][1] * m.m[2][2] - m.m[2][1] * m.m[1][2])
            - m.m[1][0] * (m.m[0][1] * m.m[2][2] - m.m[2][1] * m.m[0][2])
            + m.m[2][0] * (m.m[0][1] * m.m[1][2] - m.m[1][1] * m.m[0][2]);
    }

    friend mat3 inverse(const mat3& m) {
        float det = determinant(m);
        if (det == 0.0f) {
            return mat3(0.0f, 0.0f, 0.0f, 
                        0.0f, 0.0f, 0.0f, 
                        0.0f, 0.0f, 0.0f); // Non-invertible
        }
        float invDet = 1.0f / det;
        return mat3(
             (m.m[1][1] * m.m[2][2] - m.m[2][1] * m.m[1][2]) * invDet,
            -(m.m[0][1] * m.m[2][2] - m.m[2][1] * m.m[0][2]) * invDet,
             (m.m[0][1] * m.m[1][2] - m.m[1][1] * m.m[0][2]) * invDet,

            -(m.m[1][0] * m.m[2][2] + -m.m[2][0] * m.m[1][2]) * invDet,
             (m.m[0][0] * m.m[2][2] + -m.m[2][0] * m.m[0][2]) * invDet,
            -(m.m[0][0] * m.m[1][2] + -m.m[1][0] * m.m[0][2]) * invDet,

             (m.m[1][0] * m.m[2][1] - m.m[2][0] * m.m[1][1]) * invDet,
            -(m.m[0][0] * m.m[2][1] - m.m[2][0] * m.m[0][1]) * invDet,
             (m.m[0][0] * m.m[1][1] - m.m[1][0] * m.m[0][1]) * invDet
        );
    }

    // Element access (column, row)
    float& operator()(int col, int row) {
        return m[col][row];
    }

    float operator()(int col, int row) const {
        return m[col][row];
    }
};


// Definitions
inline vec2_proxy::operator vec2() const {
    return vec2(*a, *b);
}
inline vec2_proxy& vec2_proxy::operator=(const vec2& v) {
    *a = v.x;
    *b = v.y;
    return *this;
}
// Same for vec3_proxy
inline vec3_proxy::operator vec3() const {
    return vec3(*a, *b, *c);
}
inline vec3_proxy& vec3_proxy::operator=(const vec3& v) {
    *a = v.x;
    *b = v.y;
    *c = v.z;
    return *this;
}
