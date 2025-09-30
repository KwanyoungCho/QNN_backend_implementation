#include "io_meta.h"

#include <fstream>
#include <sstream>
#include <string>

namespace llm_test {

static bool find_next(const std::string& s, size_t& pos, const std::string& key) {
  size_t p = s.find(key, pos);
  if (p == std::string::npos) return false;
  pos = p + key.size();
  return true;
}

static bool parse_string_value(const std::string& s, size_t& pos, std::string& out) {
  size_t q1 = s.find('"', pos);
  if (q1 == std::string::npos) return false;
  size_t q2 = s.find('"', q1 + 1);
  if (q2 == std::string::npos) return false;
  out = s.substr(q1 + 1, q2 - (q1 + 1));
  pos = q2 + 1;
  return true;
}

static bool parse_uint64_value(const std::string& s, size_t& pos, uint64_t& out) {
  size_t p = s.find_first_of("0123456789", pos);
  if (p == std::string::npos) return false;
  size_t e = p;
  while (e < s.size() && (s[e] >= '0' && s[e] <= '9')) e++;
  out = std::stoull(s.substr(p, e - p));
  pos = e;
  return true;
}

static bool parse_int_value(const std::string& s, size_t& pos, int& out) {
  bool neg = false;
  size_t p = s.find_first_of("-0123456789", pos);
  if (p == std::string::npos) return false;
  if (s[p] == '-') { neg = true; p++; }
  size_t e = p;
  while (e < s.size() && (s[e] >= '0' && s[e] <= '9')) e++;
  int v = std::stoi(s.substr(p, e - p));
  out = neg ? -v : v;
  pos = e;
  return true;
}

static bool find_array_range(const std::string& s, const std::string& key, size_t& arr_begin, size_t& arr_end) {
  size_t pos = 0;
  if (!find_next(s, pos, key)) return false;
  size_t lb = s.find('[', pos);
  if (lb == std::string::npos) return false;
  int depth = 1;
  size_t i = lb + 1;
  while (i < s.size() && depth > 0) {
    if (s[i] == '[') depth++;
    else if (s[i] == ']') depth--;
    i++;
  }
  if (depth != 0) return false;
  arr_begin = lb + 1;
  arr_end = i - 1;
  return true;
}

static void extract_objects(const std::string& s, size_t begin, size_t end, std::vector<std::string>& out_objs) {
  size_t i = begin;
  while (i < end) {
    size_t obj_start = s.find('{', i);
    if (obj_start == std::string::npos || obj_start >= end) break;
    int depth = 1;
    size_t j = obj_start + 1;
    while (j < end && depth > 0) {
      if (s[j] == '{') depth++;
      else if (s[j] == '}') depth--;
      j++;
    }
    if (depth == 0) {
      out_objs.emplace_back(s.substr(obj_start, j - obj_start));
      i = j;
    } else {
      break;
    }
  }
}

static bool parse_dims_and_bpe(const std::string& obj, uint64_t& numel, uint64_t& bpe) {
  // dims
  size_t pos = 0; numel = 1; bpe = 0;
  if (find_next(obj, pos, "\"dims\"")) {
    size_t lb = obj.find('[', pos);
    size_t rb = obj.find(']', pos);
    if (lb != std::string::npos && rb != std::string::npos && rb > lb) {
      size_t p = lb + 1;
      while (p < rb) {
        uint64_t v = 0;
        if (!parse_uint64_value(obj, p, v)) break;
        numel *= v;
        size_t comma = obj.find(',', p);
        if (comma == std::string::npos || comma > rb) break;
        p = comma + 1;
      }
    }
  }
  pos = 0;
  if (find_next(obj, pos, "\"bytesPerElement\"")) {
    if (!find_next(obj, pos, ":")) return true;
    uint64_t v = 0; if (parse_uint64_value(obj, pos, v)) bpe = v;
  }
  return true;
}

bool parse_graph_io_json(const std::string& json_path, GraphIOMeta& out) {
  std::ifstream ifs(json_path);
  if (!ifs.is_open()) return false;
  std::stringstream buffer; buffer << ifs.rdbuf();
  std::string s = buffer.str();

  size_t pos = 0;
  if (find_next(s, pos, "\"graph\"")) {
    if (!find_next(s, pos, ":")) return false;
    if (!parse_string_value(s, pos, out.graph_name)) return false;
  }

  size_t in_begin = 0, in_end = 0, out_begin = 0, out_end = 0;
  if (!find_array_range(s, "\"inputs\"", in_begin, in_end)) return false;
  if (!find_array_range(s, "\"outputs\"", out_begin, out_end)) return false;

  std::vector<std::string> in_objs, out_objs;
  extract_objects(s, in_begin, in_end, in_objs);
  extract_objects(s, out_begin, out_end, out_objs);

  auto parse_tensor = [](const std::string& obj, TensorDesc& td) {
    size_t q = 0; td.name.clear(); td.nbytes = 0; td.mutable_buffer_id = -1;
    if (find_next(obj, q, "\"name\"")) {
      if (find_next(obj, q, ":")) parse_string_value(obj, q, td.name);
    }
    q = 0;
    if (find_next(obj, q, "\"nbytes\"")) {
      if (find_next(obj, q, ":")) parse_uint64_value(obj, q, td.nbytes);
    } else {
      // fallback compute from dims * bytesPerElement if available
      uint64_t numel = 1, bpe = 0;
      parse_dims_and_bpe(obj, numel, bpe);
      if (bpe > 0) td.nbytes = numel * bpe;
    }
    q = 0;
    if (find_next(obj, q, "\"mutableBufferId\"")) {
      if (find_next(obj, q, ":")) parse_int_value(obj, q, td.mutable_buffer_id);
    }
  };

  for (const auto& o : in_objs) {
    TensorDesc td; parse_tensor(o, td); if (!td.name.empty()) out.inputs.push_back(td);
  }
  for (const auto& o : out_objs) {
    TensorDesc td; parse_tensor(o, td); if (!td.name.empty()) out.outputs.push_back(td);
  }
  return !out.inputs.empty() || !out.outputs.empty();
}

}


