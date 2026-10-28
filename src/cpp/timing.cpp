// Compiled CPU timing of every paper pipeline: ONNX Runtime C++ API for the graphs and C++ ports of the correction
// (correction.OnlineRule) and the detector features (detector.OnlineDetector). The export of
// src/idsvae/timing/export.py gives the graphs, parameters, inputs, Python labels and the protocol settings (manifest.json).
//
// Usage: timing <export dir> check <report.json>
//        timing <export dir> time <output.json>
//        timing <export dir> stream <stream prefix> <pipeline name>
// "check" runs each pipeline once over the timing sample from the warmed state and compares every final label with the
// Python label; "time" runs the sessions and compares the labels of every timed pass, outside the clock; "stream" runs the
// correction and detector of one pipeline on given model outputs (tests/test_compiled_stream.py).
#include <onnxruntime_cxx_api.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using json = nlohmann::ordered_json;
using Clock = std::chrono::steady_clock;

namespace {

constexpr int64_t kSecondsPerMinute = 60;
constexpr int64_t kSecondsPerDay = 86400;
constexpr double kDecay = 0.5;

// CPython evaluates float ** number with the C library pow; a direct std::pow(x, 2.0) can become x * x at compile time.
double (*volatile library_pow)(double, double) = std::pow;

double elapsed_seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }

std::string read_text(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot open " + path);
  return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

template <typename T>
std::vector<T> read_array(const std::string& path) {
  std::string bytes = read_text(path);
  if (bytes.size() % sizeof(T) != 0) throw std::runtime_error(path + ": size is not a multiple of the element size");
  std::vector<T> values(bytes.size() / sizeof(T));
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

struct Matrix {
  std::vector<double> values;
  size_t rows = 0;
  size_t columns = 0;
  const double* row(size_t index) const { return values.data() + index * columns; }
};

Matrix read_matrix(const std::string& path, size_t columns) {
  Matrix matrix{read_array<double>(path), 0, columns};
  if (matrix.values.size() % columns != 0) throw std::runtime_error(path + ": not a whole number of rows");
  matrix.rows = matrix.values.size() / columns;
  return matrix;
}

// First index of the largest value, as numpy argmax.
template <typename T>
int32_t first_argmax(const T* values, size_t count) {
  size_t best = 0;
  for (size_t index = 1; index < count; ++index) {
    if (values[index] > values[best]) best = index;
  }
  return static_cast<int32_t>(best);
}

// ------------------------------------------------------------------------------------------------ ONNX Runtime graphs

Ort::SessionOptions session_options() {
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(1);
  options.SetInterOpNumThreads(1);
  options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  // onnx_graph.session: the memory arena is off for every graph, in Python and here.
  options.DisableCpuMemArena();
  return options;
}

// One graph with batch-1 input and output buffers that the session reads and writes in place.
class Graph {
 public:
  // input_shapes: shape of each input per call; output_widths: columns of each output, in the order of output_names.
  Graph(Ort::Env& env, const std::string& path, const std::vector<std::string>& output_names, const std::vector<size_t>& output_widths,
        const std::vector<std::vector<int64_t>>& input_shapes)
      : session_(env, path.c_str(), session_options()), memory_(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault)) {
    Ort::AllocatorWithDefaultOptions allocator;
    if (session_.GetInputCount() != input_shapes.size()) throw std::runtime_error(path + ": unexpected input count");
    for (size_t index = 0; index < input_shapes.size(); ++index) {
      input_names_.push_back(session_.GetInputNameAllocated(index, allocator).get());
      auto type = session_.GetInputTypeInfo(index).GetTensorTypeAndShapeInfo().GetElementType();
      input_types_.push_back(type);
      size_t count = 1;
      for (int64_t dimension : input_shapes[index]) count *= static_cast<size_t>(dimension);
      if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
        double_inputs_.emplace_back(count);
        float_inputs_.emplace_back();
        long_inputs_.emplace_back();
        inputs_.push_back(Ort::Value::CreateTensor<double>(memory_, double_inputs_.back().data(), count, input_shapes[index].data(), input_shapes[index].size()));
      } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        double_inputs_.emplace_back();
        float_inputs_.emplace_back(count);
        long_inputs_.emplace_back();
        inputs_.push_back(Ort::Value::CreateTensor<float>(memory_, float_inputs_.back().data(), count, input_shapes[index].data(), input_shapes[index].size()));
      } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        double_inputs_.emplace_back();
        float_inputs_.emplace_back();
        long_inputs_.emplace_back(count);
        inputs_.push_back(Ort::Value::CreateTensor<int64_t>(memory_, long_inputs_.back().data(), count, input_shapes[index].data(), input_shapes[index].size()));
      } else {
        throw std::runtime_error(path + ": unsupported input type");
      }
    }
    std::vector<std::string> session_outputs;
    for (size_t index = 0; index < session_.GetOutputCount(); ++index) session_outputs.push_back(session_.GetOutputNameAllocated(index, allocator).get());
    for (size_t index = 0; index < output_names.size(); ++index) {
      auto found = std::find(session_outputs.begin(), session_outputs.end(), output_names[index]);
      if (found == session_outputs.end()) throw std::runtime_error(path + ": no output " + output_names[index]);
      auto type = session_.GetOutputTypeInfo(static_cast<size_t>(found - session_outputs.begin())).GetTensorTypeAndShapeInfo().GetElementType();
      output_names_.push_back(output_names[index]);
      // Every timed call classifies one flow, so every output has one row.
      output_shapes_.push_back({1, static_cast<int64_t>(output_widths[index])});
      size_t count = output_widths[index];
      if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        float_outputs_.emplace_back(count);
        double_outputs_.emplace_back();
        outputs_.push_back(Ort::Value::CreateTensor<float>(memory_, float_outputs_.back().data(), count, output_shapes_.back().data(), output_shapes_.back().size()));
      } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
        float_outputs_.emplace_back();
        double_outputs_.emplace_back(count);
        outputs_.push_back(Ort::Value::CreateTensor<double>(memory_, double_outputs_.back().data(), count, output_shapes_.back().data(), output_shapes_.back().size()));
      } else {
        throw std::runtime_error(path + ": unsupported output type of " + output_names[index]);
      }
      output_types_.push_back(type);
    }
    for (auto& name : input_names_) input_name_pointers_.push_back(name.c_str());
    for (auto& name : output_names_) output_name_pointers_.push_back(name.c_str());
  }

  size_t input_index(const std::string& name) const {
    auto found = std::find(input_names_.begin(), input_names_.end(), name);
    if (found == input_names_.end()) throw std::runtime_error("no graph input " + name);
    return static_cast<size_t>(found - input_names_.begin());
  }

  // Writes values into input `index`, converted to its element type (onnx_graph.graph_function casts float inputs the same way).
  template <typename T>
  void set_input(size_t index, const T* values, size_t count, size_t offset = 0) {
    switch (input_types_[index]) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        for (size_t position = 0; position < count; ++position) double_inputs_[index][offset + position] = static_cast<double>(values[position]);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        for (size_t position = 0; position < count; ++position) float_inputs_[index][offset + position] = static_cast<float>(values[position]);
        break;
      default:
        for (size_t position = 0; position < count; ++position) long_inputs_[index][offset + position] = static_cast<int64_t>(values[position]);
    }
  }
  double* double_input(size_t index) {
    if (input_types_[index] != ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) throw std::runtime_error("input " + input_names_[index] + " is not float64");
    return double_inputs_[index].data();
  }

  void run() {
    session_.Run(run_options_, input_name_pointers_.data(), inputs_.data(), inputs_.size(), output_name_pointers_.data(), outputs_.data(), outputs_.size());
  }
  const float* output(size_t index) const {
    if (output_types_[index] != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) throw std::runtime_error("output " + output_names_[index] + " is not float32");
    return float_outputs_[index].data();
  }
  // numpy argmax of the one output row, in the element type of the output.
  int32_t output_argmax(size_t index, size_t count) const {
    if (output_types_[index] == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) return first_argmax(double_outputs_[index].data(), count);
    return first_argmax(float_outputs_[index].data(), count);
  }

  // Untimed batch run for the warm state: ONNX Runtime allocates the outputs. rows: (count, columns) of the first input.
  std::vector<std::vector<float>> run_batch(const std::vector<double>& rows, size_t count, size_t columns) {
    if (input_names_.size() != 1) throw std::runtime_error("batch runs take one input");
    std::vector<int64_t> shape{static_cast<int64_t>(count), static_cast<int64_t>(columns)};
    std::vector<float> float_rows;
    Ort::Value input{nullptr};
    std::vector<double> double_rows;
    if (input_types_[0] == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
      double_rows = rows;
      input = Ort::Value::CreateTensor<double>(memory_, double_rows.data(), double_rows.size(), shape.data(), shape.size());
    } else {
      float_rows.assign(rows.begin(), rows.end());
      input = Ort::Value::CreateTensor<float>(memory_, float_rows.data(), float_rows.size(), shape.data(), shape.size());
    }
    auto results = session_.Run(run_options_, input_name_pointers_.data(), &input, 1, output_name_pointers_.data(), output_name_pointers_.size());
    std::vector<std::vector<float>> outputs;
    for (auto& result : results) {
      if (result.GetTensorTypeAndShapeInfo().GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) throw std::runtime_error("batch outputs must be float32");
      size_t size = result.GetTensorTypeAndShapeInfo().GetElementCount();
      const float* data = result.GetTensorData<float>();
      outputs.emplace_back(data, data + size);
    }
    return outputs;
  }

 private:
  Ort::Session session_;
  Ort::MemoryInfo memory_;
  Ort::RunOptions run_options_;
  std::vector<std::string> input_names_, output_names_;
  std::vector<const char*> input_name_pointers_, output_name_pointers_;
  std::vector<ONNXTensorElementDataType> input_types_;
  std::vector<std::vector<int64_t>> output_shapes_;
  std::vector<std::vector<double>> double_inputs_;
  std::vector<std::vector<float>> float_inputs_;
  std::vector<std::vector<int64_t>> long_inputs_;
  std::vector<ONNXTensorElementDataType> output_types_;
  std::vector<std::vector<float>> float_outputs_;
  std::vector<std::vector<double>> double_outputs_;
  std::vector<Ort::Value> inputs_, outputs_;
};

// ------------------------------------------------------------------------------------------------ stream state ports

// correction.cell_key: floor(latent / divisor) per axis as int64 bytes; the float32 latent value is promoted to float64.
std::string cell_key(const float* latent, const std::vector<double>& divisor) {
  std::string key(divisor.size() * sizeof(int64_t), '\0');
  for (size_t axis = 0; axis < divisor.size(); ++axis) {
    int64_t cell = static_cast<int64_t>(std::floor(static_cast<double>(latent[axis]) / divisor[axis]));
    std::memcpy(key.data() + axis * sizeof(int64_t), &cell, sizeof(int64_t));
  }
  return key;
}

// correction.MidasCounter.
class MidasCounter {
 public:
  void start(int64_t minute) {
    started_ = true;
    first_minute_ = minute;
  }

  double score(const std::string& key, int64_t minute) {
    if (!started_) start(minute);
    int64_t now = minute - first_minute_ + 1;
    auto [entry, inserted] = state_.try_emplace(key, Entry{0, 0.0, now});
    Entry& value = entry->second;
    value.decayed = value.decayed * library_pow(kDecay, static_cast<double>(now - value.last)) + 1.0;
    value.total += 1;
    value.last = now;
    if (now == 1) return 0.0;
    double difference = value.decayed - static_cast<double>(value.total) / static_cast<double>(now);
    double squared = library_pow(std::fabs(difference), 2.0);
    return squared * static_cast<double>(now * now) / static_cast<double>(value.total * (now - 1));
  }

 private:
  struct Entry {
    int64_t total;
    double decayed;
    int64_t last;
  };
  bool started_ = false;
  int64_t first_minute_ = 0;
  std::unordered_map<std::string, Entry> state_;
};

// correction.AttackContext; class labels are indices into the Stage 3 classes.
class AttackContext {
 public:
  AttackContext(std::vector<int32_t> attack_order, size_t class_count, int64_t window_seconds)
      : attack_order_(std::move(attack_order)), is_attack_(class_count, false), counts_(class_count, 0), window_seconds_(window_seconds) {
    for (int32_t attack : attack_order_) is_attack_[attack] = true;
  }

  // (most frequent attack class or -1, its count) among the flows before this second.
  std::pair<int32_t, int64_t> top(int64_t second) {
    if (!has_second_ || second != current_second_) {
      for (int32_t label : current_) {
        window_.emplace_back(current_second_, label);
        counts_[label] += 1;
      }
      current_.clear();
      current_second_ = second;
      has_second_ = true;
    }
    while (!window_.empty() && window_.front().first < second - window_seconds_) {
      counts_[window_.front().second] -= 1;
      window_.pop_front();
    }
    int32_t best = -1;
    int64_t best_count = 0;
    for (int32_t attack : attack_order_) {
      if (counts_[attack] > best_count) {
        best = attack;
        best_count = counts_[attack];
      }
    }
    return {best, best_count};
  }

  void add(int32_t predicted) {
    if (is_attack_[predicted]) current_.push_back(predicted);
  }

 private:
  std::vector<int32_t> attack_order_;
  std::vector<bool> is_attack_;
  std::vector<int64_t> counts_;
  int64_t window_seconds_;
  std::deque<std::pair<int64_t, int32_t>> window_;
  bool has_second_ = false;
  int64_t current_second_ = 0;
  std::vector<int32_t> current_;
};

struct RuleParams {
  size_t latent_dim;
  std::vector<double> divisor;
  double novelty;
  double burst;
  int64_t min_support;
  int64_t window_seconds;
  std::vector<int32_t> attack_order;
};

// correction.OnlineRule.
class OnlineRule {
 public:
  OnlineRule(const RuleParams& params, size_t class_count) : params_(&params), context_(params.attack_order, class_count, params.window_seconds) {}

  int32_t step(int64_t second, const float* latent, double recon_nll, int32_t predicted) {
    int64_t day = second / kSecondsPerDay;
    if (!has_day_ || day != day_) {
      day_ = day;
      has_day_ = true;
      midas_ = MidasCounter();
    }
    double burst_score = midas_.score(cell_key(latent, params_->divisor), second / kSecondsPerMinute);
    auto [attack, count] = context_.top(second);
    context_.add(predicted);
    bool is_novel = recon_nll > params_->novelty;
    bool is_burst = burst_score > params_->burst;
    if (is_novel && is_burst && count >= params_->min_support) return attack;
    return predicted;
  }

 private:
  const RuleParams* params_;
  AttackContext context_;
  bool has_day_ = false;
  int64_t day_ = 0;
  MidasCounter midas_;
};

// detector.CellWindow.
class CellWindow {
 public:
  double earlier_novel_share(int64_t second, int64_t window_seconds) {
    while (!entries_.empty() && entries_.front().first < second - window_seconds) {
      novel_ -= entries_.front().second;
      count_ -= 1;
      entries_.pop_front();
    }
    int64_t count = count_;
    int64_t novel = novel_;
    if (has_last_ && last_second_ == second) {
      count -= last_count_;
      novel -= last_novel_;
    }
    return count > 0 ? static_cast<double>(novel) / static_cast<double>(count) : 0.0;
  }

  void add(int64_t second, bool is_novel) {
    if (!has_last_ || last_second_ != second) {
      last_second_ = second;
      has_last_ = true;
      last_count_ = 0;
      last_novel_ = 0;
    }
    entries_.emplace_back(second, is_novel ? 1 : 0);
    count_ += 1;
    novel_ += is_novel ? 1 : 0;
    last_count_ += 1;
    last_novel_ += is_novel ? 1 : 0;
  }

 private:
  std::deque<std::pair<int64_t, int64_t>> entries_;
  int64_t count_ = 0, novel_ = 0;
  bool has_last_ = false;
  int64_t last_second_ = 0, last_count_ = 0, last_novel_ = 0;
};

struct StreamValues {
  double burst, novel_burst, share;
};

// detector.CellStreams.
class CellStreams {
 public:
  explicit CellStreams(int64_t window_seconds) : window_seconds_(window_seconds) {}

  void new_day(int64_t minute) {
    all_flows_ = MidasCounter();
    novel_flows_ = MidasCounter();
    windows_.clear();
    all_flows_.start(minute);
    novel_flows_.start(minute);
  }

  StreamValues step(const std::string& key, int64_t second, int64_t minute, bool is_novel) {
    double burst = all_flows_.score(key, minute);
    double novel_burst = is_novel ? novel_flows_.score(key, minute) : 0.0;
    CellWindow& window = windows_[key];
    double share = window.earlier_novel_share(second, window_seconds_);
    window.add(second, is_novel);
    return {burst, novel_burst, share};
  }

 private:
  int64_t window_seconds_;
  MidasCounter all_flows_, novel_flows_;
  std::unordered_map<std::string, CellWindow> windows_;
};

struct BackgroundParams {
  double burst_cutoff, fallback_median;
  size_t reservoir, warm_up, refresh;
};

// detector.DayBackground. The reservoir never fills on the timed day (1,976 of 20,000 flows), so the
// random replacement draw of NumPy is not ported: a full reservoir stops the program instead of giving other labels.
class DayBackground {
 public:
  explicit DayBackground(const BackgroundParams& params) : params_(&params) { new_day(); }

  void new_day() {
    reservoir_.clear();
    seen_ = 0;
    added_since_refresh_ = 0;
    median_ = params_->fallback_median;
  }

  double relative(double nll) const { return nll - median_; }

  void add(double nll, double burst_score) {
    if (burst_score > params_->burst_cutoff) return;
    seen_ += 1;
    if (reservoir_.size() >= params_->reservoir) throw std::runtime_error("the background reservoir is full: the NumPy replacement draw is not ported");
    reservoir_.push_back(nll);
    added_since_refresh_ += 1;
    if (seen_ >= params_->warm_up && added_since_refresh_ >= params_->refresh) {
      median_ = median(reservoir_);
      added_since_refresh_ = 0;
    }
  }

 private:
  // np.median: the middle value, or the mean of the two middle values for an even count.
  static double median(std::vector<double> values) {
    size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    double upper = values[middle];
    if (values.size() % 2 == 1) return upper;
    double lower = *std::max_element(values.begin(), values.begin() + middle);
    return (lower + upper) / 2.0;
  }

  const BackgroundParams* params_;
  std::vector<double> reservoir_;
  size_t seen_ = 0, added_since_refresh_ = 0;
  double median_ = 0.0;
};

// scalar_quantile: searchsorted(side="right") / len.
double scalar_quantile(const std::vector<double>& sorted_reference, double value) {
  auto count = std::upper_bound(sorted_reference.begin(), sorted_reference.end(), value) - sorted_reference.begin();
  return static_cast<double>(count) / static_cast<double>(sorted_reference.size());
}

enum class Feature { kNll, kKlTail, kBurst, kNovelBurst, kNovelShare };

struct DetectorParams {
  size_t latent_dim;
  std::vector<Feature> columns;
  std::vector<double> divisor;
  int64_t window_seconds;
  double novelty;
  std::vector<double> nll_reference, kl_reference, burst_reference, novel_burst_reference, share_reference;
  std::optional<BackgroundParams> background;
  // NumPy compares the float32 scores with the Python float cut-off in float32 (NEP 50).
  float cutoff;
};

// detector.OnlineDetector.features.
class OnlineDetector {
 public:
  explicit OnlineDetector(const DetectorParams& params) : params_(&params), streams_(params.window_seconds) {
    if (params.background) background_.emplace(*params.background);
  }

  void features(int64_t second, const float* latent, double nll, double kl, double* row) {
    int64_t minute = second / kSecondsPerMinute;
    int64_t day = second / kSecondsPerDay;
    if (!has_day_ || day != day_) {
      day_ = day;
      has_day_ = true;
      streams_.new_day(minute);
      if (background_) background_->new_day();
    }
    double novelty_value = background_ ? background_->relative(nll) : nll;
    StreamValues stream = streams_.step(cell_key(latent, params_->divisor), second, minute, novelty_value > params_->novelty);
    if (background_) background_->add(nll, stream.burst);
    double kl_quantile = scalar_quantile(params_->kl_reference, kl);
    for (size_t index = 0; index < params_->columns.size(); ++index) {
      switch (params_->columns[index]) {
        case Feature::kNll: row[index] = scalar_quantile(params_->nll_reference, novelty_value); break;
        case Feature::kKlTail: row[index] = 1.0 - 2.0 * std::min(kl_quantile, 1.0 - kl_quantile); break;
        case Feature::kBurst: row[index] = scalar_quantile(params_->burst_reference, stream.burst); break;
        case Feature::kNovelBurst: row[index] = scalar_quantile(params_->novel_burst_reference, stream.novel_burst); break;
        case Feature::kNovelShare: row[index] = scalar_quantile(params_->share_reference, stream.share); break;
      }
    }
  }

 private:
  const DetectorParams* params_;
  CellStreams streams_;
  std::optional<DayBackground> background_;
  bool has_day_ = false;
  int64_t day_ = 0;
};

// ------------------------------------------------------------------------------------------------ pipelines

struct Data {
  std::vector<int64_t> window_seconds, earlier_seconds;
  std::unordered_map<std::string, Matrix> window, earlier;
  std::string benign_label;
  // timing.WARM_BATCH: the states are warmed with model outputs of batches of this size, as in Python, so that the warmed
  // states are equal bit for bit.
  size_t warm_batch = 0;
};

class Pipeline {
 public:
  virtual ~Pipeline() = default;
  const std::vector<std::string>& parts() const { return parts_; }
  // Copies the warmed stream state into the state of the next pass; called outside the clock.
  virtual void start_pass() = 0;
  virtual void end_pass() {}
  // Final label index of one sample flow; adds the seconds of each part.
  virtual int32_t step(size_t flow, double* part_seconds) = 0;

 protected:
  std::vector<std::string> parts_;
};

class GraphPipeline : public Pipeline {
 public:
  GraphPipeline(Ort::Env& env, const std::string& path, const Matrix& window, size_t classes)
      : window_(window), classes_(classes), graph_(env, path, {"probabilities"}, {classes}, {{1, static_cast<int64_t>(window.columns)}}) {
    parts_ = {"model"};
  }
  void start_pass() override {}
  int32_t step(size_t flow, double* part_seconds) override {
    auto start = Clock::now();
    graph_.set_input(0, window_.row(flow), window_.columns);
    graph_.run();
    int32_t label = graph_.output_argmax(0, classes_);
    part_seconds[0] += elapsed_seconds(start);
    return label;
  }

 private:
  const Matrix& window_;
  size_t classes_;
  Graph graph_;
};

// baseline_graphs.OnnxFlowTransformer.stream at batch 1: the context is the last window_size - 1 flows and the flow; the first
// flow of a stream stands in for the flows before it; the positions are then always 0 .. window_size - 1.
class FlowTransformerPipeline : public Pipeline {
 public:
  FlowTransformerPipeline(Ort::Env& env, const std::string& path, const Matrix& window, size_t classes, size_t window_size)
      : window_(window), classes_(classes), window_size_(window_size),
        graph_(env, path, {"probabilities"}, {classes}, flow_shapes(env, path, window.columns, window_size)) {
    parts_ = {"model"};
    raw_index_ = graph_.input_index("raw");
    std::vector<int64_t> positions(window_size);
    for (size_t index = 0; index < window_size; ++index) positions[index] = static_cast<int64_t>(index);
    graph_.set_input(graph_.input_index("positions"), positions.data(), window_size);
  }
  void start_pass() override { first_ = true; }
  int32_t step(size_t flow, double* part_seconds) override {
    auto start = Clock::now();
    size_t columns = window_.columns;
    double* context = graph_.double_input(raw_index_);
    if (first_) {
      for (size_t row = 0; row + 1 < window_size_; ++row) std::memcpy(context + row * columns, window_.row(flow), columns * sizeof(double));
      first_ = false;
    } else {
      std::memmove(context, context + columns, (window_size_ - 1) * columns * sizeof(double));
    }
    std::memcpy(context + (window_size_ - 1) * columns, window_.row(flow), columns * sizeof(double));
    graph_.run();
    int32_t label = graph_.output_argmax(0, classes_);
    part_seconds[0] += elapsed_seconds(start);
    return label;
  }

 private:
  static std::vector<std::vector<int64_t>> flow_shapes(Ort::Env& env, const std::string& path, size_t columns, size_t window_size) {
    Ort::Session probe(env, path.c_str(), session_options());
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<std::vector<int64_t>> shapes;
    for (size_t index = 0; index < probe.GetInputCount(); ++index) {
      std::string name = probe.GetInputNameAllocated(index, allocator).get();
      if (name == "raw") shapes.push_back({static_cast<int64_t>(window_size), static_cast<int64_t>(columns)});
      else if (name == "positions") shapes.push_back({1, static_cast<int64_t>(window_size)});
      else throw std::runtime_error("unexpected FlowTransformer input " + name);
    }
    return shapes;
  }

  const Matrix& window_;
  size_t classes_, window_size_;
  Graph graph_;
  size_t raw_index_ = 0;
  bool first_ = true;
};

struct StreamState {
  OnlineRule rule;
  std::optional<OnlineDetector> detector;
};

int32_t index_of(const std::vector<std::string>& names, const std::string& name) {
  auto found = std::find(names.begin(), names.end(), name);
  if (found == names.end()) throw std::runtime_error("class " + name + " is not a Stage 3 class");
  return static_cast<int32_t>(found - names.begin());
}

Feature feature_of(const std::string& name) {
  if (name == "nll_q") return Feature::kNll;
  if (name == "kl_tail_q") return Feature::kKlTail;
  if (name == "burst_q") return Feature::kBurst;
  if (name == "novel_burst_q") return Feature::kNovelBurst;
  if (name == "novel_share_q") return Feature::kNovelShare;
  throw std::runtime_error("unknown detector feature " + name);
}

RuleParams parse_rule(const json& spec) {
  const json& rule = spec["rule"];
  std::vector<std::string> class_names = spec["classes"];
  RuleParams params;
  params.latent_dim = rule["latent_dim"];
  params.divisor = rule["divisor"].get<std::vector<double>>();
  params.novelty = rule["novelty"];
  params.burst = rule["burst"];
  params.min_support = rule["min_support"];
  params.window_seconds = rule["window_seconds"];
  for (const json& attack : rule["attack_classes"]) params.attack_order.push_back(index_of(class_names, attack.get<std::string>()));
  if (params.divisor.size() != params.latent_dim) throw std::runtime_error("rule divisor size differs from the latent size");
  return params;
}

DetectorParams parse_detector(const std::string& root, const json& detector_spec) {
  DetectorParams params;
  params.latent_dim = detector_spec["latent_dim"];
  for (const json& column : detector_spec["columns"]) params.columns.push_back(feature_of(column.get<std::string>()));
  params.divisor = detector_spec["divisor"].get<std::vector<double>>();
  params.window_seconds = detector_spec["window_seconds"];
  params.novelty = detector_spec["novelty"];
  params.nll_reference = read_array<double>(root + "/" + detector_spec["nll_reference"].get<std::string>());
  params.kl_reference = read_array<double>(root + "/" + detector_spec["kl_reference"].get<std::string>());
  params.burst_reference = read_array<double>(root + "/" + detector_spec["stream_references"][0].get<std::string>());
  params.novel_burst_reference = read_array<double>(root + "/" + detector_spec["stream_references"][1].get<std::string>());
  params.share_reference = read_array<double>(root + "/" + detector_spec["stream_references"][2].get<std::string>());
  if (!detector_spec["background"].is_null()) {
    const json& background = detector_spec["background"];
    params.background = BackgroundParams{background["burst_cutoff"], background["fallback_median"], background["reservoir"], background["warm_up"], background["refresh"]};
  }
  params.cutoff = static_cast<float>(detector_spec["cutoff"].get<double>());
  if (params.divisor.size() != params.latent_dim) throw std::runtime_error("detector divisor size differs from the latent size");
  return params;
}

// merged_rule, separate_rule and raw_rule: model graphs, then the correction, then (optional) the detector.
class RulePipeline : public Pipeline {
 public:
  enum class Kind { kMerged, kSeparate, kRaw };

  RulePipeline(Ort::Env& env, const std::string& root, const json& spec, const Data& data) : kind_(kind_of(spec["kind"])), classes_(spec["classes"].size()) {
    const std::string& input = spec["input"];
    window_ = &data.window.at(input);
    const Matrix& earlier = data.earlier.at(input);
    window_seconds_ = &data.window_seconds;
    std::vector<std::string> class_names = spec["classes"];
    rule_params_ = parse_rule(spec);
    benign_ = index_of(class_names, data.benign_label);
    latent_dim_ = rule_params_.latent_dim;
    int64_t feature_width = static_cast<int64_t>(latent_dim_ + 2);
    int64_t columns = static_cast<int64_t>(window_->columns);
    if (kind_ == Kind::kMerged) {
      parts_ = {"model"};
      merged_.emplace(env, root + "/" + spec["graph"].get<std::string>(), std::vector<std::string>{"vae_features", "probabilities"},
                      std::vector<size_t>{latent_dim_ + 2, classes_}, std::vector<std::vector<int64_t>>{{1, columns}});
    } else {
      parts_ = {"vae", "classifier"};
      if (kind_ == Kind::kRaw) {
        vae_columns_ = spec["vae_columns"].get<std::vector<size_t>>();
        classifier_columns_ = spec["classifier_columns"].get<std::vector<size_t>>();
      }
      int64_t vae_width = kind_ == Kind::kRaw ? static_cast<int64_t>(vae_columns_.size()) : columns;
      int64_t classifier_width = kind_ == Kind::kRaw ? static_cast<int64_t>(classifier_columns_.size()) : feature_width;
      vae_.emplace(env, root + "/" + spec["vae_graph"].get<std::string>(), std::vector<std::string>{"vae_features"}, std::vector<size_t>{latent_dim_ + 2},
                   std::vector<std::vector<int64_t>>{{1, vae_width}});
      classifier_.emplace(env, root + "/" + spec["classifier_graph"].get<std::string>(), std::vector<std::string>{"probabilities"}, std::vector<size_t>{classes_},
                          std::vector<std::vector<int64_t>>{{1, classifier_width}});
      scratch_.resize(std::max(vae_columns_.size(), classifier_columns_.size()));
    }
    parts_.push_back("rule");
    if (!spec["detector"].is_null()) {
      const json& detector_spec = spec["detector"];
      parts_.push_back("detector");
      DetectorParams params = parse_detector(root, detector_spec);
      if (params.latent_dim != latent_dim_) throw std::runtime_error("the rule and the detector have different latent sizes");
      detector_params_ = std::move(params);
      detector_.emplace(env, root + "/" + detector_spec["graph"].get<std::string>(), std::vector<std::string>{"probabilities"}, std::vector<size_t>{2},
                        std::vector<std::vector<int64_t>>{{1, static_cast<int64_t>(detector_params_->columns.size())}});
      detector_row_.resize(detector_params_->columns.size());
    }
    warm_.emplace(fresh_state());
    warm(earlier, data.earlier_seconds, data.warm_batch);
  }

  void start_pass() override { state_.emplace(*warm_); }
  void end_pass() override { state_.reset(); }

  int32_t step(size_t flow, double* part_seconds) override {
    const double* raw = window_->row(flow);
    int64_t second = (*window_seconds_)[flow];
    size_t part = 0;
    const float* features;
    const float* probabilities;
    auto start = Clock::now();
    if (kind_ == Kind::kMerged) {
      merged_->set_input(0, raw, window_->columns);
      merged_->run();
      features = merged_->output(0);
      probabilities = merged_->output(1);
      part_seconds[part++] += elapsed_seconds(start);
    } else {
      if (kind_ == Kind::kRaw) {
        for (size_t index = 0; index < vae_columns_.size(); ++index) scratch_[index] = raw[vae_columns_[index]];
        vae_->set_input(0, scratch_.data(), vae_columns_.size());
      } else {
        vae_->set_input(0, raw, window_->columns);
      }
      vae_->run();
      features = vae_->output(0);
      part_seconds[part++] += elapsed_seconds(start);
      start = Clock::now();
      if (kind_ == Kind::kRaw) {
        for (size_t index = 0; index < classifier_columns_.size(); ++index) scratch_[index] = raw[classifier_columns_[index]];
        classifier_->set_input(0, scratch_.data(), classifier_columns_.size());
      } else {
        classifier_->set_input(0, features, latent_dim_ + 2);
      }
      classifier_->run();
      probabilities = classifier_->output(0);
      part_seconds[part++] += elapsed_seconds(start);
    }
    start = Clock::now();
    int32_t predicted = first_argmax(probabilities, classes_);
    int32_t label = state_->rule.step(second, features, static_cast<double>(features[latent_dim_]), predicted);
    part_seconds[part++] += elapsed_seconds(start);
    if (!detector_) return label;
    start = Clock::now();
    state_->detector->features(second, features, static_cast<double>(features[latent_dim_]), static_cast<double>(features[latent_dim_ + 1]), detector_row_.data());
    detector_->set_input(0, detector_row_.data(), detector_row_.size());
    detector_->run();
    bool flagged = detector_->output(0)[1] > detector_params_->cutoff;
    if (label == benign_ && flagged) label = static_cast<int32_t>(classes_);
    part_seconds[part] += elapsed_seconds(start);
    return label;
  }

 private:
  static Kind kind_of(const std::string& kind) {
    if (kind == "merged_rule") return Kind::kMerged;
    if (kind == "separate_rule") return Kind::kSeparate;
    if (kind == "raw_rule") return Kind::kRaw;
    throw std::runtime_error("unknown rule pipeline kind " + kind);
  }
  StreamState fresh_state() const {
    StreamState state{OnlineRule(rule_params_, classes_), std::nullopt};
    if (detector_params_) state.detector.emplace(*detector_params_);
    return state;
  }

  std::vector<double> gather(const Matrix& matrix, size_t first, size_t count, const std::vector<size_t>& columns) const {
    std::vector<double> rows(count * columns.size());
    for (size_t row = 0; row < count; ++row) {
      for (size_t index = 0; index < columns.size(); ++index) rows[row * columns.size() + index] = matrix.row(first + row)[columns[index]];
    }
    return rows;
  }

  // python_parts.warm_states: the correction and detector states see the earlier flows of the day, with the model outputs of
  // batches of warm_batch flows. The detector score does not change any state, so its graph is not run here.
  void warm(const Matrix& earlier, const std::vector<int64_t>& seconds, size_t warm_batch) {
    size_t width = latent_dim_ + 2;
    std::vector<double> detector_row(detector_params_ ? detector_params_->columns.size() : 0);
    for (size_t first = 0; first < earlier.rows; first += warm_batch) {
      size_t count = std::min(warm_batch, earlier.rows - first);
      std::vector<float> features, probabilities;
      if (kind_ == Kind::kMerged) {
        std::vector<double> rows(earlier.row(first), earlier.row(first) + count * earlier.columns);
        auto outputs = merged_->run_batch(rows, count, earlier.columns);
        features = std::move(outputs[0]);
        probabilities = std::move(outputs[1]);
      } else if (kind_ == Kind::kSeparate) {
        std::vector<double> rows(earlier.row(first), earlier.row(first) + count * earlier.columns);
        features = std::move(vae_->run_batch(rows, count, earlier.columns)[0]);
        std::vector<double> feature_rows(features.begin(), features.end());
        probabilities = std::move(classifier_->run_batch(feature_rows, count, width)[0]);
      } else {
        features = std::move(vae_->run_batch(gather(earlier, first, count, vae_columns_), count, vae_columns_.size())[0]);
        probabilities = std::move(classifier_->run_batch(gather(earlier, first, count, classifier_columns_), count, classifier_columns_.size())[0]);
      }
      if (features.size() != count * width || probabilities.size() != count * classes_) throw std::runtime_error("unexpected warm output size");
      for (size_t row = 0; row < count; ++row) {
        const float* feature_row = features.data() + row * width;
        int64_t second = seconds[first + row];
        warm_->rule.step(second, feature_row, static_cast<double>(feature_row[latent_dim_]), first_argmax(probabilities.data() + row * classes_, classes_));
        if (warm_->detector) {
          warm_->detector->features(second, feature_row, static_cast<double>(feature_row[latent_dim_]), static_cast<double>(feature_row[latent_dim_ + 1]), detector_row.data());
        }
      }
    }
  }

  Kind kind_;
  size_t classes_;
  size_t latent_dim_ = 0;
  int32_t benign_ = 0;
  const Matrix* window_ = nullptr;
  const std::vector<int64_t>* window_seconds_ = nullptr;
  RuleParams rule_params_;
  std::optional<DetectorParams> detector_params_;
  std::optional<Graph> merged_, vae_, classifier_, detector_;
  std::vector<size_t> vae_columns_, classifier_columns_;
  std::vector<double> scratch_, detector_row_;
  std::optional<StreamState> warm_, state_;
};

// ------------------------------------------------------------------------------------------------ protocol

struct Entry {
  std::string name;
  std::unique_ptr<Pipeline> pipeline;
  std::vector<int32_t> reference;
};

// One pass over the first `flows` sample flows; returns the loop seconds and fills the labels and part seconds.
double timed_pass(Pipeline& pipeline, size_t flows, std::vector<int32_t>& labels, std::vector<double>& part_seconds) {
  std::fill(part_seconds.begin(), part_seconds.end(), 0.0);
  pipeline.start_pass();
  auto loop_start = Clock::now();
  for (size_t flow = 0; flow < flows; ++flow) labels[flow] = pipeline.step(flow, part_seconds.data());
  double loop_seconds = elapsed_seconds(loop_start);
  pipeline.end_pass();
  return loop_seconds;
}

size_t mismatches(const std::vector<int32_t>& labels, const std::vector<int32_t>& reference, size_t flows) {
  size_t count = 0;
  for (size_t flow = 0; flow < flows; ++flow) count += labels[flow] != reference[flow];
  return count;
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  size_t middle = values.size() / 2;
  return values.size() % 2 == 1 ? values[middle] : (values[middle - 1] + values[middle]) / 2.0;
}

std::string cpu_model() {
  std::ifstream cpuinfo("/proc/cpuinfo");
  std::string line;
  while (std::getline(cpuinfo, line)) {
    if (line.rfind("model name", 0) == 0) {
      std::string value = line.substr(line.find(':') + 1);
      return value.substr(value.find_first_not_of(' '));
    }
  }
  return "unknown";
}

void write_json(const std::string& path, const json& value) {
  std::string temporary = path + ".tmp";
  {
    std::ofstream stream(temporary);
    stream << value.dump(2) << "\n";
    if (!stream) throw std::runtime_error("cannot write " + temporary);
  }
  if (std::rename(temporary.c_str(), path.c_str()) != 0) throw std::runtime_error("cannot move " + temporary + " to " + path);
}

template <typename T>
void write_array(const std::string& path, const std::vector<T>& values) {
  std::ofstream stream(path, std::ios::binary);
  stream.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(T)));
  if (!stream) throw std::runtime_error("cannot write " + path);
}

// CPU ticks (user + system) of every thread of this process except the main thread. The ONNX Runtime environment starts
// one idle thread; the timing requires that no thread but the main one computes.
long other_thread_ticks() {
  long ticks = 0;
  std::string main_id = std::to_string(getpid());
  for (const auto& task : std::filesystem::directory_iterator("/proc/self/task")) {
    if (task.path().filename() == main_id) continue;
    std::string stat = read_text((task.path() / "stat").string());
    // Fields after the parenthesised name: state is field 3, utime field 14, stime field 15.
    std::istringstream fields(stat.substr(stat.rfind(')') + 2));
    std::string field;
    long utime = 0, stime = 0;
    for (int number = 3; number <= 15 && fields >> field; ++number) {
      if (number == 14) utime = std::stol(field);
      if (number == 15) stime = std::stol(field);
    }
    ticks += utime + stime;
  }
  return ticks;
}

// Correction and detector of one pipeline on a stream of given model outputs, from an empty state (tests/test_compiled_stream.py):
// <prefix>_seconds.i64, <prefix>_features.f32 (rows of latent_dim + 2), <prefix>_predicted.i32 -> <prefix>_cpp_labels.i32 (rule
// labels) and <prefix>_cpp_features.f64 (detector rows).
int stream_mode(const std::string& root, const json& manifest, const std::string& prefix, const std::string& name) {
  const json* spec = nullptr;
  for (const json& pipeline : manifest["pipelines"]) {
    if (pipeline["name"] == name) spec = &pipeline;
  }
  if (spec == nullptr || !spec->contains("rule")) throw std::runtime_error("no pipeline with a correction named " + name);
  RuleParams rule_params = parse_rule(*spec);
  std::optional<DetectorParams> detector_params;
  if (!(*spec)["detector"].is_null()) detector_params = parse_detector(root, (*spec)["detector"]);
  std::vector<int64_t> seconds = read_array<int64_t>(prefix + "_seconds.i64");
  std::vector<float> features = read_array<float>(prefix + "_features.f32");
  std::vector<int32_t> predicted = read_array<int32_t>(prefix + "_predicted.i32");
  size_t width = rule_params.latent_dim + 2;
  if (features.size() != seconds.size() * width || predicted.size() != seconds.size()) throw std::runtime_error("stream arrays have different lengths");
  OnlineRule rule(rule_params, (*spec)["classes"].size());
  std::optional<OnlineDetector> online;
  size_t columns = detector_params ? detector_params->columns.size() : 0;
  if (detector_params) online.emplace(*detector_params);
  std::vector<int32_t> labels(seconds.size());
  std::vector<double> rows(seconds.size() * columns);
  for (size_t flow = 0; flow < seconds.size(); ++flow) {
    const float* row = features.data() + flow * width;
    labels[flow] = rule.step(seconds[flow], row, static_cast<double>(row[rule_params.latent_dim]), predicted[flow]);
    if (online) online->features(seconds[flow], row, static_cast<double>(row[rule_params.latent_dim]), static_cast<double>(row[rule_params.latent_dim + 1]), rows.data() + flow * columns);
  }
  write_array(prefix + "_cpp_labels.i32", labels);
  write_array(prefix + "_cpp_features.f64", rows);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: timing <export dir> check|time <output.json> | timing <export dir> stream <stream prefix> <pipeline name>\n";
    return 2;
  }
  try {
    std::string root = argv[1];
    std::string mode = argv[2];
    std::string output = argv[3];
    json manifest = json::parse(read_text(root + "/manifest.json"));
    if (mode == "stream") {
      if (argc != 5) throw std::runtime_error("usage: timing <export dir> stream <stream prefix> <pipeline name>");
      return stream_mode(root, manifest, argv[3], argv[4]);
    }
    if (argc != 4) throw std::runtime_error("usage: timing <export dir> check|time <output.json>");
    double session_seconds = manifest["session_seconds"];
    double cooldown_seconds = manifest["cooldown_seconds"];
    if (!(session_seconds > 0.0) || !(cooldown_seconds >= 0.0)) throw std::runtime_error("session seconds must be positive and cooldown seconds not negative");
    // The sessions use one thread (session_options) and one flow per call (Graph); the export must ask for the same.
    if (manifest["threads"] != 1 || manifest["batch_size"] != 1) throw std::runtime_error("the program times one thread at batch 1");
    size_t sample_count = manifest["sample_count"];
    size_t warmup_flows = manifest["warmup_flows"];
    int sessions = manifest["sessions"];

    Data data;
    data.window_seconds = read_array<int64_t>(root + "/" + manifest["seconds"]["window"].get<std::string>());
    data.earlier_seconds = read_array<int64_t>(root + "/" + manifest["seconds"]["earlier"].get<std::string>());
    data.benign_label = manifest["benign_label"];
    data.warm_batch = manifest["warm_batch"];
    if (data.warm_batch == 0) throw std::runtime_error("warm_batch must be positive");
    for (auto& [id, input] : manifest["inputs"].items()) {
      size_t columns = input["columns"];
      data.window[id] = read_matrix(root + "/" + input["window"].get<std::string>(), columns);
      if (!input["earlier"].is_null()) {
        data.earlier[id] = read_matrix(root + "/" + input["earlier"].get<std::string>(), columns);
        if (data.earlier[id].rows != data.earlier_seconds.size()) throw std::runtime_error(id + ": earlier rows differ from the earlier seconds");
      }
    }
    if (data.window_seconds.size() != sample_count) throw std::runtime_error("the sample seconds do not hold sample_count flows");

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "timing");
    std::vector<Entry> entries;
    for (const json& spec : manifest["pipelines"]) {
      const std::string& kind = spec["kind"];
      const Matrix& window = data.window.at(spec["input"].get<std::string>());
      if (window.rows != sample_count) throw std::runtime_error(spec["name"].get<std::string>() + ": sample rows differ from sample_count");
      Entry entry{spec["name"], nullptr, read_array<int32_t>(root + "/" + spec["reference"].get<std::string>())};
      if (kind == "graph") {
        entry.pipeline = std::make_unique<GraphPipeline>(env, root + "/" + spec["graph"].get<std::string>(), window, spec["classes"].size());
      } else if (kind == "flow_transformer") {
        entry.pipeline = std::make_unique<FlowTransformerPipeline>(env, root + "/" + spec["graph"].get<std::string>(), window, spec["classes"].size(),
                                                                   spec["window_size"].get<size_t>());
      } else {
        entry.pipeline = std::make_unique<RulePipeline>(env, root, spec, data);
      }
      if (entry.reference.size() != sample_count) throw std::runtime_error(entry.name + ": reference labels differ from sample_count");
      entries.push_back(std::move(entry));
    }
    long ticks_before = other_thread_ticks();

    std::vector<int32_t> labels(sample_count);
    if (mode == "check") {
      json report = json::object();
      size_t total_mismatches = 0;
      for (Entry& entry : entries) {
        std::vector<double> part_seconds(entry.pipeline->parts().size());
        timed_pass(*entry.pipeline, sample_count, labels, part_seconds);
        size_t wrong = mismatches(labels, entry.reference, sample_count);
        total_mismatches += wrong;
        report[entry.name] = {{"flows", sample_count}, {"label_mismatches", wrong}};
        std::cout << entry.name << ": " << wrong << " label mismatches\n" << std::flush;
      }
      write_json(output, report);
      return total_mismatches == 0 ? 0 : 1;
    }
    if (mode != "time") throw std::runtime_error("unknown mode " + mode);

    for (Entry& entry : entries) {
      std::vector<double> part_seconds(entry.pipeline->parts().size());
      timed_pass(*entry.pipeline, warmup_flows, labels, part_seconds);
      if (mismatches(labels, entry.reference, warmup_flows) != 0) throw std::runtime_error(entry.name + ": warm-up labels differ from Python");
    }
    std::vector<std::vector<std::vector<double>>> pass_values(entries.size(), std::vector<std::vector<double>>(sessions));
    std::vector<std::vector<std::vector<double>>> part_values(entries.size());
    for (size_t index = 0; index < entries.size(); ++index) part_values[index].resize(entries[index].pipeline->parts().size());
    for (int session = 0; session < sessions; ++session) {
      if (session > 0) std::this_thread::sleep_for(std::chrono::duration<double>(cooldown_seconds));
      auto end = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(session_seconds));
      while (Clock::now() < end) {
        for (size_t index = 0; index < entries.size(); ++index) {
          Entry& entry = entries[index];
          std::vector<double> part_seconds(entry.pipeline->parts().size());
          double seconds = timed_pass(*entry.pipeline, sample_count, labels, part_seconds);
          if (mismatches(labels, entry.reference, sample_count) != 0) throw std::runtime_error(entry.name + ": timed-pass labels differ from Python");
          pass_values[index][session].push_back(1000.0 * seconds / sample_count);
          for (size_t part = 0; part < part_seconds.size(); ++part) part_values[index][part].push_back(1000.0 * part_seconds[part] / sample_count);
        }
      }
    }
    long other_ticks = other_thread_ticks() - ticks_before;
    if (other_ticks != 0) throw std::runtime_error("threads other than the main thread used " + std::to_string(other_ticks) + " CPU ticks");
    json timings = json::object();
    for (size_t index = 0; index < entries.size(); ++index) {
      json session_list = json::array();
      std::vector<double> all_passes;
      for (const auto& values : pass_values[index]) {
        session_list.push_back({{"passes", values.size()}, {"median", median(values)}, {"min", *std::min_element(values.begin(), values.end())},
                                {"max", *std::max_element(values.begin(), values.end())}});
        all_passes.insert(all_passes.end(), values.begin(), values.end());
      }
      json parts = json::object();
      for (size_t part = 0; part < part_values[index].size(); ++part) parts[entries[index].pipeline->parts()[part]] = median(part_values[index][part]);
      timings[entries[index].name + "_batch1"] = {{"sessions", session_list}, {"median", median(all_passes)}, {"parts", parts}};
    }
    json report = {{"cpu", cpu_model()}, {"threads", manifest["threads"]}, {"seed", manifest["seed"]}, {"sample_count", sample_count},
                   {"batch_size", manifest["batch_size"]}, {"session_seconds", session_seconds}, {"cooldown_seconds", cooldown_seconds},
                   {"runtime", "C++ ONNX Runtime " + std::string(Ort::GetVersionString())},
                   {"other_thread_cpu_ticks", other_ticks}, {"timings", timings}};
    write_json(output, report);
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
  return 0;
}
