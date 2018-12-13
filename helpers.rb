# ===================  Get raw data from the data base ================ #
def parse_row_to_sample(row)
  res = Hash.new
  data = Hash.new
  row.each do |k, v|
    if k == "SK_ID_CURR"
      res["id"] = v
    elsif k == "TARGET"
      res["label"] = v
    elsif k.is_a? String
      data[k.downcase] = v
    end
  end
    res["features"] = data
  return res 
end

def create_dataset(db, sql)
  dataset = []
  db.execute sql do |row|
    dataset << parse_row_to_sample(row)
  end
  puts "finish getting data from the server"
  return dataset
end
# ==================================================================== #

# ====================== Dataset Preprocess ========================== #

# Sets the categorical features' values to be string type.
# dataset <- dataset
def set_categorical_labels(data)
  puts "set categorical features to be String"
  
  data = data.each do |row|
    row["features"].each do |k, v|
      # some futures are categorical, but use integer, so we need to convert to string 
      if (k.start_with?("flag"))
        row["features"][k] = v.to_s
      end
      
      # some features are numeric, but use string, so we need to convert to float number 
      if (k.start_with?("obs_", "def_") && row["features"][k] != "")
        row["features"][k] = v.to_f  
      end
    end
  end

  puts "finish setting features to be String"
end

def preprocess(dataset)
  dataset = set_categorical_labels(dataset)
  numeric_features, categorical_features = get_numeric_cateforical_features_from_the_raw_dataset(dataset)
  dataset = one_hot_encoding_using_feature_map(dataset, get_one_hot_feature_map_from_the_origin_dataset(dataset))
  
  # NaN values for DAYS_EMPLOYED: 365243 -> nan
  dataset.each do |row|
    if row["features"]["days_empoyed"] == 365243
      row["features"]["days_employed"] = ""
      row["features"]["days_employed_anom"] = 1
    else
      row["features"]["days_employed_anom"] = 0 
    end    
  end
  fill_numeric_missing_values(numeric_features, dataset)    
  return dataset   
end

# ==================================================================== #
    
def find_features_using_information_gain(dataset, threshold)
  features = dataset[0]["features"].keys
    
  numeric_features = features.select do |k| 
    dataset.reject {|row| row["features"][k] == ""}.all? do |row|
      row["features"].fetch(k, 0.0).is_a? Numeric 
    end
  end
    
  dist = class_distribution(dataset)
  h0 = entropy dist
  feature_ig = Hash.new
  features.each do |feature|
    if numeric_features.include?(feature)
      t, ig = find_split_point_numeric(dataset, h0, feature)
    else
      ig = calculate_ig_categorical(dataset, h0, feature) 
    end
    feature_ig[feature] = ig
  end
  feature_ig = feature_ig.sort_by {|_key, value| -value}
  return feature_ig.select {|r| r[1] >= threshold}.collect {|r| r[0]}
end

def extract_features(db)
  dataset = []
  # BEGIN YOUR CODE
  application_train = create_dataset(db, "SELECT * FROM application_train")
  application_train = preprocess(application_train)
  application_train_features = find_features_using_information_gain(application_train, 0.005)
  sql = get_sql_from_features(application_train_features)
  dataset = create_dataset(db, sql)
  #END YOUR CODE
  return dataset
end
# ====================== Join Table functions ====================== #
def group_data(data)
  res = Hash.new
  data.each do |r|
    id = r["id"]
    if !res.has_key?(id)
      res[id] = Array.new
    end
    # puts id
    # puts res[id].class
    res[id] << r["features"]
  end
  return res
end

def agg_group_data(grouped, numeric_features, table_name)
  res = Hash.new 
  grouped.each do |k, v|
    if !res.has_key?(k)
      res[k] = Hash.new
    end
    res[k][table_name + "count"] = v.length
    numeric_features.each do |f|
      f = f.downcase
      not_null = v.select {|r| r[f] != ""}.collect {|row| row[f]}
      if (not_null.length == 0)
        res[k][f + "_avg"] = ""
        res[k][f + "_min"] = ""
        res[k][f + "_max"] = ""
        res[k][f + "_sum"] = ""
        next
      end

      min = not_null.min
      max = not_null.max
      sum = not_null.sum(0.0)
      avg = sum / not_null.length

      res[k][f + "_avg"] = avg
      res[k][f + "_min"] = min
      res[k][f + "_max"] = max
      res[k][f + "_sum"] = sum
    end
  end
  return res
end

def merge_to_dataset(dataset, agged_data, numeric_features) 
  data = dataset["data"]
  data.each do |r|
    id = r["id"].to_f
    if agged_data.has_key?(id)
      r["features"].merge!(agged_data[id])
    else
      tmp = Hash.new("")
      numeric_features.each do |f|
        f = f.downcase
        tmp[f + "_avg"] = ""
        tmp[f + "_min"] = ""
        tmp[f + "_max"] = ""
        tmp[f + "_sum"] = ""
      end
      r["features"].merge!(tmp)
    end
  end
end

# ====================== Helper functions ========================== #

# Returns the sql to get all columns in features from the application_train table
def get_sql_from_features(features)
  sql = "SELECT TARGET, sk_id_curr"
  features.each {|f| sql = sql + ", " + f}
  sql = sql + " FROM application_train"
  return sql
end

# Calculates the class distribution of a dataset
def class_distribution dataset
  counts = Hash.new {|h,k| h[k] = 0}
  sum = 0.0
  
  dataset.each do |row| 
    counts[row["label"]] += 1
    sum += 1
  end
  
  counts.each_key {|k| counts[k] /= sum}
  return counts
end

# Calculates the entropy of a distribution 
def entropy dist
  freq_x = dist.values
  total_freq = freq_x.inject(0.0) {|u,x| u += x}
  return 0.0 if total_freq <= 0.0
  
  prob_x = freq_x.collect {|f| f / total_freq}
  
  -prob_x.inject(0.0) do |u,p| 
    u += p > 0 ? p * Math.log(p) : 0.0
  end
end
# Caculates the information gain of the splits
def information_gain h0, splits
  splits.delete("")
  total_size = splits.values.inject(0.0) {|u,v| u += v.size}
  split_entropy = splits.values.inject(0.0) do |u, v|
    p_v = class_distribution(v)
    h_c_v = entropy(p_v)
    u += (v.size / total_size) * h_c_v
  end
  h0 - split_entropy
end

# Calcutlates information gain for categorical feature
#
# dataset: the dataset  
# h0:      the entropy of the original dataset   
# fname:   the feature name
# => the information gain using fname
def calculate_ig_categorical(dataset, h0, fname)
  splits = dataset.group_by {|row| row["features"][fname]}
  ig = information_gain(h0, splits)
end

# Finds the best split point and the information gain of a numeric feature.
# x: the dataset
# h0: the entropy
# fname : the feature's name
def find_split_point_numeric(x, h0, fname)
  ig_max = 0
  t_max = nil

  feature_groups = x.reject{|r| r["features"][fname] == ""}.group_by {|r| r["features"].fetch(fname, 0.0)}
  counts_right = Hash.new {|h,k| h[k] = 0}
  counts_left = Hash.new {|h,k| h[k] = 0}
  v_left = 0.0
  v_right = x.size.to_f

  feature_groups.each_key do |t|
    counts = Hash.new {|h,k| h[k] = 0}  
    feature_groups[t].each do |r| 
      counts[r["label"]] += 1
      counts_right[r["label"]] += 1
    end
    feature_groups[t] = counts
  end
  
  thresholds = feature_groups.keys.sort
  t = thresholds.shift
  
  feature_groups[t].each_key do |k| 
    counts_left[k] += feature_groups[t][k]
    counts_right[k] -= feature_groups[t][k]
    v_left += feature_groups[t][k]
    v_right -= feature_groups[t][k]
  end
  
  thresholds.each.with_index do |t, i|
    p_left = v_left / x.size
    p_right = v_right / x.size
    
    d_left = Hash.new
    d_right = Hash.new
    counts_left.each_key {|k| d_left[k] = counts_left[k] / v_left}
    counts_right.each_key {|k| d_right[k] = counts_right[k] / v_right}
        
    h_left = entropy(d_left)
    h_right = entropy(d_right)    
    ig = h0 - (p_left * h_left + p_right * h_right)
    if ig > ig_max
      ig_max = ig
      t_max = t
    end

    feature_groups[t].each_key do |k| 
      counts_left[k] += feature_groups[t][k]
      counts_right[k] -= feature_groups[t][k]
      v_left += feature_groups[t][k]
      v_right -= feature_groups[t][k]
    end
  end

  return [t_max, ig_max]
end

def get_numeric_cateforical_features_from_the_raw_dataset(dataset)
  features = dataset.flat_map {|row| row["features"].keys}.uniq
  categorical_features = features.select {|k| dataset.all? {|row| row["features"].fetch(k, "").is_a? String}}
  numeric_features = features.select {|k| dataset.reject {|row| row["features"][k] == ""}.all? {|row| row["features"].fetch(k, 0.0).is_a? Numeric}}
  return numeric_features, categorical_features
end

def get_one_hot_feature_map_from_the_origin_dataset(dataset)
  numeric_features, categorical_features = get_numeric_cateforical_features_from_the_raw_dataset(dataset)
  categorical_features_one_hot_encode = Hash.new
  categorical_features.each do |f|
    uniq_values = dataset.select{|r| r["features"][f] != ""}.collect {|r| r["features"][f]}.uniq
    new_columns = []
    uniq_values.each do |o_h|
      new_columns << "#{f}*is*#{o_h}"
    end
    categorical_features_one_hot_encode[f] = new_columns
  end
  return categorical_features_one_hot_encode
end

# One hot encodes the original dataset, the change to the dataset is in-place.
#
# dataset:                               the original dataset
# categorical_features_one_hot_encoding: a map that maps the original categorical features to the corresponding
#                                        one-hot encodeds features bease on the unique categorial values in this
#                                        feature
def one_hot_encoding_using_feature_map(dataset, categorical_features_one_hot_encode)
  categorical_features_one_hot_encode.each do |k, v|
    dataset.each do |r|
      i = 0
      v.each do |new_feature|
        if i == 0
          # for the one hot encoding, we hope to leave out one category  
        else
          value = new_feature.split("*")[2]
          if r["features"][k] == value
            r["features"][new_feature] = 1
          else
            r["features"][new_feature] = 0
          end
        end
        i += 1
      end
      r["features"].delete(k)
    end
  end
end

def score_binary_classification_model(data, weights, model)
  scores = data.collect do |row|
    s = model.predict(row, weights)
    [s, row["label"] > 0 ? 1.0 : 0.0]
  end
  return scores
end

def dot x, w
  x.keys.select {|k| w.has_key?(k)}.inject(0.0) {|u, k| u += x[k] * w[k]}
end

def norm w
  Math.sqrt(dot(w,w))
end

def calc_auc_only_1(scores)
  total_neg = scores.inject(0.0) {|u,s| u += (1 - s.last)}
  total_pos = scores.inject(0.0) {|u,s| u += s.last}
  c_neg = 0.0
  c_pos = 0.0
  fp = [0.0]
  tp = [0.0]
  auc = 0.0
  scores.sort_by {|s| -s.first}.each do |s|
    c_neg += 1 if s.last <= 0
    c_pos += 1 if s.last > 0  

    fpr = c_neg / total_neg
    tpr = c_pos / total_pos
    auc += 0.5 * (tpr + tp.last) * (fpr - fp.last)
    fp << fpr
    tp << tpr
  end
  return auc
end

def calc_auc_only_2(scores)
  sorted_by_disc = scores.sort { |a,b| a[0] <=> b[0] }
  sorted_labels = sorted_by_disc.collect { |d| d[1] }
  # now let's count the number of positive and negatives:
  pos = sorted_labels.count { |l| l == 1 }.to_f 
  neg = sorted_labels.count { |l| l == 0 }.to_f
  auc = 0.0
  c = 0.0 # how many positives we've seen thus far
  n = 0.0 # how many negatives we've seen thus far
  (sorted_labels.length - 1).downto(0) do |i| # walk backwards through the data...
    if sorted_labels[i] > 0 # pos?
      c += 1.0
    else
      n += 1.0
      auc += (c / (pos * neg)) # update auc
    end
  end
  return auc
end

# ================================================================== #

# ============================= SGD ================================ #
class StochasticGradientDescent
  attr_reader :weights
  attr_reader :objective
  def initialize obj, w_0, lr = 0.01
    @objective = obj
    @weights = w_0
    @n = 1.0
    @lr = lr
  end

  def update x
    dw = @objective.grad(x, @weights)
    learning_rate = @lr / Math.sqrt(@n)
    
    dw.each_key do |k|
      @weights[k] -= learning_rate * dw[k]
    end

    @objective.adjust @weights
    @n += 1.0
  end
end
# ================================================================== #


# This class is used to impute the missing values in the data, only works for numeric features.
class SimpleImputer
  def fit(dataset)
    features = dataset["features"]
    data = dataset["data"]
    total_len = data.length
    features.each do |f|
      series = data.collect {|r| r["features"][f]}.select {|x| x != ''}
      not_missing_len = series.length
      missing_rate = 1 - not_missing_len.to_f / total_len.to_f
      if (missing_rate > 0.75)
        puts "Warning: #{f} missing rate: #{missing_rate}"
      else 
      end
      mean = series.sum(0.0) / series.size
      data.each do |r|
        if (missing_rate > 0.75)
          r["features"].delete(f)
          next
        end
        if r["features"][f] == ''
          r["features"][f] = (mean.nan? ? 0.0 : mean) 
        end
      end
    end
  end
end

# This class is a normalizer that is able to normalize numeric features in the
# dataset. We will use Z normlize.   
class Normalizer

  # Normalizes numeric features in the dataset 
  def normalize(dataset, numeric_features)
    data = dataset["data"]
    means = Hash.new
    stdevs = Hash.new
    numeric_features.each do |f|
      x = data.collect {|r| r["features"][f]}.select {|d| d != ''}
      means[f] = mean(x)
      stdevs[f] = stdev(x)    
      if (stdevs[f] == 0.0)
        next
      end
      data.each do |r|
        if r["features"][f] == ''
          next
        end
        r["features"][f] = (r["features"][f] - means[f]) / stdevs[f]
      end
    end
  end

  def mean x
    sum = x.inject(0.0) {|u,v| u += v}
    sum / x.size
  end

  def stdev x
    m = mean x
    sum = x.inject(0.0) {|u,v| u += (v - m) ** 2.0}
    Math.sqrt(sum / (x.size - 1))
  end
end

# Magic code
def magic
  $test_db = SQLite3::Database.new "/home/abagher/cs6140-datasets/credit_risk_data_test.db", results_as_hash: true, readonly: true
  predictions = eval_one_classifier_on $test_db, model
  assert_equal 15510, predictions.size
  scores = get_labels_for $test_db, predictions
  assert_equal 15510, scores.size
  fp, tp, auc = roc_curve scores
  puts auc
end