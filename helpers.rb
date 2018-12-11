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
    # BEGIN YOUR CODE
    dataset << parse_row_to_sample(row)
    #END YOUR CODE
  end
  puts "finish getting data from the server"
  return dataset
end
# ==================================================================== #

# ====================== Dataset Preprocess ========================== #

# Sets the categorical features' values to be string type.
# dataset <- dataset
def set_categorical_labels(dataset)
  puts "set categorical features to be String"
  
  dataset = dataset.map do |row|
    row["features"].each do |k, v|
      # some futures are categorical, but use integer, so we need to convert to string 
      if (k.start_with?("flag"))
        row["features"][k] = v.to_s
      end
      
      # some features are numeric, but use string, so we need to convert to float number 
      if (k.start_with?("obs_") or k.start_with?("def_"))
        row["features"][k] = v.to_f  
      end
    end
    row  
  end
  puts "finish setting features to be String"
  return dataset
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

# ====================== Helper functions ========================== #

# Returns the sql to get all columns in features from the application_train table
def get_sql_from_features(features)
  sql = "SELECT target, sk_id_curr"
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
  categorical_features = get_numeric_cateforical_features_from_the_raw_dataset(dataset)[1]
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

# @param dataset: the original dataset
# @param categorical_features_one_hot_encoding: a map that maps the original categorical features to the corresponding
#                                               one-hot encodeds features bease on the unique categorial values in this
#                                               feature
def one_hot_encoding_using_feature_map(dataset, categorical_features_one_hot_encode)
  categorical_features_one_hot_encode.each do |k, v|
    if v.length > 2
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
    else
      dataset.each do |r|
        r["features"][k] = r["features"][k].to_f
      end
    end
  end
  return dataset
end

# Fill missing values using the average value of the feature, the change is in-place
#
# numerical_features: all numerical_features as a list
# dataset:            the dataset
def fill_numeric_missing_values(numerical_features, dataset)
  numerical_features.each do |f|
    not_missing = dataset.select {|r| r["features"][f] != ""}.collect {|r| r["features"][f]}
    mean = not_missing.sum(0.0) / not_missing.length
    dataset.each do |r|
      if r["features"][f] == ""
        r["features"][f] = mean
      end
    end
  end
end

def score_binary_classification_model(data, weights, model)
  ### BEGIN SOLUTION
  scores = data.collect do |row|
    s = model.predict row, weights
    [s, row["label"] > 0 ? 1.0 : 0.0]
  end
  ### END SOLUTION
  return scores
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



