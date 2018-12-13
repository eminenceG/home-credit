require "./helpers"
require "./models.rb"
require "./data-reader"
require "daru"

$bureau_numeric_features = ["DAYS_CREDIT", "CREDIT_DAY_OVERDUE", "DAYS_CREDIT_ENDDATE", "DAYS_ENDDATE_FACT",
                            "AMT_CREDIT_MAX_OVERDUE", "CNT_CREDIT_PROLONG", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_LIMIT",
                            "AMT_CREDIT_SUM_OVERDUE", "DAYS_CREDIT_UPDATE", "AMT_ANNUITY"]

def add_ratio_feature(new_feature_name, feature1, feature2, row)
  r_features = row["features"]
  if (r_features[feature1] == "" || r_features[feature2] == "")
    r_features[new_feature_name] = ""
  else
    r_features[new_feature_name] = r_features[feature1] / r_features[feature2]
    r_features[new_feature_name] = "" if r_features[new_feature_name] == nil
  end
end

# Returns the model ready data
def prepare_data(num_rows)
  app_train = read_dataset_from_csv('./input/application_train', num_rows)
  data = app_train["data"]
  set_categorical_labels(data)
  numeric_features, _categorical_features = get_numeric_cateforical_features_from_the_raw_dataset(app_train["data"])
  categorical_features_one_hot_encode = get_one_hot_feature_map_from_the_origin_dataset(data)
  one_hot_encoding_using_feature_map(data, categorical_features_one_hot_encode)

  # NaN values for DAYS_EMPLOYED: 365243 -> nan
  data.each do |r|
    r_features = r["features"]
    if r_features["days_empoyed"] == 365243
      r_features["days_employed"] = ""
      r_features["days_employed_anom"] = 1
    else
      r_features["days_employed_anom"] = 0 
    end 
    
    add_ratio_feature("payment_rate", "amt_annuity", "amt_credit", r)
    add_ratio_feature("annuity_income_ratio", "amt_annuity", "amt_income_total", r)
    add_ratio_feature("credit_goods_ratio", "amt_credit", "amt_goods_price", r)
    # add_ratio_feature("income_person_ratio", "amt_income_total", "cnt_fam_members", r)
    add_ratio_feature("employed_birth_ratio", "days_employed", "days_birth", r)
  end
  # categorical_features << "days_employed_anom"

  bureau = read_dataset_from_csv('./input/bureau', 1000, $bureau_numeric_features)
  # bureau["data"].each do |r|
    # puts r["features"]["days_enddate_fact"]
  # end
  # return
  grouped = group_data(bureau["data"])
  agged = agg_group_data(grouped, $bureau_numeric_features, "bureau")
  merge_to_dataset(app_train, agged, $bureau_numeric_features)

  app_train["features"] = app_train["data"][0]["features"].keys

  puts "begin to normalize the dataset......"
  nomalizer = Normalizer.new
  nomalizer.normalize(app_train, numeric_features)

  puts "begin to impute missing value......"
  imputer = SimpleImputer.new
  imputer.fit(app_train)
  
  puts "finish preparing the dataset!"
  return app_train
end

# Trains the model using SGD
def train(sgd, obj, w, train_set, num_epoch = 50, batch_size = 2000)
  iters = []
  losses = []
  iters << 0
  losses << obj.func(train_set["data"], sgd.weights)

  num_epoch.times do |t|
    if t % 10 == 0 
      puts "# epoch: #{t}"
    end
    train_set["data"].each_slice(batch_size) do |batch|    
      sgd.update(batch)
      # iters << i
      # losses << obj.func(batch, sgd.weights)
      # i += 1
    end
    iters << t + 1
    losses << obj.func(train_set["data"], sgd.weights)
  end
  scores = score_binary_classification_model(train_set["data"], w, obj)
  train_auc_1 = calc_auc_only_1(scores)
  puts train_auc_1
  return [sgd, iters, losses]
end

# Gets the trained model from the train_set.
def train_one_classifier(dataset)
  model = LogisticRegressionModelL2.new(0.000)
  lr = 1
  w = Hash.new {|h,k| h[k] = (rand * 0.1) - 0.05}
  sgd = StochasticGradientDescent.new(model, w, lr)
  sgd, iter, losses = train(sgd, model, w, dataset, num_epoch = 18, batch_size = 20)
  df = Daru::DataFrame.new({x: iter, y: losses})
  df.plot(type: :line, x: :x, y: :y) do |plot, diagram|
    plot.x_label "Batches"
    plot.y_label "Cumulative Loss"
  end.export_html
  return sgd
end

def train_test_split(dataset)
  train_set =  {"classes" => dataset["classes"], "features" => dataset["features"], "data" => nil}
  dev_set   =  {"classes" => dataset["classes"], "features" => dataset["features"], "data" => nil}
  data = dataset["data"]
  train_data = []
  dev_data = []
  total_len = data.length
  train_len = total_len * 7 / 10
  train_len = (total_len - train_len > 10000 ? total_len - 10000 : train_len)  
  data.shuffle.each.with_index do |r, i|
    if i < train_len
      train_data << r
    else
      dev_data << r
    end
  end
  train_set["data"] = train_data
  dev_set["data"] = dev_data
  return train_set, dev_set
end

if __FILE__ == $0
  num_rows = 30000
  dataset = prepare_data(num_rows)
  # data = Marshal.dump(dataset)

  train_set, dev_set = train_test_split(dataset)
  puts train_set["data"].length
  puts dev_set["data"].length

  1.times do |i| 
    sgd = train_one_classifier(train_set)
    scores = score_binary_classification_model(dev_set["data"], sgd.weights, sgd.objective)
    puts calc_auc_only_1(scores)
  end
end