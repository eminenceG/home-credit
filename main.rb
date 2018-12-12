require "./helpers"
require "./models.rb"
require "./data-reader"
require "daru"

# Returns the model ready data
def prepare_data(num_rows)
  app_train = read_dataset_from_csv('./input/application_train', num_rows)
  data = app_train["data"]
  set_categorical_labels(data)
  numeric_features, _categorical_features = get_numeric_cateforical_features_from_the_raw_dataset(app_train["data"])
  categorical_features_one_hot_encode = get_one_hot_feature_map_from_the_origin_dataset(data)
  one_hot_encoding_using_feature_map(data, categorical_features_one_hot_encode)

  # NaN values for DAYS_EMPLOYED: 365243 -> nan
  data.each do |row|
    if row["features"]["days_empoyed"] == 365243
      row["features"]["days_employed"] = ""
      row["features"]["days_employed_anom"] = 1
    else
      row["features"]["days_employed_anom"] = 0 
    end    
  end
  # categorical_features << "days_employed_anom"

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
  i = 0
  iters = []
  losses = []

  num_epoch.times do |t|
    if t % 10 == 0 
      puts "# epoch: #{t}"
    end
    train_set["data"].each_slice(batch_size) do |batch|    
      sgd.update(batch)
      iters << i
      losses << obj.func(batch, sgd.weights)
      i += 1
    end
  end
  scores = score_binary_classification_model(train_set["data"], w, obj)
  train_auc_1 = calc_auc_only_1(scores)
  puts train_auc_1
  return [sgd, iters, losses]
end

# Gets the trained model from the train_set.
def train_one_classifier(dataset)
  model = LogisticRegressionModelL2.new(0.001)
  lr = 0.01
  w = Hash.new {|h,k| h[k] = (rand * 0.1) - 0.05}
  sgd = StochasticGradientDescent.new(model, w, lr)
  sgd, iter, losses = train(sgd, model, w, dataset, num_epoch = 18, batch_size = 20)
  df = Daru::DataFrame.new({x: iter, y: losses})
  puts losses[-100..-1]
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
  num_rows = 1000
  dataset = prepare_data(num_rows)
  train_set, dev_set = train_test_split(dataset)
  puts train_set["data"].length
  puts dev_set["data"].length

  1.times do 
    sgd = train_one_classifier(train_set)
    scores = score_binary_classification_model(dev_set["data"], sgd.weights, sgd.objective)
    puts calc_auc_only_1(scores)
  end
end