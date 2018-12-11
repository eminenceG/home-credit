require "csv"
require "./helpers"
require "./models.rb"
require "daru"

def read_dataset_from_csv(prefix)
  # features = ['EXT_SOURCE_3', 'DAYS_BIRTH', 'DAYS_ID_PUBLISH', 'DAYS_REGISTRATION', 'DAYS_EMPLOYED', 'AMT_ANNUITY',
  #   'DAYS_LAST_PHONE_CHANGE', 'AMT_CREDIT', 'EXT_SOURCE_1', 'AMT_INCOME_TOTAL', 'REGION_POPULATION_RELATIVE',
  #   'AMT_GOODS_PRICE', 'HOUR_APPR_PROCESS_START']
  data = []
  classes = Hash.new {|h,k| h[k] = 0}
  header = []
  File.open(prefix + ".csv").each_line.with_index do |l, i|
    if (i == 0)
      header = l.chomp.split(",")
    else
      # TODO: delete later, just for speed
      # next if i > 100000
      a = CSV.parse(l)
      next if a[0].empty?
      row = Hash.new(0.0)
      row["features"] = Hash.new(0.0)
      # row = {"features" => Hash.new}
      header.each.with_index do |k, i|
        v = a[0][i]
        if k == "TARGET"
          row["label"] = v.to_f
        elsif k == "SK_ID_CURR" 
          row[k.downcase] = v.to_f
        else
          # TODO: delete
          # if !features.include?(k)
          #   next
          # end
          row["features"][k.downcase] = is_number?(v) ? v.to_f : v.to_s
        end
      end
      row["features"]["bias"] = 1.0
      data << row
    end
  end
  return {"classes" => classes, "features" => header[2, header.size - 1].map {|x| x.downcase}, "data" => data}
  # return {"classes" => classes, "features" => features, "data" => data}
end

# Sets the categorical features' values to be string type.
# dataset <- dataset
def set_categorical_labels(dataset)
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
  return dataset
end

def is_number? string
  true if Float(string) rescue false
end

# Returns the model ready data
def prepare_data()
  app_train = read_dataset_from_csv('./input/application_train')
  set_categorical_labels(app_train["data"])
  numeric_features, _categorical_features = get_numeric_cateforical_features_from_the_raw_dataset(app_train["data"])
  categorical_features = one_hot_encoding_using_feature_map(app_train["data"],
     get_one_hot_feature_map_from_the_origin_dataset(app_train["data"]))
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
def train(sgd, obj, w, train_set, num_epoch = 500, batch_size = 2000)
  i = 0
  iters = []
  losses = []

  num_epoch.times do |t|
    if (t % 50 == 0)
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
  train_auc = calc_auc_only(scores)
  puts train_auc
  return [sgd, iters, losses]
end

# Gets the trained model from the train_set.
def train_one_classifier(dataset)
  model = LogisticRegressionModelL2.new(0.0)
  lr = 0.1
  w = Hash.new {|h,k| h[k] = (rand * 0.1) - 0.05}
  sgd = StochasticGradientDescent.new(model, w, lr)
  sgd, iter, losses = train(sgd, model, w, dataset, num_epoch = 1, batch_size = 200)
  df = Daru::DataFrame.new({x: iter, y: losses})
  df.plot(type: :line, x: :x, y: :y) do |plot, diagram|
    plot.x_label "Batches"
    plot.y_label "Cumulative Loss"
  end.export_html
  return sgd
end

if __FILE__ == $0
  dataset = prepare_data()
  sgd = train_one_classifier(dataset)
end