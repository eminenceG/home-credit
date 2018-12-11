require "csv"
require "./helpers"

def read_dataset_from_csv(prefix)
  data = []
  classes = Hash.new {|h,k| h[k] = 0}
  header = []
  File.open(prefix + ".csv").each_line.with_index do |l, i|
    if (i == 0)
      header = l.chomp.split(",")
    else
      # TODO: delete later, just for speed
      next if i > 100
      a = CSV.parse(l)
      next if a[0].empty?
      row = {"features" => Hash.new}
      header.each.with_index do |k, i|
        v = a[0][i]
        if k == "TARGET"
          row[k.downcase] = v.to_f
        elsif k == "SK_ID_CURR" 
          row[k.downcase] = v.to_f
        else
          row["features"][k.downcase] = is_number?(v) ? v.to_f : v.to_s
        end
      end
      data << row
    end
  end
  return {"classes" => classes, "features" => header[2, header.size - 1].map {|x| x.downcase}, "data" => data}
end

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

def is_number? string
  true if Float(string) rescue false
end

# This class is used to impute the missing values in the data, only works for numeric features.
class SimpleImputer
  def fit(dataset)
    features = dataset["features"]
    data = dataset["data"]
    features.each do |f|
      series = data.collect {|r| r["features"][f]}.select {|x| x != ''}
      mean = series.sum(0.0) / series.size
      data.each do |r|
        if r["features"][f] == ''
          r["features"][f] = mean
        end
      end
    end
  end
end



if __FILE__ == $0
  app_train = read_dataset_from_csv('./input/application_train')
  app_train["data"] = set_categorical_labels(app_train["data"])
  app_train["data"] = one_hot_encoding_using_feature_map(app_train["data"],
     get_one_hot_feature_map_from_the_origin_dataset(app_train["data"]))
  app_train["features"] = app_train["data"][0]["features"].keys
  imputer = SimpleImputer.new
  imputer.fit(app_train)
  puts app_train["data"][0]
  # puts app_train["data"][1]["features"]["name_type_suite"].class
  # puts app_train["data"][0]["features"]["name_type_suite"].class
  # puts app_train["data"].select {|r| r["features"]["name_type_suite"].class != String}
end