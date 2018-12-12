require "csv"

def read_dataset_from_csv(prefix, num_rows)
  # features = ['EXT_SOURCE_3', 'DAYS_BIRTH', 'DAYS_ID_PUBLISH', 'DAYS_REGISTRATION', 'DAYS_EMPLOYED', 'AMT_ANNUITY',
  #   'DAYS_LAST_PHONE_CHANGE', 'AMT_CREDIT', 'EXT_SOURCE_1', 'AMT_INCOME_TOTAL', 'REGION_POPULATION_RELATIVE',
  #   'AMT_GOODS_PRICE', 'HOUR_APPR_PROCESS_START']
  data = []
  classes = Hash.new {|h,k| h[k] = 0}
  header = []
  File.open(prefix + ".csv").each_line.with_index do |l, i|
    if (i == 0)
      header = CSV.parse(l)[0]
    else
      
      if num_rows != nil 
        next if i > num_rows
      end

      a = CSV.parse(l)[0]
      next if a.empty?

      row = {"features" => Hash.new}
      header.each.with_index do |k, i|
        v = a[i]
        if k == "TARGET"
          row["label"] = v.to_f
        elsif k == "SK_ID_CURR" 
          row["id"] = v.to_f
        else
          # TODO: delete
          # if !features.include?(k)
          #   next
          # end

          # original v is always a string
          row["features"][k.downcase] = is_number?(v) ? v.to_f : v.to_s
        end
      end
      row["features"]["bias"] = 1.0
      data << row
    end
  end
  header << "bias"
  return {"classes" => classes, "features" => header[2, header.size - 1].map {|x| x.downcase}, "data" => data}
  # return {"classes" => classes, "features" => features, "data" => data}
end

# helper functions 

def is_number? string
  true if Float(string) rescue false
end