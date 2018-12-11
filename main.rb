def read_dataset_from_csv(prefix)
  data = []
  classes = Hash.new {|h,k| h[k] = 0}
  header = []
  File.open(prefix + ".csv").each_line.with_index do |l, i|
    if (i == 0)
      header = l.chomp.split(",")
    else
      a = l.chomp.split ","
      next if a.empty?
      row = {"features" => Hash.new}
      header.each.with_index do |k, i|
        v = a[i].to_f
        if k == "label"
          row["label"] = v.to_f
        else
          next if v.zero?
          row["features"][k] = v
        end
      end
      data << row
    end
  end
  return {"classes" => classes, "features" => header[0,header.size - 1], "data" => data}
end

if __FILE__ == $0
  app_train = read_dataset_from_csv('./input/application_train')
  puts app_train["data"][0]
  puts "Fuck world!"
end